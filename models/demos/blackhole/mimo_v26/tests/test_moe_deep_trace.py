# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deep-trace hang repro for the MoE decode path alone: chain MIMO_NLAY (24/48) MiMoMoEDecode calls (3 real layers 1,5,16 reused round-robin) in ONE trace, replay MIMO_REPS times.
Knobs: MIMO_DEAD=sentinel|spread, MIMO_U (tokens per row, MIMO_LIVE live of them), MIMO_MOE_AR=ag_sum|rs, MOE_DISPATCH_SPARSE_MCAST, MIMO_ROUTER, MIMO_DISP_WORKERS/LINKS.
MIMO_FIXED_DEAD=<expert id>: dead slots routed to that single expert (weight 0) instead of the sentinel. A hang shows as a stalled log (launch.py hangwatch triages)."""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tt.moe_decode import SLOTS, MiMoMoEDecode
from models.demos.blackhole.mimo_v26.tt.moe_prefill import (
    COLS,
    HIDDEN,
    ROWS,
    TOPK,
    MiMoPrefillMoE,
    build_expert_weights,
    make_ckc,
)
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate

D1 = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
D2 = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-device-s128b16"
E = os.environ.get
NLAY, REPS = int(E("MIMO_NLAY", "24")), int(E("MIMO_REPS", "50"))
U, LIVE = int(E("MIMO_U", "32")), int(E("MIMO_LIVE", E("MIMO_U", "32")))
DEAD, ROUTER = E("MIMO_DEAD", "sentinel"), E("MIMO_ROUTER", "fast")
LAYERS = [1, 5, 16]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 600_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_moe_deep_trace(mesh_device):
    md = mesh_device
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    dec, xs = {}, {}
    sel = torch.arange(0, 512, 16)[:32]
    for L in LAYERS:
        a = torch.load(f"{D1 if L <= 5 else D2}/layer_{L}.pt", mmap=True)["prefill"]
        x = a["ffn_in"].reshape(ROWS, 512, HIDDEN)[:, sel][:, :U].reshape(ROWS, 1, U, HIDDEN).clone()
        x[:, :, LIVE:] = 0
        W = load_layer_experts(L)
        wts = build_expert_weights(md, lambda e: W[e], nd_sharded=True)
        del W
        gw, gb = load_gate(L)
        gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=None)
        for n in ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices"):
            ttnn.deallocate(getattr(gate, n))
        moe = MiMoPrefillMoE(
            md, wts, ckc=make_ckc("hifi2", fp32acc=True, packer_l1_acc=True), expert_op="unified", cf=4
        )
        dec[L] = MiMoMoEDecode(md, gate, moe, router=ROUTER, dead=DEAD)
        if E("MIMO_FIXED_DEAD"):
            fid = int(E("MIMO_FIXED_DEAD"))
            rep = ttnn.ReplicateTensorToMesh(md)
            old = dec[L].sp_idx
            dec[L].sp_idx = ttnn.from_torch(
                torch.full((1, 1, SLOTS, TOPK), fid).bfloat16(),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=rep,
            )
        xs[L] = ttnn.from_torch(
            x,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m2,
        )
    d0 = dec[LAYERS[0]]
    mask = d0.alloc_mask(d0.make_mask_host([LIVE] * ROWS))
    order = [LAYERS[i % len(LAYERS)] for i in range(NLAY)]

    def run_all():
        return [dec[L].forward(xs[L], mask) for L in order]

    print(
        f"RESULT config: nlay={NLAY} U={U} live={LIVE} dead={DEAD} router={ROUTER} ar={E('MIMO_MOE_AR', 'ag_sum')} sparse_mcast={E('MOE_DISPATCH_SPARSE_MCAST', 'on')} fixed_dead={E('MIMO_FIXED_DEAD')}",
        flush=True,
    )
    outs = run_all()
    ttnn.synchronize_device(md)
    print("RESULT eager ok", flush=True)
    for o in outs:
        ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    outs = run_all()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    print("RESULT trace captured", flush=True)
    t0 = time.time()
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    print(f"RESULT first replay ok ({time.time() - t0:.2f}s)", flush=True)
    t0 = time.time()
    for i in range(REPS):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        if i % 10 == 9:
            ttnn.synchronize_device(md)
    ttnn.synchronize_device(md)
    dt = (time.time() - t0) / REPS * 1e3
    print(f"RESULT {REPS} replays ok: {dt:.2f} ms per replay of {NLAY} layers = {dt / NLAY:.3f} ms/layer", flush=True)
