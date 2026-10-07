# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Race hunt for the rare decode-MoE glitch (LOG E261007-0818-033): chain MIMO_NL real layers (default 6) of MiMoMoEDecode (the decode agent's path: live mask, spread, fast/exact router)
in ONE trace, replay MIMO_REPS times on identical inputs and compare every layer's output bitwise with the first replay, per mesh row (columns identical after the all-reduce).
Inputs: real dump hidden states, distinct per mesh row, MIMO_LIVE live slots per row. Records every glitch (replay, layer, rows, max abs diff). Knobs: MIMO_ROUTER (exact|fast),
MIMO_STAGE (99 = full layer | 5 = per-column partial sums before the all-reduce | 4 = combine output etc: see MiMoMoEDecode.forward upto), env MOE_DISPATCH_SPARSE_MCAST, MIMO_DISP_LINKS/WORKERS.
"""
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tt.moe_decode import MiMoMoEDecode
from models.demos.blackhole.mimo_v26.tt.moe_prefill import (
    COLS,
    HIDDEN,
    ROWS,
    MiMoPrefillMoE,
    build_expert_weights,
    make_ckc,
)
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate

DUMP = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
DDUMP = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-device-s128b16"
NL = int(os.environ.get("MIMO_NL", "6"))
REPS = int(os.environ.get("MIMO_REPS", "3000"))
LIVE = int(os.environ.get("MIMO_LIVE", "8"))
ROUTER = os.environ.get("MIMO_ROUTER", "exact")
STAGE = int(os.environ.get("MIMO_STAGE", "99"))
NSLOT = 32


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 400_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_moe_race_stress(mesh_device):
    md = mesh_device
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    dec, xdev, layers = {}, {}, list(range(1, 1 + NL))
    sel = torch.arange(0, 512, 512 // NSLOT)[:NSLOT]
    for L in layers:
        src = DUMP if L <= 5 else DDUMP
        a = torch.load(f"{src}/layer_{L}.pt", mmap=True)["prefill"]
        x = a["ffn_in"].reshape(ROWS, 512, HIDDEN)[:, sel].reshape(ROWS, 1, NSLOT, HIDDEN).clone()
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
        dec[L] = MiMoMoEDecode(md, gate, moe, router=ROUTER)
        xdev[L] = ttnn.from_torch(
            x,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m2,
        )
    d0 = dec[layers[0]]
    mask_dev = d0.alloc_mask(d0.make_mask_host([LIVE] * ROWS))

    def run_all():
        outs = []
        for L in layers:
            o = dec[L].forward(xdev[L], mask_dev, upto=STAGE)
            outs.append(o)
        return outs

    run_all()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    outs = run_all()
    ttnn.end_trace_capture(md, tid, cq_id=0)

    def grab():
        res = []
        for o in outs:
            d = ttnn.get_device_tensors(o)
            if STAGE == 99:  # [1,1,32,D] per device, columns identical -> read column 0 of every row
                res.append(
                    torch.stack([ttnn.to_torch(d[r * COLS]).reshape(-1, HIDDEN).float()[:LIVE] for r in range(ROWS)])
                )
            else:  # per-column partials [1,1,64,D]: every device
                res.append(
                    torch.stack([ttnn.to_torch(d[i]).reshape(-1, HIDDEN).float()[:LIVE] for i in range(ROWS * COLS)])
                )
        return res

    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    ref = grab()
    glitches = []
    t0 = time.time()
    for rep in range(REPS):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        cur = grab()
        for li, (c, r) in enumerate(zip(cur, ref)):
            if not torch.equal(c, r):
                diff = (c - r).abs()
                bad = [i for i in range(diff.shape[0]) if diff[i].max() > 0]
                glitches.append({"rep": rep, "layer": layers[li], "rows_or_devices": bad, "maxdiff": float(diff.max())})
                print("RESULT GLITCH", json.dumps(glitches[-1]), flush=True)
    print(
        f"RESULT race stress: router={ROUTER} stage={STAGE} layers={layers} live={LIVE} replays={REPS} ({REPS * NL} layer-calls) in {time.time() - t0:.0f}s: "
        f"glitching layer-calls {len(glitches)}; env sparse_mcast={os.environ.get('MOE_DISPATCH_SPARSE_MCAST', 'default on')} links={os.environ.get('MIMO_DISP_LINKS', 2)} workers={os.environ.get('MIMO_DISP_WORKERS', 2)}",
        flush=True,
    )
    assert not glitches
