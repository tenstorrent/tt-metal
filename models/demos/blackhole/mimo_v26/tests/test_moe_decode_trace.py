# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode-sized MoE in a TRACE on 4x8 with REAL layers (MIMO_LAYERS, default 1..5: real gate, bias, 256 real experts bfp8, real dump hidden states):
MiMoMoEDecode (exact router + spread dead slots + dispatch + unified bf16-activation op + combine + all-reduce). Needs the patched _ttnncpp.so
(LD_LIBRARY_PATH) and MOE_UNI_BF16_X/INTERMED/OUT=1. Reports: eager vs trace PCC vs dump ffn_out (live tokens), eager==trace bit identity, per-layer traced time
at users/row 1,8,16,32 (host-timed over MIMO_REPLAYS=30 warm replays of ONE trace holding all layers; hint), the same for the router alone, 47-layer estimate.
"""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tests.test_moe_prefill_4x8 import pcc
from models.demos.blackhole.mimo_v26.tt.moe_decode import SLOTS, MiMoMoEDecode
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

DUMP = os.environ.get("MIMO_DUMP", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16")
LAYERS = [int(v) for v in os.environ.get("MIMO_LAYERS", "1,2,3,4,5").split(",")]
USERS = [int(v) for v in os.environ.get("MIMO_USERS", "1,8,16,32").split(",")]
REPLAYS = int(os.environ.get("MIMO_REPLAYS", "30"))
NSLOT = 32


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 300_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_moe_decode_trace(mesh_device):
    md = mesh_device
    dec, xdev, refs, xs = {}, {}, {}, {}
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    sel = torch.arange(0, 512, 16)[:NSLOT]  # 32 tokens spread over the 4 users of a mesh row
    for L in LAYERS:
        a = torch.load(f"{DUMP}/layer_{L}.pt", mmap=True)["prefill"]
        xs[L] = a["ffn_in"].reshape(ROWS, 512, HIDDEN)[:, sel].reshape(ROWS, 1, NSLOT, HIDDEN)
        refs[L] = a["ffn_out"].reshape(ROWS, 512, HIDDEN)[:, sel].float()
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
        dec[L] = MiMoMoEDecode(md, gate, moe)
        xdev[L] = ttnn.from_torch(
            xs[L],
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m2,
        )
        print(f"RESULT layer {L} built", flush=True)
    d0 = dec[LAYERS[0]]

    def host_x(L, n):
        x = xs[L].clone()
        x[:, :, n:] = 0
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=m2)

    def to_host(t):
        d = ttnn.get_device_tensors(t)
        return torch.stack([ttnn.to_torch(d[r * COLS]).reshape(NSLOT, HIDDEN).float() for r in range(ROWS)])

    mask_dev = d0.alloc_mask(d0.make_mask_host([NSLOT] * ROWS))

    def set_users(n):
        d0.update_mask(mask_dev, d0.make_mask_host([n] * ROWS))
        for L in LAYERS:
            ttnn.copy_host_to_device_tensor(host_x(L, n), xdev[L])

    def run_all():
        return [dec[L].forward(xdev[L], mask_dev) for L in LAYERS]

    # eager correctness
    for n in (32, 8):
        set_users(n)
        outs = run_all()
        ttnn.synchronize_device(md)
        for L, o in zip(LAYERS, outs):
            h = to_host(o)
            print(
                f"RESULT eager L{L} users/row={n}: PCC vs dump ffn_out (live tokens) per mesh row {[round(pcc(h[r, :n], refs[L][r, :n]), 5) for r in range(ROWS)]}",
                flush=True,
            )
        eager = [to_host(o) for o in outs]
        for o in outs:
            ttnn.deallocate(o)
    set_users(32)
    eager32 = None
    outs = run_all()
    ttnn.synchronize_device(md)
    eager32 = [to_host(o) for o in outs]
    for o in outs:
        ttnn.deallocate(o)

    # trace
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    t_outs = run_all()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    same = [bool(torch.equal(to_host(o), e)) for o, e in zip(t_outs, eager32)]
    print(f"RESULT trace output bit-identical to eager (users/row 32) per layer: {same}", flush=True)
    res = {}
    for n in USERS:
        set_users(n)
        for _ in range(5):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        t0 = time.time()
        for _ in range(REPLAYS):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        dt = (time.time() - t0) / REPLAYS * 1e3
        res[n] = dt / len(LAYERS)
        h = [to_host(o) for o in t_outs]
        pc = [min(round(pcc(h[i][r, :n], refs[L][r, :n]), 5) for r in range(ROWS)) for i, L in enumerate(LAYERS)]
        print(
            f"RESULT TRACE users/row={n}: {dt:.2f} ms per replay of {len(LAYERS)} layers = {res[n]:.2f} ms/layer (host-timed, {REPLAYS} warm replays); x47 layers = {res[n] * 47:.0f} ms; min PCC vs dump per layer {pc}",
            flush=True,
        )
    ttnn.release_trace(md, tid)

    # router alone in a trace (T=64 rows incl. pad rows, as used by MiMoMoEDecode)
    def route_all():
        outs = []
        for L in LAYERS:
            xp = ttnn.pad(xdev[L], [(0, 0), (0, 0), (0, SLOTS - NSLOT), (0, 0)], 0.0)
            outs.append(dec[L].gate.route(xp, "exact"))
        return outs

    route_all()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    ro = route_all()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    for _ in range(5):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    t0 = time.time()
    for _ in range(REPLAYS):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    dt = (time.time() - t0) / REPLAYS * 1e3
    print(
        f"RESULT TRACE exact router alone (64 rows): {dt / len(LAYERS):.3f} ms/layer (host-timed); x47 = {dt / len(LAYERS) * 47:.1f} ms",
        flush=True,
    )
    ttnn.release_trace(md, tid)
