# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paired in-process A/B of decode MoE variants (MiMoMoEDecode router exact|fast, capacity factor) on 4x8 with REAL layers (MIMO_LAYERS default 1,5 + 2,3,4).
Per variant: eager PCC vs dump ffn_out (live tokens) at 32 and 8 users/row, 5 eager runs bit-identical, trace == eager bit-identical, expert-set identity of the
router vs the exact router on the live slots; then ALTERNATING trace replays (3 rounds x MIMO_REPLAYS=30 per variant, one process, host-timed hint).
Variants: MIMO_VARIANTS="exact:4,fast:4,fast:2,fast:1" (router:cf). Needs patched _ttnncpp.so + MOE_UNI_BF16_* = 1."""
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

DUMP = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
LAYERS = [int(v) for v in os.environ.get("MIMO_LAYERS", "1,2,3,4,5").split(",")]
VARIANTS = [
    (v.split(":")[0], int(v.split(":")[1]))
    for v in os.environ.get("MIMO_VARIANTS", "exact:4,fast:4,fast:2,fast:1").split(",")
]
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
                "trace_region_size": 400_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_moe_decode_overhead(mesh_device):
    md = mesh_device
    dec, xdev, refs, xs = {}, {}, {}, {}
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    sel = torch.arange(0, 512, 16)[:NSLOT]
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
    d0 = dec[LAYERS[0]]
    mask_dev = d0.alloc_mask(d0.make_mask_host([NSLOT] * ROWS))

    def set_users(n):
        d0.update_mask(mask_dev, d0.make_mask_host([n] * ROWS))
        for L in LAYERS:
            x = xs[L].clone()
            x[:, :, n:] = 0
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=m2), xdev[L]
            )

    def to_host(t):
        d = ttnn.get_device_tensors(t)
        return torch.stack([ttnn.to_torch(d[r * COLS]).reshape(NSLOT, HIDDEN).float() for r in range(ROWS)])

    def configure(router, cf):
        for L in LAYERS:
            dec[L].router = router
            dec[L].moe.cf = cf
            dec[L].moe._shared = {}

    def run_all():
        return [dec[L].forward(xdev[L], mask_dev) for L in LAYERS]

    traces, base_pcc, failures = {}, {}, []
    for router, cf in VARIANTS:
        name = f"{router}:cf{cf}"
        configure(router, cf)
        res = {}
        for n in (32, 8):
            set_users(n)
            outs = run_all()
            ttnn.synchronize_device(md)
            h = [to_host(o) for o in outs]
            res[n] = [[round(pcc(h[i][r, :n], refs[L][r, :n]), 5) for r in range(ROWS)] for i, L in enumerate(LAYERS)]
            if n == 32:
                eager32 = h
                ident = True
                for o in outs:
                    ttnn.deallocate(o)
                for _ in range(4):
                    o2 = run_all()
                    ttnn.synchronize_device(md)
                    ident &= all(bool(torch.equal(to_host(o), e)) for o, e in zip(o2, eager32))
                    for o in o2:
                        ttnn.deallocate(o)
            else:
                for o in outs:
                    ttnn.deallocate(o)
        set_users(32)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        t_outs = run_all()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        same = all(bool(torch.equal(to_host(o), e)) for o, e in zip(t_outs, eager32))
        traces[name] = tid
        for i, L in enumerate(LAYERS):
            print(f"RESULT {name} L{L}: PCC vs dump live per row users=32 {res[32][i]} users=8 {res[8][i]}", flush=True)
        print(f"RESULT {name}: 5 eager runs bit-identical {ident}; trace == eager {same}", flush=True)
        if not base_pcc:
            base_pcc = res
        else:
            worse = [
                (L, n) for n in (32, 8) for i, L in enumerate(LAYERS) if min(res[n][i]) < min(base_pcc[n][i]) - 5e-4
            ]
            if worse:
                failures.append((name, "PCC worse than baseline", worse))
        if not (ident and same):
            failures.append((name, "determinism/trace", ident, same))
    # router set identity fast vs exact on the live slots (eager, routes only)
    set_users(32)
    for L in LAYERS:
        g = dec[L].gate
        xp = ttnn.pad(xdev[L], [(0, 0), (0, 0), (0, SLOTS - NSLOT), (0, 0)], 0.0)
        we, ie = g.route(xp, "exact")
        wf, i_f = g.route_v(xdev[L], "sel_mm1x8")
        he = torch.stack(
            [
                ttnn.to_torch(ttnn.get_device_tensors(ie)[r * COLS]).reshape(SLOTS, -1).long()[:NSLOT]
                for r in range(ROWS)
            ]
        ).reshape(-1, 8)
        hf = torch.stack(
            [ttnn.to_torch(ttnn.get_device_tensors(i_f)[r * COLS]).reshape(NSLOT, -1).long() for r in range(ROWS)]
        ).reshape(-1, 8)
        print(
            f"RESULT L{L} expert sets fast == exact on live slots: {float((he.sort(-1).values == hf.sort(-1).values).all(-1).float().mean()):.4f}",
            flush=True,
        )
    # alternating timing
    names = list(traces)
    tm = {n: [] for n in names}
    for rnd in range(3):
        for n in names:
            tid = traces[n]
            for _ in range(5):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            t0 = time.time()
            for _ in range(REPLAYS):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            tm[n].append((time.time() - t0) / REPLAYS * 1e3 / len(LAYERS))
    for n in names:
        print(
            f"RESULT TIME {n}: {sorted(tm[n])[1]:.3f} ms/layer (median of 3 alternating rounds {['%.3f' % v for v in tm[n]]}); x47 = {sorted(tm[n])[1] * 47:.1f} ms",
            flush=True,
        )
    assert not failures, failures
