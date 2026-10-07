# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deep-trace hang test + bitwise glitch stress for the RING decode MoE: MIMO_NLAY (48 for the hang gate, 6 for the glitch stress) chained MiMoRingMoEDecode calls (real layers 1,5,16 reused
round-robin, shared ring buffers/semaphores as in the model) in ONE trace, T = MIMO_T tokens per mesh row (MIMO_LIVE live, rest dead), MIMO_REPS replays; with MIMO_COMPARE=1 every
layer output of every replay is compared bitwise with the first replay (per mesh row) and each glitch is printed. A hang shows as a stalled log."""
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tt.moe_prefill import COLS, HIDDEN, ROWS
from models.demos.blackhole.mimo_v26.tt.moe_ring import MiMoRingMoEDecode, RingExpertWeights
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate

D1 = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
D2 = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-device-s128b16"
E = os.environ.get
NLAY, REPS, T = int(E("MIMO_NLAY", "48")), int(E("MIMO_REPS", "1000")), int(E("MIMO_T", "8"))
LIVE = int(E("MIMO_LIVE", str(T)))
COMPARE = E("MIMO_COMPARE", "0") == "1"
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
def test_moe_ring_deep(mesh_device):
    md = mesh_device
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    dec, xs = {}, {}
    sel = torch.arange(0, 512, 16)[:32]
    for L in LAYERS:
        a = torch.load(f"{D1 if L <= 5 else D2}/layer_{L}.pt", mmap=True)["prefill"]
        x = a["ffn_in"].reshape(ROWS, 512, HIDDEN)[:, sel][:, :T].reshape(ROWS, 1, T, HIDDEN).clone()
        x[:, :, LIVE:] = 0
        gw, gb = load_gate(L)
        gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=None)
        for n in ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices"):
            ttnn.deallocate(getattr(gate, n))
        dec[L] = MiMoRingMoEDecode(md, gate, RingExpertWeights(md, L), router="fast")
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

    print(f"RESULT config: ring nlay={NLAY} T={T} live={LIVE} reps={REPS} compare={COMPARE}", flush=True)
    o = run_all()
    ttnn.synchronize_device(md)
    print("RESULT eager ok", flush=True)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    outs = run_all()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    print("RESULT trace captured", flush=True)
    t0 = time.time()
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    print(f"RESULT first replay ok ({time.time() - t0:.2f}s)", flush=True)

    def grab():
        return [
            torch.stack(
                [
                    ttnn.to_torch(ttnn.get_device_tensors(t)[r * COLS]).reshape(-1, HIDDEN).float()[:LIVE]
                    for r in range(ROWS)
                ]
            )
            for t in outs
        ]

    glitches = []
    if COMPARE:
        ref = grab()
        t0 = time.time()
        for rep in range(REPS):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            cur = grab()
            for li, (c, r) in enumerate(zip(cur, ref)):
                if not torch.equal(c, r):
                    diff = (c - r).abs()
                    g = {
                        "rep": rep,
                        "layer_pos": li,
                        "rows": [i for i in range(diff.shape[0]) if diff[i].max() > 0],
                        "maxdiff": float(diff.max()),
                    }
                    glitches.append(g)
                    print("RESULT GLITCH", json.dumps(g), flush=True)
        print(
            f"RESULT glitch stress: ring T={T} {NLAY} layers x {REPS} replays = {NLAY * REPS} layer-calls in {time.time() - t0:.0f}s: glitching layer-calls {len(glitches)}",
            flush=True,
        )
    else:
        t0 = time.time()
        for i in range(REPS):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            if i % 10 == 9:
                ttnn.synchronize_device(md)
        ttnn.synchronize_device(md)
        dt = (time.time() - t0) / REPS * 1e3
        print(
            f"RESULT {REPS} replays ok: {dt:.2f} ms per replay of {NLAY} layers = {dt / NLAY:.3f} ms/layer", flush=True
        )
    assert not glitches
