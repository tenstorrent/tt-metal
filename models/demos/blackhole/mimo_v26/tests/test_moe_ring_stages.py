# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-cumulative traced times of MiMoRingMoEDecode (real layer MIMO_LAYER=5, T in MIMO_TVALS, MIMO_LIVE_FRAC live): route+dead-slot handling, +dispatch, +moe_compute(+combine), +tilize/fast_reduce(+reduce_scatter in
upto=99 is split below), +reduce_scatter+all_gather. 3 layers' worth of calls per trace (the same layer 3x) to amortise trace overhead."""
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
LAYER = int(os.environ.get("MIMO_LAYER", "5"))
TVALS = [int(v) for v in os.environ.get("MIMO_TVALS", "1,8,32").split(",")]
FRAC = float(os.environ.get("MIMO_LIVE_FRAC", "0.75"))
REPLAYS = 30
REP = 3


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_moe_ring_stages(mesh_device):
    md = mesh_device
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    a = torch.load(f"{D1}/layer_{LAYER}.pt", mmap=True)["prefill"]
    sel = torch.arange(0, 512, 16)[:32]
    xs = a["ffn_in"].reshape(ROWS, 512, HIDDEN)[:, sel].clone()
    gw, gb = load_gate(LAYER)
    gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=None)
    for n in ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices"):
        ttnn.deallocate(getattr(gate, n))
    dec = MiMoRingMoEDecode(md, gate, RingExpertWeights(md, LAYER), router="fast")
    names = {
        0: "router+dead-slot routing",
        1: "+dispatch",
        2: "+moe_compute(+combine)",
        3: "+tilize+fast_reduce",
        99: "+reduce_scatter+all_gather",
    }
    for T in TVALS:
        nlive = max(1, int(round(T * FRAC)))
        x = xs[:, :T].clone()
        x[:, nlive:] = 0
        xd = ttnn.from_torch(
            x.reshape(ROWS, 1, T, HIDDEN),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m2,
        )
        mask = dec.alloc_mask(dec.make_mask_host([nlive] * ROWS))
        prev, row = 0.0, []
        for st in (0, 1, 2, 3, 99):
            fn = lambda st=st: [dec.forward(xd, mask, upto=st) for _ in range(REP)]
            fn()
            ttnn.synchronize_device(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            fn()
            ttnn.end_trace_capture(md, tid, cq_id=0)
            for _ in range(5):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            t0 = time.time()
            for _ in range(REPLAYS):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            v = (time.time() - t0) / REPLAYS * 1e3 / REP
            ttnn.release_trace(md, tid)
            row.append(f"{names[st]} {v - prev:+.3f}")
            prev = v
        print(
            f"RESULT STAGES ring T={T} live={nlive} L{LAYER} (ms/layer deltas, total {prev:.3f}): " + "; ".join(row),
            flush=True,
        )
