# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Free DRAM per chip with BOTH expert copies after prefill, per (B, ISL) cell (one process per cell, PS_CASES="B:ISL", PS_LAYERS=48): build the model (unified per-expert copy + attention + KV pool sized to the cell),
run the chunked traced prefill of the cell (as tests/test_prefill_scale.py), then build the ring-layout copy of all MoE layers from the NFS cache and read the ttnn DRAM memory view. Reports free GiB per chip after
prefill (one copy) and after the ring copy, and the ring_enabled() decision (>= 1 GiB headroom)."""
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_prefill_scale import CASES, CH, GSM, NL, dram, text_tokens
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel
from models.demos.blackhole.mimo_v26.tt.moe_ring import RingExpertWeights, ring_enabled
from models.demos.blackhole.mimo_v26.tt.prefill import ChunkedPrefill

OUT = os.environ.get("PS_OUT", "/tmp/ring_dram.jsonl")


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("RD_RESULT " + s, flush=True)
    open(OUT, "a").write(s + "\n")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "trace_region_size": 1536 * 1024 * 1024,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_ring_dram_cells(mesh_device):
    md = mesh_device
    stream = [t for it in json.load(open(GSM))["items"] for t in it["prompt_ids"]]
    B, isl = CASES[0]
    upr = -(-B // 4)
    lay = KVLayout(users=upr, max_pos=-(-(isl + 8) // CH) * CH, chunk=CH)
    model = MiMoModel(md, n_layers=NL, layout=lay, log=lambda s: None)
    pf = ChunkedPrefill(model, trace=True)
    prompts = [[text_tokens(stream, isl, seed=r * 8 + u) for u in range(upr) if r * upr + u < B] for r in range(4)]
    pf.prefill(prompts)
    d1 = dram(md)
    gib = lambda d: d["free_mb_per_bank"] * d["banks"] / 1024
    rec("after_prefill_one_copy", B=B, isl=isl, dram=d1, free_gib_per_chip=round(gib(d1), 3))
    ok = ring_enabled(md, layers_to_build=NL - 1, log=lambda s: rec("decision", msg=s))
    t0 = time.time()
    exs = [RingExpertWeights(md, L) for L in range(1, NL)]
    d2 = dram(md)
    rec(
        "after_ring_copy",
        B=B,
        isl=isl,
        dram=d2,
        free_gib_per_chip=round(gib(d2), 3),
        ring_gib_per_chip=round(gib(d1) - gib(d2), 3),
        build_s=round(time.time() - t0, 1),
        decision_ring=ok,
    )
