# SPDX-License-Identifier: Apache-2.0
"""Prefill scale / memory-fit runs on the 4x8 mesh: real GSM8K text concatenated to the requested ISL, chunked traced prefill, several ISLs in ONE process.
Env: PS_LAYERS (4), PS_CH (2048), PS_CASES "B:ISL,B:ISL,..." (B total users spread over the 4 mesh rows, e.g. "4:65536,4:8192,4:65536"), PS_OUT, MIMO_EXPERT_OP.
Logs per case: wall, replay step times, prefill tokens/s (real tokens / sum of replay times), DRAM per bank (ttnn memory view: allocated/free per bank, 8 banks per chip),
first-token ids + finite check. DRAM is also logged after the model build and after the first full case; for repeated ISLs the free DRAM must be unchanged."""
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel
from models.demos.blackhole.mimo_v26.tt.prefill import ChunkedPrefill

E = os.environ.get
NL = int(E("PS_LAYERS", "4"))
CH = int(E("PS_CH", "2048"))
OUT = E("PS_OUT", "/tmp/ps.jsonl")
CASES = [tuple(int(v) for v in c.split(":")) for c in E("PS_CASES", "4:4096").split(",")]
GSM = "/mnt/tt-data/ssinghal/mimo-dumps/gsm8k/gsm8k_512.json"


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("PS_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


def dram(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    mb = lambda b: round(b / 2**20, 1)
    return dict(
        alloc_mb_per_bank=mb(v.total_bytes_allocated_per_bank),
        free_mb_per_bank=mb(v.total_bytes_free_per_bank),
        banks=v.num_banks,
    )


def text_tokens(stream, n, seed):
    off = (seed * 7919) % len(stream)
    s = stream[off:] + stream[:off]
    reps = -(-n // len(s))
    return (s * reps)[:n]


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
def test_prefill_scale(mesh_device):
    md = mesh_device
    stream = [t for it in json.load(open(GSM))["items"] for t in it["prompt_ids"]]
    umax = max(-(-b // 4) for b, _ in CASES)
    isl_max = max(i for _, i in CASES)
    lay = KVLayout(users=umax, max_pos=-(-(isl_max + 8) // CH) * CH, chunk=CH)
    rec(
        "env",
        layers=NL,
        chunk=CH,
        cases=CASES,
        users_per_row=umax,
        max_pos=lay.max_pos,
        kv_gib_per_chip_arith=round(
            lay.bytes_per_chip(
                n_ga=sum(1 for L in range(NL) if L in (0, 5, 11, 17, 23, 29, 35, 41, 47)),
                n_swa=sum(1 for L in range(NL) if L not in (0, 5, 11, 17, 23, 29, 35, 41, 47)),
            )
            / 2**30,
            3,
        ),
        dram_before=dram(md),
    )
    t0 = time.time()
    model = MiMoModel(md, n_layers=NL, layout=lay, log=lambda s: print(s, flush=True))
    rec("build", seconds=round(time.time() - t0, 1), dram_after_build=dram(md))
    pf = ChunkedPrefill(model, trace=True)
    for ci, (B, isl) in enumerate(CASES):
        upr = -(-B // 4)
        prompts = [
            [text_tokens(stream, isl, seed=r * 8 + u + ci) for u in range(upr) if r * upr + u < B] for r in range(4)
        ]
        n0 = len(pf.times)
        t0 = time.time()
        res = pf.prefill(prompts)
        wall = time.time() - t0
        st = pf.times[n0:]
        toks = sum(len(p) for row in prompts for p in row)
        ft = pf.first_tokens(res["last"])
        fin = all(bool(torch.isfinite(v[1]).all()) for v in ft.values())
        rec(
            "prefill_case",
            case_idx=ci,
            B=B,
            isl=isl,
            steps=res["steps"],
            wall_s=round(wall, 2),
            replay_s=round(sum(st) / 1e3, 3),
            step_ms_first=round(st[0], 1),
            step_ms_median=round(sorted(st)[len(st) // 2], 1),
            step_ms_last=round(st[-1], 1),
            tok_per_s_replay=round(toks / (sum(st) / 1e3), 1),
            tok_per_s_wall=round(toks / wall, 1),
            finite=fin,
            first_tokens=[v[0] for v in list(ft.values())[:8]],
            dram_after=dram(md),
        )
        assert fin
    pf.release()
    rec("final_dram", dram=dram(md))
