# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated swiglu-epilogue bake-off, round 2 (SFPLOADMACRO / instruction-level SwiGluMul).

    SWG_SET=focus scripts/run_safe_pytest.sh --run-all --profile <this file>
Each case appends a JSON line (config + precision) to results_<set>.jsonl in the experiment dir, in
execution order; the profiler's ops CSV rows (one GenericOp per case, same order) carry the ns.
Correctness gate: PCC vs fp64 torch silu(g) * u from the stored inputs > 0.999 -- perf is not asserted.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import ttnn

EXP_DIR = (
    Path(__file__).resolve().parents[7]
    / "ttnn/ttnn/operations/genify_matmul/perf_experiments/swiglu_epilogue_loadmacro"
)
_spec = importlib.util.spec_from_file_location("swg_lm_bench", EXP_DIR / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

SET = os.environ.get("SWG_SET", "focus")
F = dict(block_m=6, block_n=28)  # the focus block (R1a, 6 x 28 B-tiles -> 6 x 14 outputs)
NOGATE = ("floor", "sfpu_x2", "nosfpu_store")

_VARS = os.environ.get("SWG_VARIANTS", "")
CASES = {
    "probe": [
        dict(F, variant=v, **json.loads(os.environ.get("SWG_KW", "{}")))
        for v in (_VARS.split(",") if _VARS else ["baseline"])
    ],
    "binder": [dict(F, variant=v) for v in ("baseline", "floor", "sfpu_x2", "nosfpu_store")],
    "focus": [dict(F, variant=v) for v in ("baseline", "lm", "tuned_k2c", "tuned")],
    "focus_rev": [dict(F, variant=v) for v in ("tuned", "tuned_k2c", "lm", "baseline")],
    "fp32": [dict(F, variant=v, out_dtype="fp32") for v in ("baseline", "lm", "tuned")],
    "stress": [
        dict(F, variant=v, scale=sc, shift=sh)
        for sc, sh in ((8.0, -4.0), (60.0, 0.0))
        for v in (_VARS.split(",") if _VARS else ["baseline", "lm2"])
    ],
}


def _domain(cand):
    return [
        dict(F, variant="baseline"),
        dict(F, variant=cand),
        dict(F, variant=cand, natural_pairs=True),
        dict(F, variant="baseline", natural_pairs=True),
        dict(F, variant="baseline", fp32_dest=False),
        dict(F, variant=cand, fp32_dest=False),
        dict(F, variant=cand, out_dtype="bf8"),
        dict(F, variant="baseline", out_dtype="bf8"),
        dict(F, variant="baseline", out_dtype="fp32"),
        dict(F, variant=cand, out_dtype="fp32"),
        dict(block_m=2, block_n=40, variant=cand),
        dict(block_m=2, block_n=40, variant="baseline"),
        dict(F, variant="baseline", unpack_to_dest=True),
        dict(F, variant=cand, unpack_to_dest=True),
    ]


CASES["domain"] = _domain(os.environ.get("SWG_CAND", "lm"))
# the graduation header (drop-in lm2 SwiGluMul) on every config it must cover; correctness + ns
CASES["grad"] = [
    dict(F, variant="grad"),
    dict(F, variant="grad", out_dtype="fp32"),
    dict(F, variant="grad", fp32_dest=False),
    dict(F, variant="grad", natural_pairs=True),
    dict(F, variant="grad", out_dtype="bf8"),
    dict(F, variant="grad", unpack_to_dest=True),
    dict(F, variant="grad", scale=60.0),
    dict(F, variant="grad", out_dtype="fp32", scale=60.0),
    dict(block_m=1, block_n=2, variant="grad"),
]


def _debug_map(got, ref, bm, half):
    """Bad elements (> 4 bf16 ULP) by in-tile SFPU row (face*8 + (i%16)//4*2 + j%2) and by output tile."""
    g, r = got.double(), ref.double()
    ulp = torch.pow(2.0, torch.floor(torch.log2(r.abs().clamp_min(1e-30))) - 7)
    bad = ((g - r).abs() / ulp > 4) & (r.abs() > 1e-30)
    i = torch.arange(bm * 32).view(-1, 1) % 32
    j = torch.arange(half * 32).view(1, -1) % 32
    face = (i // 16) * 2 + (j // 16)
    srow = face * 8 + ((i % 16) // 4) * 2 + (j % 2)
    srow = srow.expand_as(bad)
    by_row = [int(bad[srow == k].sum()) for k in range(32)]
    tiles = bad.view(bm, 32, half, 32).sum(dim=(1, 3))
    print("SWG_DEBUG bad total", int(bad.sum()), "of", bad.numel())
    print("SWG_DEBUG by sfpu row", by_row)
    print("SWG_DEBUG by tile", tiles.tolist())
    idx = bad.nonzero()[:8].tolist()
    for a, b in idx:
        print("SWG_DEBUG sample", a, b, float(g[a, b]), float(r[a, b]))


def _cases():
    return CASES[SET]


@pytest.mark.parametrize("idx", range(len(_cases())))
def test_swiglu_lm(device, idx):
    case = dict(_cases()[idx])
    case.setdefault("out_dtype", "bf16")
    case.setdefault("fp32_dest", True)
    case.setdefault("natural_pairs", False)
    case.setdefault("block_size", 4)
    case.setdefault("unpack_to_dest", False)
    out_dtype = {"fp32": ttnn.float32, "bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[case["out_dtype"]]
    bm, bn = case["block_m"], case["block_n"]
    host, _, _ = bench.make_block(
        bm,
        bn,
        case["natural_pairs"],
        seed=case.get("seed", 0),
        scale=case.get("scale", 3.0),
        shift=case.get("shift", 0.0),
    )
    kw = {k: v for k, v in case.items() if k not in ("out_dtype", "block_m", "block_n", "variant", "scale", "shift")}
    out = bench.run(device, host, block_m=bm, block_n=bn, variant=case["variant"], out_dtype=out_dtype, **kw)
    got = ttnn.to_torch(out).float()
    stored = host if case["fp32_dest"] else host.to(torch.bfloat16).float()
    half = bn // 2
    t = stored.double().reshape(bm * 32, bn, 32)
    if case["natural_pairs"]:
        gate, up = t[:, 0::2, :], t[:, 1::2, :]
    else:
        gate, up = t[:, :half, :], t[:, half:, :]
    ref = (torch.nn.functional.silu(gate) * up).reshape(bm * 32, half * 32)
    m = bench.metrics(got, ref) if case["variant"] not in NOGATE else {}
    if os.environ.get("SWG_SAVE"):
        torch.save(got, EXP_DIR / f"out_{SET}_{idx}.pt")
    if os.environ.get("SWG_DEBUG"):
        _debug_map(got, ref, bm, half)
    rec = dict(set=SET, idx=idx, **case, **m)
    with open(EXP_DIR / f"results_{SET}.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("SWG_RESULT", json.dumps(rec))
    if case["variant"] not in NOGATE:
        assert m["pcc"] > 0.999, m
