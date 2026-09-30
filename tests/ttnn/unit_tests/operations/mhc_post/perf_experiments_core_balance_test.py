# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""core_balance bake-off (perf_experiments/core_balance). Baseline = the op's descriptor + kernels (copied
verbatim, `orig`); candidates = the same op with a run-time claimed tail pool (dynamic balancing) or a
position-weighted static split. Run:

  CB_SHAPES=... CB_VARIANTS=... scripts/run_safe_pytest.sh --run-all <this file> -k correct
  CB_SESSION=s1 CB_SHAPES=... CB_VARIANTS=... scripts/run_safe_pytest.sh --profile --run-all <this file> -k perf
  python3 ttnn/ttnn/operations/mhc_post/perf_experiments/core_balance/report.py <cpp_device_perf_report.csv> s1

Every perf test launches exactly ONE device op; its label is appended (in order) to run_order_<session>.jsonl.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import ttnn

REPO = Path(__file__).resolve().parents[5]
EXP_DIR = REPO / "ttnn/ttnn/operations/mhc_post/perf_experiments/core_balance"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXP_DIR / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base_pd = _load("cb_base_pd", "baseline_descriptor.py")
cand_pd = _load("cb_cand_pd", "candidate_descriptor.py")
BC = cand_pd.BalanceConfig

# name -> BalanceConfig (None = the op's own descriptor + kernels, verbatim)
VARIANTS = {
    "orig": None,
    "c0": BC(),  # candidate kernels, no pool (sequence / release plumbing only)
    # pool_mode 1: one global tail pool (the last units of the tensor), chunks claimed by any core
    "g10": BC(pool_mode=1, pool_frac=0.10),
    "g15": BC(pool_mode=1, pool_frac=0.15),
    "g15nh": BC(pool_mode=1, pool_frac=0.15, help_dynamic=False),
    "g15h": BC(pool_mode=1, pool_frac=0.15, chunk_cols=4),
    "g15c3": BC(pool_mode=1, pool_frac=0.15, coef_depth=3),
    # pool_mode 2: per-core tail queues, owner first, then stealing in stride order
    "q10": BC(pool_mode=2, pool_frac=0.10),
    "q20": BC(pool_mode=2, pool_frac=0.20),
    "q30": BC(pool_mode=2, pool_frac=0.30),
    "q50": BC(pool_mode=2, pool_frac=0.50),
    "q20h": BC(pool_mode=2, pool_frac=0.20, chunk_cols=4),
    "q30h": BC(pool_mode=2, pool_frac=0.30, chunk_cols=4),
    "q30nh": BC(pool_mode=2, pool_frac=0.30, help_dynamic=False),
    "q30c3": BC(pool_mode=2, pool_frac=0.30, coef_depth=3),
    "q30hc3": BC(pool_mode=2, pool_frac=0.30, chunk_cols=4, coef_depth=3),
    "q30h1": BC(pool_mode=2, pool_frac=0.30, chunk_cols=4, steal_min=1),
    "q50h": BC(pool_mode=2, pool_frac=0.50, chunk_cols=4),
    "q30s3": BC(pool_mode=2, pool_frac=0.30, chunk_cols=4, steal_min=3),
    "q50s3": BC(pool_mode=2, pool_frac=0.50, chunk_cols=4, steal_min=3),
    # ramped walk: block k is min(B, r0 << k) columns (shrinks the start-up read burst)
    "r2": BC(ramp0=2),
    "r4": BC(ramp0=4),
}
extra = os.environ.get("CB_VARIANTS_EXTRA")  # "name:mode,pool_frac,chunk_cols,coef_depth;..." (chunk 0 = B)
if extra:
    for item in extra.split(";"):
        name, spec = item.split(":")
        md, fr, ch, cd = spec.split(",")
        VARIANTS[name] = BC(pool_mode=int(md), pool_frac=float(fr), chunk_cols=int(ch) or None, coef_depth=int(cd))
for _b in (0.08, 0.16, 0.24, 0.32):
    VARIANTS[f"wy{int(_b*100):02d}"] = BC(weights=cand_pd.row_gradient(_b))  # model: linear in grid row
# the op's own descriptor + kernels, only _work_assignment swapped for the row-weighted drop-in (graduation form)
OP_WEIGHTED = {f"ow{int(_b*100):02d}": _b for _b in (0.08, 0.12, 0.16, 0.20, 0.24)}
for _name in OP_WEIGHTED:
    VARIANTS[_name] = "op_weighted"
# position-weighted static split, calibrated on a profiled `orig` run (per-board; option 2 of the menu)
_CAL = EXP_DIR / "calib_walls.json"
for _a in (0.5, 1.0, 1.5):
    VARIANTS[f"w7168_{_a}"] = BC(weights=cand_pd.wall_calibration(_CAL, "T640_C7168_bf16", _a))
    VARIANTS[f"w4096_{_a}"] = BC(weights=cand_pd.wall_calibration(_CAL, "T1280_C4096_bf16", _a))

bf, f32 = ttnn.bfloat16, ttnn.float32
# id -> (lead, T, C, n, x_dtype, f_dtype)
SHAPES = {
    "T640_C7168_bf16": ((1, 1), 640, 7168, 4, bf, bf),
    "T640_C1792_bf16": ((1, 1), 640, 1792, 4, bf, bf),
    "T1280_C4096_bf16": ((1, 1), 1280, 4096, 4, bf, bf),
    # domain sweep (bf16)
    "T256_C1792_bf16": ((1, 1), 256, 1792, 4, bf, bf),
    "T256_C7168_bf16": ((1, 1), 256, 7168, 4, bf, bf),
    "T512_C5120_bf16": ((1, 1), 512, 5120, 4, bf, bf),
    "T640_C2560_bf16": ((1, 1), 640, 2560, 4, bf, bf),
    "T640_C4096_bf16": ((1, 1), 640, 4096, 4, bf, bf),
    "T1024_C2560_bf16": ((1, 1), 1024, 2560, 4, bf, bf),
    "T1024_C5120_bf16": ((1, 1), 1024, 5120, 4, bf, bf),
    "T1024_C7168_bf16": ((1, 1), 1024, 7168, 4, bf, bf),
    "T1280_C6144_bf16": ((1, 1), 1280, 6144, 4, bf, bf),
    "T2048_C1792_bf16": ((1, 1), 2048, 1792, 4, bf, bf),
    "T2048_C4096_bf16": ((1, 1), 2048, 4096, 4, bf, bf),
    "T2560_C6144_bf16": ((1, 1), 2560, 6144, 4, bf, bf),
    "T4096_C1792_bf16": ((1, 1), 4096, 1792, 4, bf, bf),
    "T4096_C2560_bf16": ((1, 1), 4096, 2560, 4, bf, bf),
    # fp32 / mixed
    "T640_C1792_fp32": ((1, 1), 640, 1792, 4, f32, f32),
    "T640_C7168_fp32": ((1, 1), 640, 7168, 4, f32, f32),
    "T1000_C1792_fp32": ((1, 1), 1000, 1792, 4, f32, f32),
    "T1000_C7168_fp32": ((1, 1), 1000, 7168, 4, f32, f32),
    "T640_C1792_xf32_fbf16": ((1, 1), 640, 1792, 4, f32, bf),
    "T640_C7168_xf32_fbf16": ((1, 1), 640, 7168, 4, f32, bf),
    "T1000_C1792_xf32_fbf16": ((1, 1), 1000, 1792, 4, f32, bf),
    "T1000_C7168_xf32_fbf16": ((1, 1), 1000, 7168, 4, f32, bf),
    "T640_C7168_xbf16_ff32": ((1, 1), 640, 7168, 4, bf, f32),
    # correctness edges
    "T1000_C1792_bf16": ((1, 1), 1000, 1792, 4, bf, bf),
    "T1000_C7168_bf16": ((1, 1), 1000, 7168, 4, bf, bf),
    "T100_C224_fp32": ((1, 1), 100, 224, 4, f32, f32),
    "T17_C128_fp32": ((1, 1), 17, 128, 4, f32, f32),
    "T32_C32_bf16": ((1, 1), 32, 32, 4, bf, bf),
    "T640_C1792_n1_bf16": ((1, 1), 640, 1792, 1, bf, bf),
    "T64_C224_n2_bf16": ((1, 1), 64, 224, 2, bf, bf),  # n=2 only at B=1 (known op compile failure at B>=2)
    "T640_C1792_n3_bf16": ((1, 1), 640, 1792, 3, bf, bf),
    "T640_C1792_n5_bf16": ((1, 1), 640, 1792, 5, bf, bf),
    "T1280_C4096_n3_bf16": ((1, 1), 1280, 4096, 3, bf, bf),
    "T640_C7168_n5_bf16": ((1, 1), 640, 7168, 5, bf, bf),
    "T96_C1184_straddle_bf16": ((1, 1), 96, 1184, 4, bf, bf),
    "B2_T160_C608_straddle_fp32": ((2, 1), 160, 608, 4, f32, bf),
    "B2_T640_C4096_bf16": ((2, 1), 640, 4096, 4, bf, bf),
}

FOCUS = ["T640_C7168_bf16", "T640_C1792_bf16", "T1280_C4096_bf16"]


def _inputs(device, lead, T, C, n, x_dtype, f_dtype, seed=0):
    torch.manual_seed(seed)
    f = torch.randn(*lead, T, C)
    x = torch.randn(*lead, T, n * C)
    post = torch.rand(*lead, T, n) * 2
    comb = torch.rand(*lead, T, n * n)

    def dev(t, dt):
        return ttnn.from_torch(
            t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    tensors = (dev(f, f_dtype), dev(x, x_dtype), dev(post, f32), dev(comb, f32))
    fr = f.bfloat16().float() if f_dtype == bf else f
    xr = x.bfloat16().float() if x_dtype == bf else x
    ref = post.reshape(-1, n, 1) * fr.reshape(-1, 1, C) + torch.einsum(
        "tij,tic->tjc", comb.reshape(-1, n, n), xr.reshape(-1, n, C)
    )
    return tensors, ref.reshape(*lead, T, n * C)


def _run(device, tensors, variant):
    f, x, post, comb = tensors
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x.shape)), x.dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    bal = VARIANTS[variant]
    if bal == "op_weighted":
        saved = base_pd._work_assignment
        _drop_in = cand_pd.row_weighted_work_assignment(OP_WEIGHTED[variant])
        base_pd._work_assignment = lambda grid, total, *_: _drop_in(grid, total)  # graduated op takes a row_weight
        try:
            pd = base_pd.create_program_descriptor(f, x, post, comb, out, cfg)
        finally:
            base_pd._work_assignment = saved
    elif bal is None:
        pd = base_pd.create_program_descriptor(f, x, post, comb, out, cfg)
    else:
        pd = cand_pd.create_program_descriptor(f, x, post, comb, out, cfg, bal)
    return ttnn.generic_op([f, x, post, comb, out], pd)


def _check_ref(got, ref, x_dtype):
    tol = 1e-5 if x_dtype == f32 else 2e-2
    assert torch.allclose(got, ref, rtol=tol, atol=tol), f"max abs err {(got - ref).abs().max()}"


def _env_list(key, default):
    return [s for s in os.environ.get(key, ",".join(default)).split(",") if s]


# ---------------------------------------------------------------- correctness (bit-exact vs the op)
CORRECT_VARIANTS = _env_list("CB_VARIANTS", [v for v in VARIANTS if v != "orig"])
CORRECT_SHAPES = _env_list("CB_SHAPES", list(SHAPES))


@pytest.mark.parametrize("variant", CORRECT_VARIANTS)
@pytest.mark.parametrize("shape_id", CORRECT_SHAPES)
def test_correct(device, shape_id, variant):
    lead, T, C, n, x_dtype, f_dtype = SHAPES[shape_id]
    tensors, ref = _inputs(device, lead, T, C, n, x_dtype, f_dtype)
    base = ttnn.to_torch(_run(device, tensors, "orig")).float()
    _check_ref(base, ref, x_dtype)
    for rep in range(int(os.environ.get("CB_REPEAT", "2"))):  # repeated: the claim order differs run to run
        got = ttnn.to_torch(_run(device, tensors, variant)).float()
        assert torch.equal(got, base), f"{variant} rep {rep}: not bit-exact vs op, max diff {(got - base).abs().max()}"


# ---------------------------------------------------------------- perf (one op per test)
PERF_VARIANTS = _env_list("CB_VARIANTS", list(VARIANTS))
PERF_SHAPES = _env_list("CB_SHAPES", FOCUS)
PERF_REPS = [str(r) for r in range(int(os.environ.get("CB_REPS", "1")))]


@pytest.mark.parametrize("variant", PERF_VARIANTS)
@pytest.mark.parametrize("shape_id", PERF_SHAPES)
@pytest.mark.parametrize("rep", PERF_REPS)
def test_perf(device, rep, shape_id, variant):
    lead, T, C, n, x_dtype, f_dtype = SHAPES[shape_id]
    tensors, _ = _inputs(device, lead, T, C, n, x_dtype, f_dtype)
    session = os.environ.get("CB_SESSION", "default")
    with open(EXP_DIR / f"run_order_{session}.jsonl", "a") as fh:
        fh.write(json.dumps({"shape": shape_id, "variant": variant, "rep": rep}) + "\n")
    _run(device, tensors, variant)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
