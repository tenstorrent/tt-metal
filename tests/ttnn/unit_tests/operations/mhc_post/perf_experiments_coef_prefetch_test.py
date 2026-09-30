# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""coef_prefetch bake-off (ttnn/ttnn/operations/mhc_post/perf_experiments/coef_prefetch). Baseline = the op's
kernels + descriptor (copied verbatim); candidates = kernels/cand_dm.cpp with COEF_* defines. Run:

  scripts/run_safe_pytest.sh --run-all <this file> -k correct
  CP_PERF_SHAPES=a,b CP_PERF_VARIANTS=base,eager CP_REPEAT=3 scripts/run_safe_pytest.sh --profile --run-all <this file> -k perf

Every perf test launches exactly ONE device op; its label is appended to perf_experiments/coef_prefetch/run_order.jsonl
(report.py matches it against generated/dev<N>/profiler/.logs/cpp_device_perf_report.csv).
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import ttnn

REPO = Path(__file__).resolve().parents[5]
EXP_DIR = REPO / "ttnn/ttnn/operations/mhc_post/perf_experiments/coef_prefetch"
_spec = importlib.util.spec_from_file_location("cp_pd", EXP_DIR / "cp_descriptor.py")
cpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpd)

ORDER_LOG = EXP_DIR / "run_order.jsonl"

# name -> defines dict (None = the op's kernels, verbatim)
VARIANTS = {
    "base": None,
    "cand0": {},  # candidate kernel with every switch off (must equal the op)
    "eager": {"COEF_EAGER": 1},
    "eager_inc": {"COEF_EAGER": 1, "COEF_INC": 1},
    "fast": {"COEF_FAST": 1},
    "eager_fast": {"COEF_EAGER": 1, "COEF_FAST": 1},
    "share": {"COEF_SHARE": 1},
    "eager_share": {"COEF_EAGER": 1, "COEF_SHARE": 1},
    "eager_share_fast": {"COEF_EAGER": 1, "COEF_SHARE": 1, "COEF_FAST": 1},
    "eager_inc_share_fast": {"COEF_EAGER": 1, "COEF_INC": 1, "COEF_SHARE": 1, "COEF_FAST": 1},
    "share_fast": {"COEF_SHARE": 1, "COEF_FAST": 1},
    "nb": {"COEF_HELP_NB": 1},
    "eager_whole_fast": {"COEF_EAGER": 1, "COEF_WHOLE": 1, "COEF_FAST": 1},
    "eager_whole": {"COEF_EAGER": 1, "COEF_WHOLE": 1},
    "whole": {"COEF_WHOLE": 1},
    "eager_nb": {"COEF_EAGER": 1, "COEF_HELP_NB": 1},
    "eager_fast_nb": {"COEF_EAGER": 1, "COEF_FAST": 1, "COEF_HELP_NB": 1},
    "eager_share_fast_nb": {"COEF_EAGER": 1, "COEF_SHARE": 1, "COEF_FAST": 1, "COEF_HELP_NB": 1},
}

bf, f32 = ttnn.bfloat16, ttnn.float32
# id -> (lead, T, C, n, x_dtype, f_dtype)
SHAPES = {
    "T640_C7168": ((1, 1), 640, 7168, 4, bf, bf),
    "T640_C1792": ((1, 1), 640, 1792, 4, bf, bf),
    "T1280_C4096": ((1, 1), 1280, 4096, 4, bf, bf),
    # domain sweep (bf16)
    "T256_C1792": ((1, 1), 256, 1792, 4, bf, bf),
    "T512_C2560": ((1, 1), 512, 2560, 4, bf, bf),
    "T1024_C1792": ((1, 1), 1024, 1792, 4, bf, bf),
    "T2048_C1792": ((1, 1), 2048, 1792, 4, bf, bf),
    "T4096_C1792": ((1, 1), 4096, 1792, 4, bf, bf),
    "T640_C4096": ((1, 1), 640, 4096, 4, bf, bf),
    "T1024_C5120": ((1, 1), 1024, 5120, 4, bf, bf),
    "T2560_C6144": ((1, 1), 2560, 6144, 4, bf, bf),
    "T256_C7168": ((1, 1), 256, 7168, 4, bf, bf),
    "T1024_C7168": ((1, 1), 1024, 7168, 4, bf, bf),
    "T2048_C4096": ((1, 1), 2048, 4096, 4, bf, bf),
    "T4096_C2560": ((1, 1), 4096, 2560, 4, bf, bf),
    "T1280_C6144": ((1, 1), 1280, 6144, 4, bf, bf),
    # precision mix
    "T640_C1792_fp32": ((1, 1), 640, 1792, 4, f32, f32),
    "T640_C7168_fp32": ((1, 1), 640, 7168, 4, f32, f32),
    "T1000_C1792_xf32": ((1, 1), 1000, 1792, 4, f32, bf),
    "T1000_C7168_xf32": ((1, 1), 1000, 7168, 4, f32, bf),
    "T1000_C7168_ff32": ((1, 1), 1000, 7168, 4, bf, f32),
    "T1000_C1792": ((1, 1), 1000, 1792, 4, bf, bf),
    # correctness edges
    "T640_C1792_n1": ((1, 1), 640, 1792, 1, bf, bf),
    "T64_C224_n2": ((1, 1), 64, 224, 2, bf, bf),
    "T640_C1792_n3": ((1, 1), 640, 1792, 3, bf, bf),
    "T640_C1792_n5": ((1, 1), 640, 1792, 5, bf, bf),
    "T100_C224_fp32": ((1, 1), 100, 224, 4, f32, f32),
    "T17_C128_fp32": ((1, 1), 17, 128, 4, f32, f32),
    "T32_C32": ((1, 1), 32, 32, 4, bf, bf),
    "T96_C1184_straddle": ((1, 1), 96, 1184, 4, bf, bf),
    "B2_T160_C608_straddle": ((2, 1), 160, 608, 4, f32, bf),
    "B2x3_T64_C256": ((2, 3), 64, 256, 4, bf, bf),
}

FOCUS = ["T640_C7168", "T640_C1792", "T1280_C4096"]


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
    pd = cpd.create_program_descriptor(f, x, post, comb, out, cfg, VARIANTS[variant])
    return ttnn.generic_op([f, x, post, comb, out], pd)


def _check_ref(got, ref, x_dtype):
    tol = 1e-5 if x_dtype == f32 else 2e-2
    assert torch.allclose(got, ref, rtol=tol, atol=tol), f"max abs err {(got - ref).abs().max()}"


# ---------------------------------------------------------------- correctness (bit-exact vs the op's kernels)
CORRECT_SHAPES = os.environ.get("CP_CORRECT_SHAPES", ",".join(SHAPES)).split(",")
CORRECT_VARIANTS = os.environ.get("CP_CORRECT_VARIANTS", ",".join(v for v in VARIANTS if v != "base")).split(",")


@pytest.mark.parametrize("shape_id", CORRECT_SHAPES)
def test_correct(device, shape_id):
    lead, T, C, n, x_dtype, f_dtype = SHAPES[shape_id]
    tensors, ref = _inputs(device, lead, T, C, n, x_dtype, f_dtype)
    base = ttnn.to_torch(_run(device, tensors, "base")).float()
    _check_ref(base, ref, x_dtype)
    bad = []
    for variant in CORRECT_VARIANTS:
        for rep in range(int(os.environ.get("CP_CORRECT_REPEAT", "1"))):
            got = ttnn.to_torch(_run(device, tensors, variant)).float()
            if not torch.equal(got, base):
                bad.append(f"{variant}#{rep}: max diff {(got - base).abs().max()}")
    assert not bad, "; ".join(bad)


# ---------------------------------------------------------------- perf
PERF_SHAPES = os.environ.get("CP_PERF_SHAPES", ",".join(FOCUS)).split(",")
PERF_VARIANTS = os.environ.get("CP_PERF_VARIANTS", "base,eager").split(",")
REPEAT = int(os.environ.get("CP_REPEAT", "1"))


@pytest.mark.parametrize("rep", range(REPEAT))
@pytest.mark.parametrize("shape_id", PERF_SHAPES)
def test_perf(device, shape_id, rep):
    lead, T, C, n, x_dtype, f_dtype = SHAPES[shape_id]
    tensors, ref = _inputs(device, lead, T, C, n, x_dtype, f_dtype)
    for variant in PERF_VARIANTS:  # interleaved within a (shape, rep)
        with open(ORDER_LOG, "a") as fh:
            fh.write(json.dumps({"shape": shape_id, "variant": variant, "rep": rep, "pid": os.getpid()}) + "\n")
        out = _run(device, tensors, variant)
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)  # per op: the per-RISC marker buffer holds one op
        _check_ref(ttnn.to_torch(out).float(), ref, x_dtype)
