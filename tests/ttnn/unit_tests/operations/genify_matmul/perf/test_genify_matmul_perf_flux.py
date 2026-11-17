# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 9: PERF FOCUS 4 (FLUX.2 prompt-stream FF1 + SwiGLU, 256 tokens) and its regime twins
(the epilogue-none twin, the M = 1 guidance / timestep linear_2), at FLUX's own config: bf16 x bf16,
HiFi2, fp32 DEST, DRAM interleaved.

Skipped unless GENIFY_MATMUL_PERF=1. GENIFY_MATMUL_KNOBS="NAME=value,..." overrides program-descriptor
knobs (python literals) for A/B measurements under `scripts/run_safe_pytest.sh --profile`;
GENIFY_MATMUL_ONLY="sub1,sub2" selects cases by id substring.
"""
import ast
import math
import os
import pytest
import torch
import ttnn
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(os.environ.get("GENIFY_MATMUL_PERF") != "1", reason="perf guard: set GENIFY_MATMUL_PERF=1"),
]

# (m, k, n, epilogue, A / out dtype)
FLUX_CASES = [
    (256, 6144, 18432, "swiglu", "bfloat16"),  # PERF FOCUS 4: double FF1, txt
    (256, 6144, 9216, "none", "bfloat16"),  # its epilogue-none twin (double QKV, txt)
    (1, 6144, 6144, "none", "bfloat16"),  # guidance MLP linear_2
    (1, 6144, 6144, "none", "float32"),  # timestep MLP linear_2, fp32 A / out
]


def _apply_knobs(monkeypatch, request=None):
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    # GENIFY_MATMUL_ONLY="sub1,sub2": run only the cases whose id contains one of the substrings (the
    # --profile wrapper passes a multi-word -k expression through a shell).
    only = [t for t in os.environ.get("GENIFY_MATMUL_ONLY", "").split(",") if t]
    if only and request is not None and not any(t in request.node.callspec.id for t in only):
        pytest.skip("not selected by GENIFY_MATMUL_ONLY")

    spec = os.environ.get("GENIFY_MATMUL_KNOBS", "")
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, value = item.split("=", 1)
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, ast.literal_eval(value))


@pytest.mark.parametrize("case", FLUX_CASES, ids=[f"{m}-{k}-{n}-{e}-{d}" for m, k, n, e, d in FLUX_CASES])
def test_perf_flux(case, device, monkeypatch, request):
    _apply_knobs(monkeypatch, request)
    m, k, n, epilogue, a_dt = case
    torch.manual_seed(0)
    A = torch.randn((1, k) if m == 1 else (1, m, k))
    B = torch.randn(k, n) / math.sqrt(k)
    dt = getattr(ttnn, a_dt)
    ta = create_ttnn_input_tensor(A, device, dtype=dt, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
    )
    kwargs = {"activation": "swiglu"} if epilogue == "swiglu" else {}
    out = genify_matmul(ta, tb, compute_kernel_config=cfg, **kwargs)
    ttnn.synchronize_device(device)


# Every interleaved shape whose plan R1b / the row-aware quantum changes (Refinement 9 plan diff), at its
# model's own formats: (m, k, n, A dtype, B dtype, out dtype, fidelity, fp32 DEST, epilogue, A / out in L1).
R1B_GUARD_CASES = [
    (256, 15360, 3072, "bfloat16", "bfloat16", "bfloat16", "HiFi2", True, "none", False),  # context_embedder
    (256, 6144, 3072, "bfloat16", "bfloat16", "bfloat16", "HiFi2", True, "addcmul", False),  # attn out, txt
    (128, 6144, 3072, "bfloat16", "bfloat16", "bfloat16", "HiFi2", True, "addcmul", False),
    (256, 9216, 6144, "bfloat16", "bfloat16", "bfloat16", "HiFi2", True, "none", False),  # FF2, txt
    (2048, 128, 3072, "bfloat16", "bfloat16", "bfloat16", "HiFi2", True, "none", False),  # x_embedder
    (512, 5120, 3072, "bfloat16", "bfloat8_b", "bfloat16", "HiFi2", True, "none", False),  # text enc qkv
    (512, 16384, 5120, "bfloat16", "bfloat8_b", "bfloat16", "HiFi2", True, "none", False),  # text enc down
    (1, 6144, 52224, "float32", "bfloat16", "bfloat16", "HiFi2", True, "bias", False),  # time_mod
    (1, 256, 6144, "float32", "bfloat16", "float32", "HiFi2", True, "none", False),  # timestep linear_1
    (128, 5120, 8704, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", True, "swiglu", True),  # Qwen prefill-128
    (32, 1024, 2048, "bfloat16", "bfloat16", "bfloat16", "HiFi4", True, "none", False),  # cartesian decode-32
    (1, 2048, 1536, "bfloat16", "bfloat16", "bfloat16", "HiFi4", True, "none", False),  # cartesian M = 1
]


@pytest.mark.parametrize(
    "case",
    R1B_GUARD_CASES,
    ids=[f"{m}-{k}-{n}-{a}-{w}-{f}-{e}{'-l1' if l1 else ''}" for m, k, n, a, w, _, f, _, e, l1 in R1B_GUARD_CASES],
)
def test_perf_r1b_guard(case, device, monkeypatch, request):
    _apply_knobs(monkeypatch, request)
    m, k, n, a_dt, w_dt, o_dt, fid, acc, epilogue, l1 = case
    torch.manual_seed(0)
    A = torch.randn((1, k) if m == 1 else (1, m, k))
    B = torch.randn(k, n) / math.sqrt(k)
    mc = ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG
    ta = create_ttnn_input_tensor(A, device, dtype=getattr(ttnn, a_dt), layout=ttnn.TILE_LAYOUT, memory_config=mc)
    tb = create_ttnn_input_tensor(B, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT)
    o_dtype = getattr(ttnn, o_dt)
    kwargs = {}
    n_out = n // 2 if epilogue == "swiglu" else n
    if epilogue == "swiglu":
        kwargs["activation"] = "swiglu"
    elif epilogue == "bias":
        kwargs["bias"] = create_ttnn_input_tensor(
            torch.randn(1, n), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
    elif epilogue == "addcmul":
        kwargs["residual"] = create_ttnn_input_tensor(
            torch.randn(*A.shape[:-1], n_out), device, dtype=o_dtype, layout=ttnn.TILE_LAYOUT
        )
        kwargs["gate"] = create_ttnn_input_tensor(torch.randn(1, n_out), device, dtype=o_dtype, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=acc, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, dtype=o_dtype, compute_kernel_config=cfg, memory_config=mc, **kwargs)
    ttnn.synchronize_device(device)
