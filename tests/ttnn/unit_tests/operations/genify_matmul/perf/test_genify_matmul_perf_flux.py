# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 9: PERF FOCUS 4 (FLUX.2 prompt-stream FF1 + SwiGLU, 256 tokens) and its regime twins
(the epilogue-none twin, the M = 1 guidance / timestep linear_2), at FLUX's own config: bf16 x bf16,
HiFi2, fp32 DEST, DRAM interleaved.

Skipped unless GENIFY_MATMUL_PERF=1. GENIFY_MATMUL_KNOBS="NAME=value,..." overrides program-descriptor
knobs (python literals) for A/B measurements under `scripts/run_safe_pytest.sh --profile`.
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


def _apply_knobs(monkeypatch):
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    spec = os.environ.get("GENIFY_MATMUL_KNOBS", "")
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, value = item.split("=", 1)
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, ast.literal_eval(value))


@pytest.mark.parametrize("case", FLUX_CASES, ids=[f"{m}-{k}-{n}-{e}-{d}" for m, k, n, e, d in FLUX_CASES])
def test_perf_flux(case, device, monkeypatch):
    _apply_knobs(monkeypatch)
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
