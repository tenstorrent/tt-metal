# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in device-ns guard set for genify_matmul (bf16 x bf16, interleaved DRAM, epilogue none).

Skipped unless GENIFY_MATMUL_PERF=1. Run through the eval runner so device_kernel_ns is recorded:
    GENIFY_MATMUL_PERF=1 eval/eval_test_runner.sh tests/ttnn/unit_tests/operations/genify_matmul/perf <out_dir>
Shapes span the R1 work-split regimes: 2-D rectangle, 1-D N split (M <= 32), small-M / deep-K
DRAM-bound DiT shapes, and the tall-M shapes where the factorization trades W against re-reads.
"""
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

SHAPES = [
    (9472, 5120, 3456),
    (9472, 3456, 5120),
    (4096, 4096, 4096),
    (2048, 9216, 6144),
    (2048, 5120, 8704),
    (256, 6144, 9216),
    (256, 9216, 6144),
    (256, 15360, 3072),
    (1, 6144, 6144),
    (384, 1536, 1024),
    (2048, 128, 3072),
    (2048, 3072, 128),
    (32, 1024, 2048),
    (2048, 256, 512),
]


@pytest.mark.parametrize("fid", ["HiFi4", "HiFi2"])
@pytest.mark.parametrize("mkn", SHAPES, ids=[f"{m}-{k}-{n}" for m, k, n in SHAPES])
def test_perf(mkn, fid, device):
    m, k, n = mkn
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)


# Refinement 1: one representative per new numeric path (activation / weight dtype, DEST width),
# at the model's own fidelity. (m, k, n, A dtype, B dtype, out dtype, fidelity, fp32_dest_acc_en)
FORMAT_CASES = [
    (2048, 5120, 4160, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", True),  # Qwen prefill, bf4b weight
    (512, 5120, 3072, "bfloat16", "bfloat8_b", "bfloat16", "HiFi2", True),  # FLUX text encoder, bf8b weight
    (1, 6144, 6144, "float32", "bfloat16", "float32", "HiFi2", True),  # FLUX timestep linear_2, fp32 A
    (2048, 256, 512, "bfloat8_b", "bfloat8_b", "bfloat8_b", "LoFi", False),  # bf8b x bf8b, 16-bit DEST
    (4096, 256, 1024, "bfloat8_b", "bfloat8_b", "bfloat8_b", "LoFi", False),  # PDL upsample W pass
    (64, 16384, 256, "bfloat16", "bfloat16", "bfloat16", "HiFi4", False),  # 16-bit DEST, capped K-walk
]


@pytest.mark.parametrize(
    "case", FORMAT_CASES, ids=[f"{m}-{k}-{n}-{a}-{w}-{f}-acc{int(acc)}" for m, k, n, a, w, _, f, acc in FORMAT_CASES]
)
def test_perf_formats(case, device):
    m, k, n, a_dt, w_dt, o_dt, fid, acc = case
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = create_ttnn_input_tensor(A, device, dtype=getattr(ttnn, a_dt), layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=acc, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, dtype=getattr(ttnn, o_dt), compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
