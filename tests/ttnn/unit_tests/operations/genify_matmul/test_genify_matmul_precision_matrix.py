# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""test_genify_matmul_precision_matrix — the authoritative precision characterization (Refinement 1).

Cross-product: shapes x activation dtype (= output dtype) x weight dtype x fp32_dest_acc_en x math
fidelity x input distribution, interleaved DRAM, epilogue none. The reference multiplies the operands
as the device stores them (block-float quantized on host), so the error measured is fidelity,
accumulation width (fp32 vs 16-bit DEST / Float32 vs Float16_b partials) and output rounding.

Gate: at HiFi4, PCC and relative RMS together at helpers.TOLERANCES[(out dtype, fp32_dest_acc_en)].
Below HiFi4, PCC only at eval.metrics.LOWER_FIDELITY_PCC (relative RMS printed, not gated): truncated
multiplier operands give a fidelity-inherent error, e.g. fp32 out at LoFi measures rel-RMS ~0.025,
which is the hardware's LoFi floor, not an op defect (the golden fp32 model cases run HiFi2).
"rand" is zero-centred uniform: an all-positive input gives outputs whose mean dwarfs their spread,
so std-relative metrics then measure only output rounding at the mean's magnitude.
float32 x fp32_dest_acc_en=False is the op's permanent EXCLUSION: asserted to raise.

Shapes cover the R1 sub-regimes that change the numerical path: one resident K-block (no partials,
the packer writes the output straight from DEST), and multi-K-block walks (packer L1 accumulation
into cb_partials, Float32 or Float16_b per DEST width), plus M = 1 (m_non_aligned) and folded
leading dims.
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

from eval.golden_tests.genify_matmul.helpers import (
    create_ttnn_input_tensor,
    pytorch_genify_matmul,
    quantize,
    tolerance_for,
)
from eval.metrics import LOWER_FIDELITY_PCC
from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.genify_matmul import genify_matmul

SHAPES = [
    pytest.param((1, 1, 32, 32), (32, 32), id="single_tile"),
    pytest.param((1, 2048), (2048, 1536), id="M1_decode"),
    pytest.param((1, 384, 1536), (1536, 1024), id="prefill_one_kblock"),
    pytest.param((2, 64, 4096), (4096, 512), id="deepK_kwalk_folded_batch"),
    pytest.param((1, 64, 16384), (16384, 256), id="deepK16k_kwalk"),
]


@pytest.mark.parametrize("distribution", ["randn", "rand"])
@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
        pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
    ],
)
@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
@pytest.mark.parametrize(
    "weight_dtype",
    [
        pytest.param(ttnn.bfloat16, id="w_bf16"),
        pytest.param(ttnn.bfloat8_b, id="w_bfp8"),
        pytest.param(ttnn.bfloat4_b, id="w_bfp4"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="a_bf16"),
        pytest.param(ttnn.bfloat8_b, id="a_bfp8"),
        pytest.param(ttnn.float32, id="a_fp32"),
    ],
)
@pytest.mark.parametrize("a_shape, b_shape", SHAPES)
def test_genify_matmul_precision_matrix(
    a_shape, b_shape, dtype, weight_dtype, fp32_acc, math_fidelity, distribution, device, expect_error
):
    torch.manual_seed(4321)
    gen = torch.randn if distribution == "randn" else (lambda shape: torch.rand(shape) - 0.5)
    A = gen(a_shape)
    B = gen(b_shape) / math.sqrt(b_shape[-2])
    ta = create_ttnn_input_tensor(A, device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_acc, math_approx_mode=False)

    if dtype == ttnn.float32 and not fp32_acc:
        with expect_error(ExcludedCell, "unsupported combination"):
            genify_matmul(ta, tb, compute_kernel_config=cfg)
        return

    expected = pytorch_genify_matmul(quantize(A, dtype), quantize(B, weight_dtype))
    actual = ttnn.to_torch(genify_matmul(ta, tb, compute_kernel_config=cfg)).float()
    assert list(actual.shape) == list(expected.shape)

    _, pcc = comp_pcc(expected, actual)
    abs_err = (actual - expected).abs()
    max_abs = abs_err.max().item()
    median_abs = abs_err.median().item()
    p99_abs = torch.quantile(abs_err.flatten()[: 1 << 24], 0.99).item()
    ref_std = expected.std().item()
    abs_rms = abs_err.pow(2).mean().sqrt().item()
    rel_rms = abs_rms / ref_std if ref_std > 1e-12 else abs_rms
    mask = expected.abs() > 0.1 * ref_std
    ratio_med = (actual[mask] / expected[mask]).median().item() if mask.any() else float("nan")

    if math_fidelity == ttnn.MathFidelity.HiFi4:
        pcc_min, rms_max = tolerance_for(dtype, fp32_acc, {"math_fidelity": math_fidelity})
    else:
        pcc_min, rms_max = LOWER_FIDELITY_PCC, float("inf")
    print(
        f"\nPRECISION_MATRIX {a_shape}x{b_shape} a={dtype} w={weight_dtype} acc={fp32_acc} {math_fidelity} "
        f"{distribution}: pcc={pcc:.6f} rel_rms={rel_rms:.5f} max_abs={max_abs:.4f} median_abs={median_abs:.5f} "
        f"p99_abs={p99_abs:.4f} ratio_med={ratio_med:.4f} gate=({pcc_min}, {rms_max})"
    )
    assert pcc >= pcc_min, f"PCC {pcc} < {pcc_min}"
    assert rel_rms <= rms_max, f"relative RMS {rel_rms} > {rms_max}"
