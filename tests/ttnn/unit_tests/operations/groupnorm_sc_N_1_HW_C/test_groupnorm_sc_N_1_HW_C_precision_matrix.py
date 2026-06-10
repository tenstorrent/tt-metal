# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative precision characterization for groupnorm_sc_N_1_HW_C.

Full cross-product: 8 shapes x 3 dtypes x 4 fidelities x 2 fp32_acc x
2 distributions. All metrics are printed for every case (one MATRIX line per
case, parseable into precision_matrix_results.md); only PCC is asserted.

Shapes are tile-aligned with aligned group widths — non-aligned shapes are
Refinement 2/3 scope and not in SUPPORTED yet.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}

# Assert only on PCC (precision-matrix band: all fidelities + acc modes).
PCC_THRESHOLD = {
    ttnn.float32: 0.99,
    ttnn.bfloat16: 0.99,
    ttnn.bfloat8_b: 0.99,
}


def torch_groupnorm(x, num_groups, gamma, beta, eps=1e-5):
    N, _, HW, C = x.shape
    x_nchw = x.to(torch.float32).squeeze(1).permute(0, 2, 1)
    w = gamma.to(torch.float32).reshape(C)
    b = beta.to(torch.float32).reshape(C)
    y = torch.nn.functional.group_norm(x_nchw, num_groups, weight=w, bias=b, eps=eps)
    return y.permute(0, 2, 1).unsqueeze(1)


SHAPES = [
    pytest.param((1, 1, 32, 32), 1, id="32x32_g1_small"),
    pytest.param((1, 1, 32, 64), 2, id="32x64_g2"),
    pytest.param((1, 1, 64, 128), 4, id="64x128_g4"),
    pytest.param((1, 1, 128, 512), 8, id="128x512_g8"),
    pytest.param((1, 1, 256, 1024), 32, id="256x1024_g32_large"),
    pytest.param((1, 1, 64, 96), 3, id="64x96_g3_odd_groups"),
    pytest.param((1, 1, 512, 64), 2, id="512x64_g2_tall"),
    pytest.param((2, 1, 64, 128), 4, id="2x64x128_g4_batched"),
]


@pytest.mark.parametrize("distribution", ["rand", "randn"], ids=["uniform", "normal"])
@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32_acc", "bf16_acc"])
@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi],
    ids=["HiFi4", "HiFi3", "HiFi2", "LoFi"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bfp8"])
@pytest.mark.parametrize("shape, num_groups", SHAPES)
def test_groupnorm_sc_N_1_HW_C_precision_matrix(
    device, shape, num_groups, dtype, math_fidelity, fp32_acc, distribution
):
    torch.manual_seed(0)
    N, _, HW, C = shape
    torch_dtype = TORCH_DTYPE[dtype]
    if distribution == "rand":
        x = torch.rand(shape, dtype=torch_dtype)
    else:
        x = torch.randn(shape, dtype=torch_dtype)
    gamma = torch.randn(1, 1, 1, C, dtype=torch_dtype)
    beta = torch.randn(1, 1, 1, C, dtype=torch_dtype)

    expected = torch_groupnorm(x, num_groups, gamma, beta)

    affine_layout = ttnn.TILE_LAYOUT if dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_g = ttnn.from_torch(gamma, dtype=dtype, layout=affine_layout, device=device)
    tt_b = ttnn.from_torch(beta, dtype=dtype, layout=affine_layout, device=device)

    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=False,
        dst_full_sync_en=False,
    )
    result = ttnn.to_torch(
        groupnorm_sc_N_1_HW_C(tt_x, num_groups, gamma=tt_g, beta=tt_b, compute_kernel_config=cfg)
    ).to(torch.float32)

    # --- metrics (printed for every case, asserted only on PCC) ---
    pcc_pass, pcc_value = comp_pcc(expected, result, PCC_THRESHOLD[dtype])
    pcc = float(pcc_value)
    _, allclose_str = comp_allclose(expected, result, rtol=0.05, atol=0.06)
    abs_err = (result - expected).abs()
    max_abs = abs_err.max().item()
    median_abs = abs_err.median().item()
    p99_abs = torch.quantile(abs_err.flatten().float(), 0.99).item()
    rel_rms = (abs_err.pow(2).mean().sqrt() / expected.float().pow(2).mean().sqrt().clamp(min=1e-10)).item()

    print(
        f"\nMATRIX | {N}x{HW}x{C} G={num_groups} | {dtype} | {math_fidelity} | "
        f"fp32_acc={fp32_acc} | {distribution} | pcc={pcc:.6f} max_abs={max_abs:.5f} "
        f"median_abs={median_abs:.6f} p99_abs={p99_abs:.5f} rel_rms={rel_rms:.5f} | {allclose_str}"
    )
    assert pcc_pass, f"PCC below {PCC_THRESHOLD[dtype]}: {pcc:.6f}"
