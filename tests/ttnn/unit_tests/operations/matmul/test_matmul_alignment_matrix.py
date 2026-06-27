# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Non-tile-aligned M / K / N matrix for the 2D dual-multicast matmul (Refinement 2).

matmul handles non-tile-aligned M, K, and N natively — NO host-side pad/slice and
NO in-kernel sub-tile masking. The descriptor counts tiles with ceil_div, so the
partial last tile along each of M/K/N is a real tile the kernels process in full;
ttnn's TILE_LAYOUT representation zero-fills that tile's out-of-logical-shape
padding at from_torch time (for fp32, bf16 AND bf8b — the host bf8b tilize zeros
the pad before the shared-face exponent is computed, so no block-format
corruption). Hence:
  * K-dot-product over the padded K-region = 0*0 = 0  (correct contraction),
  * M/N output padding (also 0) is sliced off by the output's logical shape.

This matrix asserts BOTH PCC and relative-RMS at the SAME per-(effective-dtype,
acc) bands the golden suite grades by (helpers.TOLERANCES) — so a regression in
the partial-tile path surfaces here, not only in the golden run. It isolates each
non-aligned dim (K, N, M), the multi-non-aligned precedence case, and a larger
non-aligned-K shape that exercises the multi-block-per-core + grid-overflow paths
on top of the partial tile.

Device comes from the dir conftest's module-scoped fixture.
"""

import pytest
import torch
import ttnn

from ttnn.operations.matmul import matmul
from models.common.utility_functions import comp_pcc


_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; reference in bf16
}

_COARSENESS = {ttnn.float32: 0, ttnn.bfloat16: 1, ttnn.bfloat8_b: 2}

# Mirror eval/golden_tests/matmul/helpers.py::TOLERANCES (PCC, relRMS), keyed on
# the COARSER of activation/weight dtype x fp32_dest_acc_en.
_TOLERANCES = {
    (ttnn.float32, True): (0.999, 0.02),
    (ttnn.bfloat16, True): (0.997, 0.04),
    (ttnn.bfloat16, False): (0.99, 0.10),
    (ttnn.bfloat8_b, True): (0.98, 0.12),
    (ttnn.bfloat8_b, False): (0.98, 0.15),
}


def _effective_dtype(dtype, weight_dtype):
    return dtype if _COARSENESS[dtype] >= _COARSENESS[weight_dtype] else weight_dtype


# (A_shape, B_shape) — each isolates a non-aligned dim (tagger precedence K>N>M).
SHAPES = [
    pytest.param((64, 50), (50, 128), id="k_nonaligned_K50"),
    pytest.param((128, 100), (100, 256), id="k_nonaligned_K100"),
    pytest.param((1, 128, 47), (47, 256), id="k_nonaligned_3d_K47"),
    pytest.param((64, 128), (128, 50), id="n_nonaligned_N50"),
    pytest.param((4, 128, 512), (512, 47), id="n_nonaligned_3d_N47"),
    pytest.param((50, 128), (128, 256), id="m_nonaligned_M50"),
    pytest.param((1, 47, 128), (128, 256), id="m_nonaligned_3d_M47"),
    pytest.param((50, 100), (100, 47), id="multi_KNM_precedence"),
    # larger non-aligned K: M/N span many tiles (multi-block per core) on top of
    # a partial last K-tile (K=272 -> Kt=9, last tile has 16 valid K-rows).
    pytest.param((544, 272), (272, 544), id="k_nonaligned_large_544x272"),
]


# (act_dtype, weight_dtype, fp32_acc) — fp32 only with acc=True (the {fp32,
# acc=False} op EXCLUSION); bf16/bf8b at both accumulators; one mixed path.
CONFIGS = [
    pytest.param(ttnn.float32, ttnn.float32, True, id="fp32_fp32_accT"),
    pytest.param(ttnn.bfloat16, ttnn.bfloat16, True, id="bf16_bf16_accT"),
    pytest.param(ttnn.bfloat16, ttnn.bfloat16, False, id="bf16_bf16_accF"),
    pytest.param(ttnn.bfloat8_b, ttnn.bfloat8_b, True, id="bf8b_bf8b_accT"),
    pytest.param(ttnn.bfloat8_b, ttnn.bfloat8_b, False, id="bf8b_bf8b_accF"),
    pytest.param(ttnn.bfloat16, ttnn.float32, True, id="bf16_fp32_mixed_accT"),
]


@pytest.mark.parametrize("a_shape, b_shape", SHAPES)
@pytest.mark.parametrize("dtype, weight_dtype, fp32_acc", CONFIGS)
def test_matmul_alignment_matrix(device, a_shape, b_shape, dtype, weight_dtype, fp32_acc):
    # Guard against accidental tile-aligned shapes (this test is about non-alignment).
    M, K, N = a_shape[-2], a_shape[-1], b_shape[-1]
    assert (M % 32) or (K % 32) or (N % 32), f"shape {a_shape}@{b_shape} is tile-aligned"

    torch.manual_seed(0)
    A = torch.randn(a_shape, dtype=_TORCH_DTYPE[dtype])
    B = torch.randn(b_shape, dtype=_TORCH_DTYPE[weight_dtype])
    expected = torch.matmul(A.to(torch.float32), B.to(torch.float32))

    ttnn_a = ttnn.from_torch(A, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(B, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=False,
    )
    out = ttnn.to_torch(matmul(ttnn_a, ttnn_b, compute_kernel_config=config)).to(torch.float32)

    # output shape MUST be the unpadded logical shape (proves M/N pad was sliced).
    expected_shape = list(a_shape[:-1]) + [N]
    assert list(out.shape) == expected_shape, f"shape {list(out.shape)} != {expected_shape}"

    eff = _effective_dtype(dtype, weight_dtype)
    pcc_band, rms_band = _TOLERANCES[(eff, fp32_acc)]

    e = expected.flatten().to(torch.float64)
    a = out.flatten().to(torch.float64)
    abs_err = (e - a).abs()
    denom = float(e.pow(2).mean().sqrt()) or 1.0
    rel_rms = float(abs_err.pow(2).mean().sqrt()) / denom
    _, pcc_val = comp_pcc(expected, out, pcc=pcc_band)

    print(
        f"\n[alignment-matrix] {a_shape}@{b_shape} act={dtype.name} wt={weight_dtype.name} "
        f"eff={eff.name} fp32_acc={fp32_acc} | PCC={pcc_val} relRMS={rel_rms:.5g} "
        f"| bands PCC>={pcc_band} RMS<={rms_band}"
    )

    assert pcc_val >= pcc_band, f"PCC {pcc_val} < {pcc_band} (eff {eff.name}, acc={fp32_acc})"
    assert rel_rms <= rms_band, f"relRMS {rel_rms} > {rms_band} (eff {eff.name}, acc={fp32_acc})"
