# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision matrix for the 2D dual-multicast matmul (Refinement 1).

Characterizes numerical accuracy across the full numeric-configurability surface
added in Refinement 1:

  * dtype (activation) AND weight_dtype ∈ {float32, bfloat16, bfloat8_b}
  * fp32_dest_acc_en ∈ {True, False}   (16-bit DEST vs fp32 accumulator)
  * math_fidelity ∈ {HiFi4, HiFi2}
  * shapes small → deep-K (the K-accumulation stress axis)

Assert is on PCC only (per the numeric-formats precision-matrix contract); every
other metric (relative RMS, max/median abs error) is printed for observability.

Two corners are special:
  * {dtype=float32, fp32_dest_acc_en=False} is an op-side EXCLUSION (maxed input
    demands a maxed accumulator) -> skipped here, raises NotImplementedError.
  * bf16-OUTPUT + acc=False at very deep K (K>=8192) cannot meet the golden
    relative-RMS band (0.10) — the fundamental 16-bit-DEST accumulation floor
    (~O(sqrt(K))). PCC stays ~0.997, so the PCC assert here still passes; the RMS
    miss is tracked by the golden suite + changelog (Refinement 1b follow-up).

Uses comp_pcc / comp_allclose (models.common.utility_functions) — no hand-rolled
PCC. Device comes from the dir conftest's module-scoped fixture.
"""

import pytest
import torch
import ttnn

from ttnn.operations.matmul import matmul
from models.common.utility_functions import comp_allclose, comp_pcc


_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; reference in bf16
}

# Coarseness order: float32 (finest) < bfloat16 < bfloat8_b. The achievable PCC
# is bounded by the COARSER of activation/weight dtype (the effective dtype).
_COARSENESS = {ttnn.float32: 0, ttnn.bfloat16: 1, ttnn.bfloat8_b: 2}

# PCC floor per effective dtype (matrix context — looser than the golden RMS
# bands because we also sweep fidelity / the lossy 16-bit accumulator).
_PCC_FLOOR = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.99,
    ttnn.bfloat8_b: 0.98,
}


def _effective_dtype(dtype, weight_dtype):
    return dtype if _COARSENESS[dtype] >= _COARSENESS[weight_dtype] else weight_dtype


SHAPES = [
    pytest.param((32, 32), (32, 32), id="single_tile_32x32x32"),
    pytest.param((128, 256), (256, 512), id="multi_tile_128x256x512"),
    pytest.param((256, 512), (512, 1024), id="medium_256x512x1024"),
    pytest.param((256, 4096), (4096, 512), id="deepK_256x4096x512"),
]


@pytest.mark.parametrize("distribution", ["randn"])
@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
    ],
)
@pytest.mark.parametrize(
    "fp32_acc",
    [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")],
)
@pytest.mark.parametrize(
    "weight_dtype",
    [
        pytest.param(ttnn.float32, id="wt_fp32"),
        pytest.param(ttnn.bfloat16, id="wt_bf16"),
        pytest.param(ttnn.bfloat8_b, id="wt_bf8b"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.float32, id="act_fp32"),
        pytest.param(ttnn.bfloat16, id="act_bf16"),
        pytest.param(ttnn.bfloat8_b, id="act_bf8b"),
    ],
)
@pytest.mark.parametrize("a_shape, b_shape", SHAPES)
def test_matmul_precision_matrix(device, a_shape, b_shape, dtype, weight_dtype, fp32_acc, math_fidelity, distribution):
    # Op-side EXCLUSION: fp32 activation demands a fp32 accumulator.
    if dtype == ttnn.float32 and not fp32_acc:
        pytest.skip("EXCLUSIONS: {dtype=float32, fp32_dest_acc_en=False} (maxed input demands maxed accumulator)")

    torch.manual_seed(0)
    A = torch.randn(a_shape, dtype=_TORCH_DTYPE[dtype])
    B = torch.randn(b_shape, dtype=_TORCH_DTYPE[weight_dtype])
    expected = torch.matmul(A.to(torch.float32), B.to(torch.float32))

    ttnn_a = ttnn.from_torch(A, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(B, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=False,
    )
    out = ttnn.to_torch(matmul(ttnn_a, ttnn_b, compute_kernel_config=config)).to(torch.float32)

    # --- metrics (assert on PCC; print everything for observability) ---
    e = expected.flatten().to(torch.float64)
    a = out.flatten().to(torch.float64)
    abs_err = (e - a).abs()
    max_abs = float(abs_err.max())
    median_abs = float(abs_err.median())
    denom = float(e.pow(2).mean().sqrt()) or 1.0
    rel_rms = float(abs_err.pow(2).mean().sqrt()) / denom

    eff = _effective_dtype(dtype, weight_dtype)
    pcc_floor = _PCC_FLOOR[eff]
    _, pcc_val = comp_pcc(expected, out, pcc=pcc_floor)

    print(
        f"\n[precision-matrix] {a_shape}@{b_shape} "
        f"act={dtype.name} wt={weight_dtype.name} eff={eff.name} "
        f"fp32_acc={fp32_acc} fid={math_fidelity.name} "
        f"| PCC={pcc_val} relRMS={rel_rms:.5g} max_abs={max_abs:.5g} median_abs={median_abs:.5g} "
        f"| allclose={comp_allclose(expected, out)}"
    )

    assert pcc_val >= pcc_floor, f"PCC {pcc_val} < {pcc_floor} (eff dtype {eff.name})"
