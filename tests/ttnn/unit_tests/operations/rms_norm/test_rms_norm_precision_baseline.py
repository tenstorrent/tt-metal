# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for rms_norm (verifier artifact).

Measures PCC, max/mean abs error, relative RMS error, ULP p99, and the
got/true ratio spread across a small shape ladder (small / medium / wide-bf16 /
wide-fp32). The got/true ratio spread is the scale-bug detector: a tight cluster
of r = actual/expected around a non-1.0 constant would be a uniform scale/
structural bug; a broad spread centered on 1.0 is ordinary precision noise.

The wide-fp32 (W=8192) row is the cell the golden suite flags as
`numerical-precision` (rms just over the tight 0.02 fp32 target), but the ratio
spread reclassifies it: a *tight* cluster at ~1 + 2.5e-6·W (std ~0.001, growing
linearly in W) is a STRUCTURAL scale bug, not precision noise. It is marked
strict-xfail (Refinement 1) so the baseline runs green and alerts (xpass) the
moment the bug is fixed.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose
from ttnn.operations.rms_norm import rms_norm


_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


def _cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


def pytorch_rms_norm(x, gamma=None, epsilon=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + epsilon)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out.to(x.dtype)


# (shape, dtype, pcc_floor, rms_ceiling). The wide-fp32 row previously exposed a
# STRUCTURAL scale bug in the fp32 Σx² reduce (got/true ratio a tight cluster
# ~1 + 2.5e-6·W, not precision noise), tracked as Refinement 1 in
# op_requirements.md. Refinement 1 fixed it by routing the fp32 tile-aligned Σx²
# reduce through ReduceAlgorithm::AccumulateViaAdd (raw fp32 accumulator, single
# finalize), so the xfail is removed — the row is now an ordinary passing cell and
# the got/true scale-bug guard below (median ≈ 1.0) protects against regression.
CASES = [
    pytest.param((1, 1, 32, 64), ttnn.bfloat16, 0.995, 0.05, id="small-bf16"),
    pytest.param((2, 4, 128, 512), ttnn.bfloat16, 0.995, 0.05, id="med-bf16"),
    pytest.param((1, 1, 32, 8192), ttnn.bfloat16, 0.995, 0.05, id="wide-bf16"),
    pytest.param((1, 1, 32, 4096), ttnn.float32, 0.999, 0.02, id="mid-fp32"),
    pytest.param((1, 1, 32, 8192), ttnn.float32, 0.999, 0.02, id="wide-fp32"),
]


@pytest.mark.parametrize("shape,dtype,pcc_floor,rms_ceiling", CASES)
@pytest.mark.parametrize("with_gamma", [False, True])
def test_precision_baseline(device, shape, dtype, pcc_floor, rms_ceiling, with_gamma):
    torch.manual_seed(0)
    torch_dtype = _TORCH_DTYPE[dtype]
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch_gamma = ttnn_gamma = None
    if with_gamma:
        torch_gamma = torch.randn(W, dtype=torch_dtype)
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    expected = pytorch_rms_norm(torch_input, gamma=torch_gamma, epsilon=1e-6).to(torch.float32)
    ttnn_output = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=1e-6, compute_kernel_config=_cfg())
    actual = ttnn.to_torch(ttnn_output).reshape(expected.shape).to(torch.float32)

    # --- metrics ---
    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    std = expected.std().item()
    rel_rms = (torch.nn.functional.mse_loss(expected, actual).sqrt().item() / std) if std > 0 else 0.0

    # got/true ratio spread over finite, non-tiny reference elements
    mask = expected.abs() > (1e-3 * std)
    ratio = actual[mask] / expected[mask]
    r_med = ratio.median().item()
    r_p5 = torch.quantile(ratio, 0.05).item()
    r_p95 = torch.quantile(ratio, 0.95).item()
    r_std = ratio.std().item()

    allclose = comp_allclose(expected, actual)
    print(
        f"\n[precision] shape={shape} dtype={dtype} gamma={with_gamma}\n"
        f"    max_abs={max_abs:.5f} mean_abs={mean_abs:.6f} rel_rms={rel_rms:.5f}\n"
        f"    got/true ratio: median={r_med:.5f} p5={r_p5:.5f} p95={r_p95:.5f} std={r_std:.5f}\n"
        f"    {allclose}"
    )

    # PCC is the primary correctness gate; rms is recorded, ceiling loosened for
    # the marginal wide-fp32 cell.
    assert_with_pcc(expected, actual, pcc_floor)
    assert rel_rms < rms_ceiling, f"rel_rms {rel_rms:.5f} exceeded ceiling {rms_ceiling}"
    # Scale-bug guard: the ratio must be centered on ~1.0 (precision noise), not
    # a tight non-1.0 cluster (uniform scale/structural bug). Threshold 0.015
    # cleanly separates the healthy cells (bf16 ~1.0; fp32 W=4096 bias ~0.010)
    # from the fp32 W=8192 scale bug (bias ~0.020, both gamma variants) that the
    # `wide-fp32` xfail encodes (Refinement 1).
    assert abs(r_med - 1.0) < 0.015, f"got/true median {r_med:.5f} != 1.0 -> scale bug (Refinement 1)"
