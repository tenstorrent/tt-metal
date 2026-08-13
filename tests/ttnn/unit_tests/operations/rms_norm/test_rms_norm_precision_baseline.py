# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — Phase 0 precision baseline.

Measures, per (shape x dtype x layout x gamma) cell:

  * PCC                (assert_with_pcc, from tests.ttnn.utils_for_testing)
  * max / mean abs err (comp_allclose, from models.common.utility_functions)
  * relative RMS error  = ||got - true||_2 / ||true||_2
  * got/true RATIO spread — the scale-bug detector.  A tight cluster of
    r = got/true around a NON-1.0 constant is a uniform scale / structural bug
    (a masked-padding or scaler mistake), which PCC is largely blind to; a broad
    spread centred on 1.0 is ordinary rounding noise.  Printed always, because
    "high PCC + high relative RMS" is exactly the signature that must not be
    mistaken for a precision (fp32-intermediates) problem.

Run:
    scripts/run_safe_pytest.sh --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm

TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

# Same gates the golden suite uses (eval/golden_tests/rms_norm/helpers.py).
PCC_GATE = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}
RMS_GATE = {ttnn.float32: 0.02, ttnn.bfloat16: 0.04}

SHAPES = [
    pytest.param((32, 64), id="small_32x64"),
    pytest.param((2, 64, 256), id="medium_2x64x256"),
    pytest.param((1, 1, 512, 1024), id="large_512x1024"),
    pytest.param((1, 1, 32, 8192), id="wide_32x8192_crosscore"),
    pytest.param((47, 100), id="non_aligned_47x100"),
]


def _reference(x: torch.Tensor, gamma, epsilon: float) -> torch.Tensor:
    xf = x.to(torch.float32)
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def _ratio_spread(got: torch.Tensor, true: torch.Tensor):
    """median and p5/p95 of r = got/true over finite, non-tiny reference elems."""
    mask = torch.isfinite(true) & torch.isfinite(got) & (true.abs() > 1e-3 * true.abs().max())
    r = (got[mask] / true[mask]).to(torch.float32)
    if r.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95]))
    return q[1].item(), q[0].item(), q[2].item()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_rms_norm_precision_baseline(device, shape, dtype, layout, with_gamma):
    epsilon = 1e-6
    torch.manual_seed(42)
    torch_dtype = TORCH_DTYPE[dtype]

    torch_x = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch_gamma = None
    gamma = None
    if with_gamma:
        torch_gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch_dtype)
        gamma = ttnn.from_torch(
            torch_gamma,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    got = ttnn.to_torch(rms_norm(x, gamma=gamma, epsilon=epsilon)).to(torch.float32)
    true = _reference(torch_x, torch_gamma, epsilon)

    abs_err = (got - true).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rel_rms = (torch.linalg.vector_norm(got - true) / torch.linalg.vector_norm(true)).item()
    med, p5, p95 = _ratio_spread(got, true)

    _, allclose_msg = comp_allclose(true, got, rtol=1e-2, atol=1e-2)
    pcc = torch.corrcoef(torch.stack([got.flatten(), true.flatten()]))[0, 1].item()

    print(
        f"\n[precision] shape={tuple(shape)} dtype={dtype} layout={layout} "
        f"gamma={with_gamma}\n"
        f"            pcc={pcc:.7f} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_rms={rel_rms:.3e}\n"
        f"            ratio got/true: median={med:.6f} p5={p5:.6f} p95={p95:.6f}\n"
        f"            {allclose_msg}"
    )

    # Scale-bug detector: a tight cluster off 1.0 is structural, not rounding.
    assert abs(med - 1.0) < 0.02, f"got/true median {med} — uniform scale error, not precision noise"
    assert rel_rms < RMS_GATE[dtype], f"relative RMS {rel_rms} over gate {RMS_GATE[dtype]}"
    assert_with_pcc(true, got, PCC_GATE[dtype])
