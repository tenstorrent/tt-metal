# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for rms_norm_ttnn — the Phase-0 numbers the changelog records.

Measures, per (shape x dtype x fp32_dest_acc_en x operand set):

  * PCC                (tests.ttnn.utils_for_testing.assert_with_pcc)
  * max abs error      (models.common.utility_functions.comp_allclose)
  * mean abs error
  * relative RMS error  ||got - true||_2 / ||true||_2
  * the got/true RATIO SPREAD — median r, p5/p95 and std over the finite
    non-zero-reference elements.

The ratio spread is the SCALE-BUG DETECTOR the verifier protocol asks for. A
uniform multiplicative error (a CB race, a broadcast/reduce mistake, a wrong
scaler) keeps PCC very high while relative RMS is large, and lands above the
`severity=bug` PCC floor — it looks like a precision failure and is not one.
The two are told apart by `r = got / true`:

  * a TIGHT cluster of r around a NON-1.0 constant  -> scale / structural bug
    (fix the kernel; fp32 intermediates will not help)
  * a BROAD spread centred on 1.0                   -> ordinary rounding noise

so the printout carries both, and it is printed unconditionally rather than
only on failure — the baseline's job is to record the number, not to gate.

The thresholds below are the golden suite's per-dtype PCC bands, not tighter
ones: this file measures the shipped op, it does not re-specify it.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

EPSILON = 1e-5

#: The golden suite's bands (eval/golden_tests/rms_norm_ttnn/helpers.py:TOLERANCES).
#: 16-bit DEST accumulation is scored at what a 16-bit accumulator can deliver.
PCC_BAND = {
    (ttnn.float32, True): 0.999,
    (ttnn.float32, False): 0.995,
    (ttnn.bfloat16, True): 0.995,
    (ttnn.bfloat16, False): 0.995,
    (ttnn.bfloat8_b, True): 0.99,
    (ttnn.bfloat8_b, False): 0.99,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}

#: small / medium / larger / wide-reduce. The last one is the regime that makes
#: the reduction long enough for the accumulator width to be visible.
SHAPES = [
    (1, 1, 32, 64),
    (1, 1, 128, 512),
    (1, 2, 256, 1024),
    (1, 1, 32, 4096),
]


def _ratio_spread(got: torch.Tensor, true: torch.Tensor):
    """(median, p5, p95, std) of got/true over finite, non-negligible reference elements."""
    g = got.to(torch.float32).flatten()
    t = true.to(torch.float32).flatten()
    scale = t.abs().max().clamp_min(1e-30)
    keep = torch.isfinite(g) & torch.isfinite(t) & (t.abs() > 1e-3 * scale)
    if keep.sum() < 8:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = g[keep] / t[keep]
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95]))
    return q[1].item(), q[0].item(), q[2].item(), r.std().item()


def _metrics(got: torch.Tensor, true: torch.Tensor):
    g = got.to(torch.float32)
    t = true.to(torch.float32)
    err = (g - t).abs()
    rms = (err.pow(2).mean().sqrt() / t.pow(2).mean().sqrt().clamp_min(1e-30)).item()
    med, p5, p95, std = _ratio_spread(g, t)
    return {
        "max_abs": err.max().item(),
        "mean_abs": err.mean().item(),
        "rel_rms": rms,
        "r_median": med,
        "r_p5": p5,
        "r_p95": p95,
        "r_std": std,
    }


def _to_device(tensor, device, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(str(d) for d in s))
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b], ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["dest32", "dest16"])
def test_precision_baseline(device, shape, dtype, fp32_dest_acc_en):
    """The headline table: PCC / abs / RMS / ratio spread, no operands."""
    torch.manual_seed(0)
    tdt = TORCH_DTYPE[dtype]
    x = torch.randn(shape, dtype=tdt)

    expected = torch_rms_norm_ttnn(x, epsilon=EPSILON)

    got = ttnn.to_torch(
        rms_norm_ttnn(
            _to_device(x, device, dtype),
            epsilon=EPSILON,
            compute_kernel_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=fp32_dest_acc_en,
                math_approx_mode=True,
            ),
        )
    )

    m = _metrics(got, expected)
    print(
        f"\nBASELINE shape={tuple(shape)} dtype={dtype} fp32_dest_acc_en={fp32_dest_acc_en}\n"
        f"  max_abs={m['max_abs']:.3e}  mean_abs={m['mean_abs']:.3e}  rel_rms={m['rel_rms']:.3e}\n"
        f"  ratio r=got/true: median={m['r_median']:.6f}  p5={m['r_p5']:.6f}  "
        f"p95={m['r_p95']:.6f}  std={m['r_std']:.3e}\n"
        f"  {comp_allclose(expected.to(torch.float32), got.to(torch.float32), rtol=1e-2, atol=1e-2)[1]}"
    )

    # The scale-bug detector: a tight cluster of r away from 1.0 is structural,
    # not rounding.  Asserted, because a uniform 2x is a kernel bug that PCC
    # alone would wave through.
    if m["r_std"] == m["r_std"] and m["r_std"] < 5e-3:
        assert abs(m["r_median"] - 1.0) < 5e-3, (
            f"uniform scale error: got/true clusters at {m['r_median']:.6f} "
            f"(std {m['r_std']:.2e}) — a scale/structural bug, not precision"
        )

    assert_with_pcc(expected.to(torch.float32), got.to(torch.float32), PCC_BAND[(dtype, fp32_dest_acc_en)])


@pytest.mark.parametrize("shape", [(1, 1, 128, 512), (1, 1, 32, 4096)], ids=lambda s: "x".join(str(d) for d in s))
@pytest.mark.parametrize("mode", ["gamma", "gamma_bias", "residual", "gamma_bias_residual"])
def test_precision_baseline_operands(device, shape, mode):
    """The same metrics with each operand set present, at the op's DEFAULT config.

    bfloat16 / HiFi4 / 16-bit DEST — the cell a caller who omits
    `compute_kernel_config` gets, which is the one worth having a recorded
    baseline for.
    """
    torch.manual_seed(0)
    dtype = ttnn.bfloat16
    tdt = torch.bfloat16
    W = shape[-1]

    x = torch.randn(shape, dtype=tdt)
    weight = torch.randn((1, 1, 1, W), dtype=tdt) if "gamma" in mode else None
    bias = torch.randn((1, 1, 1, W), dtype=tdt) if "bias" in mode else None
    residual = torch.randn(shape, dtype=tdt) if "residual" in mode else None

    expected = torch_rms_norm_ttnn(x, epsilon=EPSILON, weight=weight, bias=bias, residual_input_tensor=residual)
    got = ttnn.to_torch(
        rms_norm_ttnn(
            _to_device(x, device, dtype),
            epsilon=EPSILON,
            weight=_to_device(weight, device, dtype) if weight is not None else None,
            bias=_to_device(bias, device, dtype) if bias is not None else None,
            residual_input_tensor=_to_device(residual, device, dtype) if residual is not None else None,
        )
    )

    m = _metrics(got, expected)
    print(
        f"\nBASELINE-OPERANDS shape={tuple(shape)} mode={mode} (bf16 / default config)\n"
        f"  max_abs={m['max_abs']:.3e}  mean_abs={m['mean_abs']:.3e}  rel_rms={m['rel_rms']:.3e}\n"
        f"  ratio r=got/true: median={m['r_median']:.6f}  p5={m['r_p5']:.6f}  "
        f"p95={m['r_p95']:.6f}  std={m['r_std']:.3e}"
    )

    if m["r_std"] == m["r_std"] and m["r_std"] < 5e-3:
        assert abs(m["r_median"] - 1.0) < 5e-3, f"uniform scale error: got/true clusters at {m['r_median']:.6f}"

    # A bias shifts the output off the normalized scale, so PCC against the
    # reference is measured on the same quantity the golden suite scores.
    assert_with_pcc(expected.to(torch.float32), got.to(torch.float32), 0.995)
