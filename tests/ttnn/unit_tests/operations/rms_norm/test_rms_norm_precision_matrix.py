# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision matrix for ttnn.operations.rms_norm — the authoritative
characterization of the op's numerical surface (/numeric-formats-metal §10).

Refinement 1 opened that surface to the full cross-product

    dtype {bfloat16, float32, bfloat8_b}
      x  fp32_dest_acc_en {True, False}
      x  math_fidelity {HiFi4, HiFi3, HiFi2, LoFi}
      x  input distribution {uniform, normal}
      x  8 shapes (tile-aligned, W-non-aligned, H-non-aligned, both, wide)

so this file walks it and prints every metric from the skill's §11 for every
cell regardless of pass/fail. Only PCC is asserted; the rest is observability.

Two things this file exists to protect, both learned the hard way:

1. **PCC alone cannot see a dropped-element bug.** rms_norm's output is
   `x * rsqrt(mean(x^2) + eps)`, so under-counting the reduce only *rescales*
   each row by a near-constant — and PCC is scale-invariant. A bfloat8_b
   partial-W reduce that silently summed 32 of 49 elements scored PCC 0.9998
   and cleared the 0.99 golden gate (probes/probe_005.py). That is why
   `test_partial_w_reduce_counts_every_element` below asserts on a recovered
   *sum* from an all-ones input, and why it runs across the whole dtype x
   config surface rather than at one corner.

2. `default_compute_kernel_config()` stays `fp32_dest_acc_en=True`. This file
   drives the config explicitly on every case so a default change cannot
   quietly reinterpret the matrix.

Run with `-s` to see the metric table.
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import calculate_detailed_ulp_stats, comp_allclose, comp_pcc

from ttnn.operations.rms_norm import rms_norm, EXCLUSIONS, SUPPORTED


# bfloat8_b has no torch counterpart — build and compare in bf16, let
# ttnn.from_torch do the block-float quantization (same convention as
# eval/golden_tests/rms_norm/helpers.py::_TORCH_DTYPE).
TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}

# PCC gates per /numeric-formats-metal §11. Deliberately looser than the golden
# suite's: this matrix sweeps LoFi and bf16-DEST corners the golden default
# config never reaches, and the point is to *characterize* them, not to gate
# the op on its worst legal precision corner.
PCC_GATE = {
    ttnn.float32: 0.99,
    ttnn.bfloat16: 0.99,
    ttnn.bfloat8_b: 0.99,
}

SHAPES = [
    pytest.param((32, 32), id="32x32_small"),
    pytest.param((1, 1, 64, 128), id="64x128"),
    pytest.param((1, 1, 128, 512), id="128x512"),
    pytest.param((1, 1, 32, 4096), id="32x4096_wide"),
    pytest.param((32, 48), id="32x48_W_non_aligned"),
    pytest.param((48, 64), id="48x64_H_non_aligned"),
    pytest.param((1, 1, 17, 50), id="17x50_both_non_aligned"),
    pytest.param((2, 1, 100, 47), id="100x47_both_non_aligned_4d"),
]


def _torch_rms_norm(x, gamma=None, eps=1e-6):
    """Reference in fp32 (mirrors op_requirements.md's PyTorch reference)."""
    xf = x.float()
    out = xf / torch.sqrt((xf**2).mean(-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.float().reshape(-1)
    return out


def _config(math_fidelity, fp32_acc):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = math_fidelity
    cfg.fp32_dest_acc_en = fp32_acc
    cfg.math_approx_mode = False
    return cfg


def _skip_if_excluded(**axes):
    """Honor the op's own EXCLUSIONS — one source of truth, never a copy."""
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            pytest.skip(f"op EXCLUSIONS refuses this cell: {exc}")


def _report(label, got, expected):
    """Print every §11 metric; return PCC."""
    got_f, exp_f = got.float(), expected.float()
    abs_err = (got_f - exp_f).abs()
    _, pcc = comp_pcc(exp_f, got_f)
    pcc_val = float(str(pcc).split()[-1]) if not isinstance(pcc, float) else pcc
    rel_rms = (abs_err.pow(2).mean().sqrt() / exp_f.pow(2).mean().sqrt().clamp(min=1e-10)).item()
    ulp = calculate_detailed_ulp_stats(exp_f, got_f)
    print(
        f"\n[{label}]"
        f"\n  {comp_pcc(exp_f, got_f)[1]}"
        f"\n  {comp_allclose(exp_f, got_f)}"
        f"\n  max_abs={abs_err.max().item():.3e}  mean_abs={abs_err.mean().item():.3e}"
        f"  median_abs={abs_err.median().item():.3e}"
        f"  p99_abs={torch.quantile(abs_err.flatten().float(), 0.99).item():.3e}"
        f"\n  relative_rms={rel_rms:.3e}"
        f"\n  ulp max={ulp.get('max_ulp')} mean={ulp.get('mean_ulp')} median={ulp.get('median_ulp')}"
        f" std={ulp.get('std_ulp')} p95={ulp.get('p95_ulp')} p99={ulp.get('p99_ulp')}"
    )
    return pcc_val


@pytest.mark.parametrize("distribution", [pytest.param("rand", id="uniform"), pytest.param("randn", id="normal")])
@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi3, id="HiFi3"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
        pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bfp8"),
    ],
)
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_precision_matrix(device, shape, dtype, math_fidelity, fp32_acc, distribution):
    _skip_if_excluded(dtype=dtype, fp32_dest_acc_en=fp32_acc)
    assert dtype in SUPPORTED["dtype"] and fp32_acc in SUPPORTED["fp32_dest_acc_en"]

    torch_dtype = TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    gen = torch.rand if distribution == "rand" else torch.randn
    torch_input = gen(*shape, dtype=torch_dtype)
    torch_gamma = gen(shape[-1], dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, shape[-1]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )

    got = ttnn.to_torch(rms_norm(ttnn_input, gamma=ttnn_gamma, compute_kernel_config=_config(math_fidelity, fp32_acc)))
    expected = _torch_rms_norm(torch_input, torch_gamma)

    label = f"{tuple(shape)} {dtype} fid={math_fidelity} fp32_acc={fp32_acc} {distribution}"
    pcc_val = _report(label, got, expected)
    assert pcc_val >= PCC_GATE[dtype], f"PCC {pcc_val} < {PCC_GATE[dtype]} for {label}"


# --- the scale-invariance blind spot ---------------------------------------
#
# Regression pin for the Refinement 1 bug: a bfloat8_b reduce datapath made the
# partial-W 0/1 mask decode as all-zeros, so the final reduce-dim tile
# contributed nothing. PCC could not see it. An all-ones input can: the kernel's
# own output inverts to the exact element count it summed.

_W_VALID_LAST_SWEEP = [33, 49, 63, 100, 4097]  # valid_last = 1, 17, 31, 4, 1


@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bfp8"),
    ],
)
@pytest.mark.parametrize("W", _W_VALID_LAST_SWEEP)
def test_partial_w_reduce_counts_every_element(device, W, dtype, fp32_acc):
    """All-ones => out = 1/sqrt(S/W + eps); invert it to recover S, the number
    of elements the reduce actually summed. S must be W, not W rounded down to
    a tile boundary. This is the assertion PCC cannot make."""
    _skip_if_excluded(dtype=dtype, fp32_dest_acc_en=fp32_acc)

    eps = 1e-6
    torch_input = torch.ones((1, 1, 32, W), dtype=TORCH_DTYPE[dtype])
    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(
        rms_norm(ttnn_input, epsilon=eps, compute_kernel_config=_config(ttnn.MathFidelity.HiFi4, fp32_acc))
    ).float()

    recovered_sum = W * (1.0 / got[0, 0, 0, 0].item() ** 2 - eps)
    print(f"\n[W={W} {dtype} fp32_acc={fp32_acc}] recovered_sum={recovered_sum:.3f} (true {W})")

    # A whole dropped tile is >= 1 element and shows up as a >= 1.0 shortfall;
    # 1% of W is far tighter than that and still well clear of rounding.
    assert abs(recovered_sum - W) <= max(0.01 * W, 1.0), (
        f"reduce summed {recovered_sum:.3f} elements, expected {W} — "
        f"the partial-W mask is dropping the last reduce-dim tile"
    )
