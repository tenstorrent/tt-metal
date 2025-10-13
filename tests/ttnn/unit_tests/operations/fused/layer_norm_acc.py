# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
import math

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import is_blackhole, comp_pcc

import torch
from torch.testing import assert_close

# Fallback defaults (same as PyTorch internal _DTYPE_PRECISIONS)
DEFAULT_TOLERANCES = {
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
}


def assert_close_robust(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    zero_thresh: float | None = None,
    zero_atol: float | None = None,
    symmetric_relative: bool = False,
    **kwargs,
):
    """
    Wrapper around torch.testing.assert_close with extra robustness.
    """
    dtype = actual.dtype
    finfo = torch.finfo(dtype) if dtype.is_floating_point else None

    # Cast actual to expected dtype
    actual = actual.to(expected.dtype)

    if actual.shape != expected.shape:
        raise ValueError(f"Shapes must match, got {actual.shape} vs {expected.shape}")

    # Use PyTorch's assert_close first (with user-supplied or default tolerances)
    try:
        assert_close(actual, expected, rtol=rtol, atol=atol, **kwargs)
        return
    except AssertionError:
        pass  # fall through to robust mode

    # Fill in defaults if not given
    if rtol is None or atol is None:
        rtol, atol = DEFAULT_TOLERANCES.get(actual.dtype, (1e-5, 1e-8))

    # Define near-zero thresholds
    if zero_thresh is None:
        scale = torch.median(expected.abs()).item() if expected.numel() > 0 else 1.0
        eps = finfo.eps if finfo else 1e-12
        zero_thresh = max(10.0 * eps * max(scale, 1.0), atol)
    if zero_atol is None:
        zero_atol = max(5 * atol, zero_thresh)

    diff = (actual - expected).abs()
    abs_expected = expected.abs()

    near_zero = abs_expected <= zero_thresh
    far_from_zero = ~near_zero
    ok = torch.ones_like(abs_expected, dtype=torch.bool)

    if near_zero.any():
        ok[near_zero] = diff[near_zero] <= zero_atol

    if far_from_zero.any():
        if symmetric_relative:
            scale = torch.maximum(actual.abs(), abs_expected)
        else:
            scale = abs_expected
        tol = atol + rtol * scale
        ok[far_from_zero] = diff[far_from_zero] <= tol[far_from_zero]

    if not bool(ok.all()):
        # Compute the relative differences for all elements
        rel_diff = torch.empty_like(diff)
        zero_mask = abs_expected == 0
        inf_val = torch.tensor(float("inf"), dtype=actual.dtype, device=actual.device)
        rel_diff[zero_mask] = torch.where(
            diff[zero_mask] > 0, inf_val, torch.tensor(0.0, dtype=actual.dtype, device=actual.device)
        )
        rel_diff[~zero_mask] = diff[~zero_mask] / abs_expected[~zero_mask]

        # Find the indices of failing elements
        failing_mask = ~ok
        failing_indices = failing_mask.nonzero(as_tuple=True)
        num_failing = failing_mask.sum().item()

        # Flatten the failing rel_diff and get their indices
        if num_failing > 0:
            flat_failing_rel_diff = rel_diff[failing_mask].flatten()
            flat_failing_indices = (
                torch.stack(
                    [
                        x[failing_mask].flatten()
                        for x in torch.meshgrid(*[torch.arange(s) for s in rel_diff.shape], indexing="ij")
                    ],
                    dim=1,
                )
                if rel_diff.ndim > 1
                else torch.nonzero(failing_mask, as_tuple=False)
            )
            num_to_show = min(20, num_failing)
            if flat_failing_rel_diff.numel() > num_to_show:
                topk_vals, topk_idx = torch.topk(flat_failing_rel_diff, num_to_show)
            else:
                topk_vals = flat_failing_rel_diff
                topk_idx = torch.arange(flat_failing_rel_diff.numel())
            details = []
            for n in range(len(topk_idx)):
                idx_flat = topk_idx[n].item()
                idx_tuple = tuple(flat_failing_indices[idx_flat].tolist())
                act = actual[idx_tuple].item()
                exp = expected[idx_tuple].item()
                absd = diff[idx_tuple].item()
                rel_d = rel_diff[idx_tuple].item()
                is_near_zero = abs(exp) <= zero_thresh
                this_atol = zero_atol if is_near_zero else atol
                this_rtol = 0.0 if is_near_zero else rtol
                details.append(
                    f"  {n+1:2d}: idx={idx_tuple}, actual={act}, expected={exp}, abs diff={absd}, "
                    f"atol={this_atol}, rtol={this_rtol}, rel diff={rel_d}, zero_thresh={zero_thresh}, "
                    f"zero_atol={zero_atol}, tol={tol[idx_tuple]}"
                )
            extra_msg = f"Top {len(details)} failing elements by relative difference:\n" + "\n".join(details)
        else:
            extra_msg = "No failing elements found."
        raise AssertionError(
            f"Robust check failed:\n"
            f"  max abs diff = {diff.max().item()}\n"
            f"  max rel diff = {rel_diff.max().item()}\n"
            f"  failing indices = {(~ok).nonzero(as_tuple=True)}\n"
            f"  {extra_msg}"
        )


def check_good_enough_outputs(
    ref64: torch.Tensor,
    torch_out: torch.Tensor,
    mine_out: torch.Tensor,
    *,
    margin: float = 1.10,
    tiny: float = 1e-30,
):
    """
    Compare 'mine_out' against 'torch_out' and 'ref64' to decide if it's 'good enough'.

    - ref64: float64 high-precision reference
    - torch_out: PyTorch output at the target dtype (cast to float32 or float64 for comparison)
    - mine_out: your implementation output at the same dtype (cast to float32 or float64)
    - margin: allowed ratio vs PyTorch’s error (default: 1.10 = 10% worse)
    - tiny: denominator floor to avoid divide-by-zero in relative error

    Returns:
        dict of error stats and "ok" flag.
    Raises:
        AssertionError if 'mine_out' is not good enough.
    """

    # 1. Relative error helper
    def rel_errors(y, ref):
        denom = torch.maximum(ref.abs(), torch.tensor(tiny, dtype=ref.dtype, device=ref.device))
        elem_rel = (y - ref).abs() / denom
        relL2 = torch.linalg.norm(y - ref) / torch.maximum(
            torch.linalg.norm(ref), torch.tensor(tiny, dtype=ref.dtype, device=ref.device)
        )
        return elem_rel, relL2.item()

    torch_rel, torch_relL2 = rel_errors(torch_out.double(), ref64)
    mine_rel, mine_relL2 = rel_errors(mine_out.double(), ref64)

    # 2. Percentile helper
    def P(t, q):
        return torch.quantile(t.flatten().float(), q).item()

    stats = {
        "torch_relL2": torch_relL2,
        "mine_relL2": mine_relL2,
        "torch_med": P(torch_rel, 0.5),
        "mine_med": P(mine_rel, 0.5),
        "torch_p95": P(torch_rel, 0.95),
        "mine_p95": P(mine_rel, 0.95),
        "torch_p99": P(torch_rel, 0.99),
        "mine_p99": P(mine_rel, 0.99),
    }

    # 3. Pass/fail checks (mine shouldn’t be worse than torch by more than margin)
    fails = []
    if not (mine_relL2 <= margin * (torch_relL2 + tiny)):
        fails.append(f"relL2 mine={mine_relL2:.3e} > {margin:.2f}× torch={torch_relL2:.3e}")
    if not (stats["mine_p95"] <= margin * (stats["torch_p95"] + tiny)):
        fails.append(f"p95 mine={stats['mine_p95']:.3e} > {margin:.2f}× torch={stats['torch_p95']:.3e}")
    if not (stats["mine_p99"] <= margin * (stats["torch_p99"] + tiny)):
        fails.append(f"p99 mine={stats['mine_p99']:.3e} > {margin:.2f}× torch={stats['torch_p99']:.3e}")

    stats["ok"] = len(fails) == 0
    stats["margin"] = margin

    if fails:
        msg = "\n".join(
            [
                "❌ Good-enough check failed:",
                *fails,
                "Stats: " + ", ".join(f"{k}={v:.3e}" for k, v in stats.items() if k != "ok"),
            ]
        )
        raise AssertionError(msg)

    return stats


def skip_welford_blackhole(use_welford):
    return pytest.mark.skipif(
        use_welford and is_blackhole(), reason="Welford's algorithm is not supported on Blackhole"
    )


def layer_norm(device, h, w, use_welford, dtype=torch.bfloat16, legacy_reduction=False):
    torch.manual_seed(0)

    # torch_input_bf16 = torch.randn((h, w), dtype=torch.bfloat16)*10 + 10
    # torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_input_tensor_64 = torch.randn((h, w), dtype=torch.float64) * 10 + 10
    print(torch_input_tensor_64)
    torch_output_64 = torch.nn.functional.layer_norm(torch_input_tensor_64, normalized_shape=[w])
    torch_input_tensor = torch_input_tensor_64.to(dtype)
    if dtype == torch.bfloat16:
        print("bf16")
        print(torch_input_tensor)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = ttnn.LayerNormDefaultProgramConfig(
        use_welford=use_welford, legacy_reduction=legacy_reduction, legacy_rsqrt=legacy_reduction
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    output_tensor = ttnn.layer_norm(
        input_tensor, program_config=program_config, compute_kernel_config=compute_kernel_config
    )
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # if dtype == torch.bfloat16:
    #     torch.testing.assert_close(torch_output_tensor.to(torch.float64), torch_output_64, rtol=1.6e-2, atol=1e-5)

    return torch_input_tensor, torch_input_tensor_64, torch_output_64, torch_output_tensor, output_tensor


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_layer_norm(device, h, w, use_welford):
    layer_norm(device, h, w, use_welford, False)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_layer_norm_with_weight_and_bias(device, h, w, use_welford):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, bias=bias, program_config=program_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_layer_norm_with_weight_bias_and_residual_input(device, h, w, use_welford):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_residual_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor + torch_residual_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_input_tensor,
        weight=weight,
        bias=bias,
        program_config=program_config,
    )
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9997)


@pytest.mark.parametrize("h", [2])
@pytest.mark.parametrize("w", [512])
def test_layer_norm_with_tile_layout(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_weight = torch.ones(w, dtype=torch.bfloat16)
    torch_bias = torch.zeros(w, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor,
        (w,),
        torch_weight,
        torch_bias,
    )

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    weight = ttnn.from_torch(torch_weight)
    weight = ttnn.to_layout(weight, ttnn.TILE_LAYOUT)
    weight = ttnn.to_device(weight, device)

    bias = ttnn.from_torch(torch_bias)
    bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
    bias = ttnn.to_device(bias, device)

    output_tensor = ttnn.layer_norm(
        input_tensor,
        weight=weight,
        bias=bias,
    )

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [1024, 2080])
@pytest.mark.parametrize("w", [3200, 4128])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_large_layer_norm(device, h, w, use_welford):
    if h == 2080:
        pytest.skip("Bug, see https://github.com/tenstorrent/tt-metal/issues/27126")

    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w))
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, program_config=program_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [4096])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_large_layer_norm_with_weight_and_bias(device, h, w, use_welford):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, bias=bias, program_config=program_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [4096])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_large_layer_norm_with_weight(device, h, w, use_welford):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], weight=torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, program_config=program_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [4096])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_large_layer_norm_with_bias(device, h, w, use_welford):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], bias=torch_bias)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, bias=bias, program_config=program_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize("h, w", [(2048, 2048)])
@pytest.mark.parametrize("legacy_reduction", [True, False])
@pytest.mark.parametrize("legacy_rsqrt", [True, False])
def test_large_layer_norm_with_legacy_reduction_and_rsqrt(device, h, w, legacy_reduction, legacy_rsqrt):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], bias=torch_bias)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(
        legacy_reduction=legacy_reduction, legacy_rsqrt=legacy_rsqrt, use_welford=False
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    output_tensor = ttnn.layer_norm(
        input_tensor, bias=bias, compute_kernel_config=compute_kernel_config, program_config=program_config
    )
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize("h", [32, 1024])
@pytest.mark.parametrize("w", [2880, 4096])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_large_layer_norm_with_weight_bias_and_residual_input(device, h, w, use_welford):
    if not use_welford:
        pytest.skip("Low PCC, see https://github.com/tenstorrent/tt-metal/issues/27291")

    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_residual_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor + torch_residual_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_input_tensor,
        weight=weight,
        bias=bias,
        program_config=program_config,
    )
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9997)


def bfloat16_ulp_spacing(x64: torch.Tensor) -> torch.Tensor:
    """
    Compute the size of 1 ULP in bfloat16 at each reference value x64.
    Returns a float64 tensor of the spacing (distance between adjacent bf16 values).
    """
    if not x64.dtype.is_floating_point:
        raise TypeError("Input must be a floating tensor")

    # view bits as int64 to extract exponent
    x = x64.abs()
    # Handle zeros, infs, NaNs cleanly
    ulp = torch.zeros_like(x)

    finite_mask = torch.isfinite(x) & (x != 0)
    if not finite_mask.any():
        return ulp

    xf = x[finite_mask]
    bits = xf.view(torch.int64)
    exp_bits = (bits >> 52) & 0x7FF  # 11-bit exponent field
    exp_unbiased = exp_bits - 1023  # float64 bias = 1023

    # Convert exponent to bf16 domain: same exponent range, just fewer mantissa bits
    # bf16 has 7 mantissa bits → spacing = 2^(exponent - 7)
    spacing = torch.ldexp(torch.ones_like(xf), exp_unbiased - 7)

    ulp[finite_mask] = spacing
    return ulp


def within_bf16_ulp_band(mine, ref64, n_ulp=1.0):
    spacing = bfloat16_ulp_spacing(ref64)
    lower = ref64 - n_ulp * spacing
    upper = ref64 + n_ulp * spacing
    return (mine >= lower) & (mine <= upper)


def layer_norm_sharded(device, use_welford, two_stage, tensor_type, large, dtype, legacy_reduction=False):
    torch.manual_seed(0)
    tile_height = 32
    tile_width = 32

    # Test parameters
    if two_stage:
        # Two-stage
        if large:
            tensor_height = 32 * 8
            tensor_width = 32 * 16
            block_wt = 2
        else:
            tensor_height = 32 * 4
            tensor_width = 32 * 8
            block_wt = 1
        shard_grid_rows = 2
        shard_grid_cols = 4
        block_ht = tensor_height // tile_height
        subblock_w = 1
        shard_height = tensor_height
        shard_width = tensor_width // (shard_grid_cols * shard_grid_rows)
        mem_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    else:
        # Block-sharded
        if large:
            tensor_height = 32 * 8
            tensor_width = 32 * 8
        else:
            tensor_height = 32 * 2
            tensor_width = 32 * 2
        shard_grid_rows = 2
        shard_grid_cols = 2
        block_wt = tensor_width // tile_width // shard_grid_cols
        block_ht = tensor_height // tile_height // shard_grid_rows
        subblock_w = 1
        shard_height = tensor_height // shard_grid_rows
        shard_width = tensor_width // shard_grid_cols
        mem_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    if tensor_type == "repeating":
        torch_input_tensor = torch.arange(tensor_width).repeat(tensor_height, 1).to(dtype)
    elif tensor_type == "ascending":
        torch_input_tensor = torch.arange(tensor_height * tensor_width).reshape(tensor_height, tensor_width).to(dtype)
    elif tensor_type == "random":
        torch_input_tensor = torch.rand((tensor_height, tensor_width), dtype=dtype)
    elif tensor_type == "random_normal":
        torch_input_tensor = torch.randn((tensor_height, tensor_width), dtype=dtype)
    elif tensor_type == "random_normal_skewed_mean":
        torch_input_tensor = torch.randn((tensor_height, tensor_width), dtype=dtype) + 10.0
    elif tensor_type == "random_normal_high_variance":
        torch_input_tensor = torch.randn((tensor_height, tensor_width), dtype=dtype) * 10.0
    elif tensor_type == "random_normal_very_high_variance":
        torch_input_tensor = torch.randn((tensor_height, tensor_width), dtype=dtype) * 100.0
    elif tensor_type == "random_normal_skewed_mean_high_variance":
        torch_input_tensor = torch.randn((tensor_height, tensor_width), dtype=dtype) * 10.0 + 10.0
    elif tensor_type == "random_normal_negative_skewed_mean_and_very_high_variance":
        torch_input_tensor = torch.randn((tensor_height, tensor_width), dtype=dtype) * 100.0 - 10.0
    elif tensor_type == "fa_rand":
        torch_input_tensor = fa_rand(tensor_height, tensor_width)

    torch_residual_input_tensor = torch.rand((tensor_height, tensor_width), dtype=dtype)
    torch_weight = torch.rand((tensor_width,), dtype=dtype)
    torch_bias = torch.rand((tensor_width,), dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor + torch_residual_input_tensor,
        normalized_shape=[tensor_width],
        weight=torch_weight,
        bias=torch_bias,
    )
    torch_output_64 = torch.nn.functional.layer_norm(
        torch_input_tensor.to(torch.float64), normalized_shape=[tensor_width]
    )

    # Create shard spec
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(shard_grid_cols - 1, shard_grid_rows - 1),
                )
            }
        ),
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Create memory config with sharding
    memory_config = ttnn.MemoryConfig(memory_layout=mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec)

    # Convert to TTNN tensor
    input_ttnn = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=memory_config,
    )
    residual_input_ttnn = ttnn.from_torch(
        torch_residual_input_tensor,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=memory_config,
    )
    weight_ttnn = ttnn.from_torch(
        torch_weight,
        layout=ttnn.Layout.TILE,
        device=device,
        # memory_config=memory_config,
    )
    bias_ttnn = ttnn.from_torch(
        torch_bias,
        layout=ttnn.Layout.TILE,
        device=device,
        # memory_config=memory_config,
    )

    # Create output memory config (same sharding as input)
    output_memory_config = ttnn.MemoryConfig(
        memory_layout=mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec
    )

    # Run layernorm
    output_ttnn = ttnn.layer_norm(
        input_ttnn,
        residual_input_tensor=residual_input_ttnn,
        weight=weight_ttnn,
        bias=bias_ttnn,
        memory_config=output_memory_config,
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            subblock_w=subblock_w,
            block_h=block_ht,
            block_w=block_wt,
            use_welford=use_welford,
            inplace=False,
            legacy_reduction=legacy_reduction,
            legacy_rsqrt=legacy_reduction,
        ),
    )
    output_ttnn = ttnn.to_layout(output_ttnn, ttnn.ROW_MAJOR_LAYOUT)
    output_ttnn = ttnn.from_device(output_ttnn)
    output_ttnn = ttnn.to_torch(output_ttnn)

    # INSERT_YOUR_CODE
    # Compute 3-ULP band in bfloat16 around torch_output_64 and check output_ttnn is within band
    import numpy as np

    def bfloat16_roundtrip(x64):
        """Quantize float64 array to bfloat16 precision and return float64."""
        x32 = np.atleast_1d(x64.astype(np.float32))
        x_bits = x32.view(np.uint32)
        # mask out lower 16 mantissa bits
        x_bf16_bits = x_bits & 0xFFFF0000
        x_bf16 = x_bf16_bits.view(np.float32)
        out = x_bf16.astype(np.float64)
        if np.ndim(x64) == 0:
            return out.item()
        return out

    def bfloat16_ulp_bits(x64):
        """Compute true bfloat16 ULP spacing by incrementing bf16 bits."""
        x32 = np.atleast_1d(x64.astype(np.float32))
        x_bits = x32.view(np.uint32)
        bf16_bits = x_bits & 0xFFFF0000

        # add 1 to the bfloat16 bit pattern (next representable bf16)
        next_bits = bf16_bits + 0x00010000
        next_bits = np.where(np.isfinite(x32), next_bits, bf16_bits)  # guard NaN/Inf

        x_bf16 = bf16_bits.view(np.float32).astype(np.float64)
        next_bf16 = next_bits.view(np.float32).astype(np.float64)
        ulp = np.abs(next_bf16 - x_bf16)
        if np.ndim(x64) == 0:
            return ulp.item()
        return ulp

    # Convert tensors to numpy for elementwise ops
    ref = torch_output_64.cpu().numpy()
    out = output_ttnn.to(torch.float64).cpu().numpy()

    ulp_band = 3
    ulp = bfloat16_ulp_bits(ref)
    band = ulp_band * ulp

    lower = ref - band
    upper = ref + band
    print(f"Number of values outside 3-ULP band: {np.sum(~((out >= lower) & (out <= upper)))}")
    ULP1 = 2**-7  # 1 ULP at 1.0 for bf16 = 0.0078125
    zero_limit = 3 * ULP1  # 3 ULPs at 1.0 ≈ 0.0234375
    diff = np.abs(out - ref)
    atol = 3 * ULP1
    mask = np.where(np.abs(ref) < atol, diff <= zero_limit, diff <= band)
    if not np.all(mask):
        failing = ~mask
        # Find the true worst violation: max (diff / ulp)
        ulp_diff = diff / ulp
        idx = np.unravel_index(np.argmax(ulp_diff), ulp_diff.shape)
        max_ulp_diff = ulp_diff[idx]
        # raise AssertionError(
        #     f"Absolute diff at max ULP diff: {diff[idx]} at {idx}, ref={ref[idx]}, out={out[idx]}, band={band[idx]}",
        #     f"Failing ref values: {ref[failing]} with corresponding out values: {out[failing]} and corresponding ULP diffs: {ulp_diff[failing]}"
        # )

    print(f"PCC: {comp_pcc(torch_output_tensor, output_ttnn)[1]}")
    return torch_input_tensor, torch_output_64, torch_output_tensor, output_ttnn


@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("two_stage", [True, False])
@pytest.mark.parametrize("tensor_type", ["random"])  # , "repeating", "random"])
@pytest.mark.parametrize("large", [True])  # , False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])  # , torch.float32])
def test_layer_norm_sharded(device, use_welford, two_stage, tensor_type, large, dtype):
    layer_norm_sharded(device, use_welford, two_stage, tensor_type, large, dtype)
