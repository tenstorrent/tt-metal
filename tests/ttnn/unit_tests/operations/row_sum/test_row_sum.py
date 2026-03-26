# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import math

from ttnn.operations.row_sum import row_sum


def compute_metrics(actual: torch.Tensor, expected: torch.Tensor):
    """Compute numerical accuracy metrics between two float tensors."""
    actual = actual.float().flatten()
    expected = expected.float().flatten()

    diff = (actual - expected).abs()

    # Mean / max absolute diff
    mean_abs_diff = diff.mean().item()
    max_abs_diff = diff.max().item()

    # Pearson correlation coefficient
    a_mean = actual.mean()
    e_mean = expected.mean()
    a_centered = actual - a_mean
    e_centered = expected - e_mean
    num = (a_centered * e_centered).sum()
    den = a_centered.norm() * e_centered.norm()
    pcc = (num / den).item() if den > 0 else 1.0

    # Relative RMS error: rms(diff) / rms(expected)
    rms_diff = diff.pow(2).mean().sqrt().item()
    rms_expected = expected.pow(2).mean().sqrt().item()
    rel_rms = rms_diff / rms_expected if rms_expected > 0 else float("inf")

    # ULP (units in last place) — approximate via nextafter
    ulp_diffs = []
    for a_val, e_val in zip(actual.tolist(), expected.tolist()):
        if e_val == 0.0:
            ulp = abs(a_val) / 1e-45 if a_val != 0 else 0.0
        else:
            eps_at = abs(math.nextafter(e_val, float("inf")) - e_val)
            ulp = abs(a_val - e_val) / eps_at
        ulp_diffs.append(ulp)
    ulp_tensor = torch.tensor(ulp_diffs)
    mean_ulp = ulp_tensor.mean().item()
    max_ulp = ulp_tensor.max().item()

    return {
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "pcc": pcc,
        "mean_ulp": mean_ulp,
        "max_ulp": max_ulp,
        "rel_rms_error": rel_rms,
    }


def format_layout(layout):
    return "TILE" if layout == ttnn.TILE_LAYOUT else "ROW_MAJOR"


def format_dtype(dtype):
    return "bfloat16" if dtype == ttnn.bfloat16 else "float32"


def make_input(distribution, shape, torch_dtype):
    torch.manual_seed(0)
    if distribution == "randn":
        return torch.randn(shape, dtype=torch_dtype)
    elif distribution == "rand":
        return torch.rand(shape, dtype=torch_dtype)
    elif distribution == "ones":
        return torch.ones(shape, dtype=torch_dtype)
    raise ValueError(f"Unknown distribution: {distribution}")


def print_report(shape, dtype, input_layout, output_layout, fp32_dest_acc_en, distribution, metrics):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  ROW SUM ACCURACY REPORT")
    print(f"{sep}")
    print(f"  Shape:          {shape[0]} x {shape[1]}")
    print(f"  Dtype:          {format_dtype(dtype)}")
    print(f"  Input layout:   {format_layout(input_layout)}")
    print(f"  Output layout:  {format_layout(output_layout)}")
    print(f"  FP32 dest acc:  {fp32_dest_acc_en}")
    print(f"  Distribution:   {distribution}")
    print(f"{'-' * 60}")
    print(f"  Mean abs diff:  {metrics['mean_abs_diff']:.6f}")
    print(f"  Max abs diff:   {metrics['max_abs_diff']:.6f}")
    print(f"  PCC:            {metrics['pcc']:.8f}")
    print(f"  Mean ULP:       {metrics['mean_ulp']:.1f}")
    print(f"  Max ULP:        {metrics['max_ulp']:.1f}")
    print(f"  Rel RMS error:  {metrics['rel_rms_error']:.6f}")
    print(f"{sep}\n")


@pytest.mark.parametrize(
    "shape",
    [
        # pytest.param((32, 32), id="single_tile"),
        # pytest.param((64, 128), id="2x4_tiles"),
        pytest.param((32, 64), id="1x2_tiles"),
        # pytest.param((128, 256), id="4x8_tiles"),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT], ids=["tile_in"])  # , ttnn.ROW_MAJOR_LAYOUT]  # , "rm_in"]
@pytest.mark.parametrize(
    "output_layout", [ttnn.TILE_LAYOUT], ids=["tile_out"]  # , ttnn.ROW_MAJOR_LAYOUT]  # , "rm_out"]
)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp16_acc", "fp32_acc"])
@pytest.mark.parametrize("distribution", ["rand"], ids=["rand"])  # , "randn", "ones"],  # , "randn", "ones"]
def test_row_sum(device, shape, dtype, input_layout, output_layout, fp32_dest_acc_en, distribution):
    """Test row_sum against PyTorch reference for all parameter combinations."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    torch_input = make_input(distribution, shape, torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=input_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = row_sum(ttnn_input, output_layout=output_layout, fp32_dest_acc_en=fp32_dest_acc_en)

    # Output shape is [H, 32]
    H = shape[0]
    assert list(result.shape) == [H, 32], f"Expected shape [{H}, 32], got {list(result.shape)}"

    torch_result = ttnn.to_torch(result)  # [H, 32]
    torch_expected = torch_input.float().sum(dim=-1, keepdim=True)  # [H, 1] in fp32

    actual = torch_result[:, 0:1].float()
    expected = torch_expected

    metrics = compute_metrics(actual, expected)
    print_report(shape, dtype, input_layout, output_layout, fp32_dest_acc_en, distribution, metrics)

    # Pass/fail based on PCC threshold
    # Known bug: tilize + bf16 + fp32_dest_acc_en corrupts data (DEST 32-bit mode
    # incompatible with bf16 tilize). Leave failing for reproducibility.
    if fp32_dest_acc_en:
        pcc_threshold = 0.9999
    elif dtype == ttnn.bfloat16:
        pcc_threshold = 0.999
    else:
        pcc_threshold = 0.9999
    assert metrics["pcc"] >= pcc_threshold, f"PCC {metrics['pcc']:.8f} below threshold {pcc_threshold}"


@pytest.mark.parametrize("shape", [pytest.param((32, 32), id="minimal")])
def test_row_sum_smoke(device, shape):
    """Smoke test: verify operation runs without error on simplest case."""
    torch_input = torch.ones(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = row_sum(ttnn_input)

    assert list(result.shape) == [32, 32]

    torch_result = ttnn.to_torch(result)
    expected_val = 32.0
    assert torch.allclose(
        torch_result[:, 0:1].float(),
        torch.full((32, 1), expected_val),
        rtol=0.02,
        atol=0.1,
    )
