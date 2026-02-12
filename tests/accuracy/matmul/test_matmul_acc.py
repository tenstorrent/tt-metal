# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import os
from collections import defaultdict

# Global results collector
_test_results = defaultdict(lambda: defaultdict(dict))


@pytest.mark.parametrize(
    "k_size",
    [
        32,  # Small K (1 tile)
        64,  # 2 tiles
        128,  # 4 tiles
        256,  # 8 tiles
        384,
        512,  # 16 tiles
        768,
        1024,  # 32 tiles
        1536,
        2048,  # 64 tiles
        2560,
        3072,
        4096,  # 128 tiles
        5120,
        6144,
        7168,
        8192,  # 256 tiles
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32])
@pytest.mark.parametrize(
    "math_fi,fp32_acc",
    [
        [ttnn.MathFidelity.LoFi, False],
        [ttnn.MathFidelity.HiFi4, False],
        [ttnn.MathFidelity.HiFi4, True],
    ],
)
@pytest.mark.parametrize("batch_size", [2])
def test_matmul_varying_k(device, k_size, dtype, batch_size, math_fi, fp32_acc):
    """Test matmul accuracy with varying dot product lengths (K dimension).

    Matrix dimensions: (batch_size, M, K) @ (batch_size, K, N) -> (batch_size, M, N)
    where M=32, N=32, and K varies.

    Args:
        device: Test device
        k_size: Dot product length (K dimension)
        dtype: Data type for computation
        batch_size: Batch size for the matmul
    """
    m_size = 32
    n_size = 32

    torch.manual_seed(0)

    # Create input tensors
    input_shape_a = (batch_size, m_size, k_size)
    input_shape_b = (batch_size, k_size, n_size)

    torch_dtype = torch.float64
    torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch_dtype)
    torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch_dtype)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    # Convert to ttnn tensors
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Perform matmul on device
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fi,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=fp32_acc,  # Enable packer L1 accumulation for Fp32 for perf
    )

    output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b, compute_kernel_config=compute_config)
    output_tensor = ttnn.to_torch(output_tensor)

    # Calculate metrics
    diff = torch.abs(torch_output_tensor - output_tensor)
    rel_diff = diff / (torch.abs(torch_output_tensor) + 1e-8)
    max_diff = torch.max(diff)
    max_diff_index = torch.argmax(diff)
    print(
        f"Max diff ref = {torch_output_tensor.view(-1)[max_diff_index].item()}, output = {output_tensor.view(-1)[max_diff_index].item()}"
    )
    max_rel_diff = torch.max(rel_diff).item()
    mean_diff = torch.mean(diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()

    # Store results for charting
    dtype_str = str(dtype).split(".")[-1]
    math_fi_str = str(math_fi).split(".")[-1]
    config_name = f"{dtype_str}_{math_fi_str}_fp32acc={fp32_acc}"

    _test_results[config_name][k_size] = {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
    }

    # Verify accuracy
    assert_with_pcc(torch_output_tensor, output_tensor, 0.95)
    logger.info(
        f"PASSED: M=32, N=32, K={k_size}, batch={batch_size}, dtype={dtype}, "
        f"max_diff={max_diff:.6e}, max_rel_diff={max_rel_diff:.6e}"
    )


def generate_charts(output_dir="./test_results"):
    """Generate charts from collected test results.

    Creates two charts:
    1. Max relative difference vs K for each config
    2. Max absolute difference vs K for each config

    Args:
        output_dir: Directory to save the charts
    """
    if not _test_results:
        logger.warning("No test results collected. Run tests first.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    configs = sorted(_test_results.keys())
    k_values = sorted(list(next(iter(_test_results.values())).keys()))

    # Chart 1: Max Relative Difference vs K
    plt.figure(figsize=(12, 7))
    for config in configs:
        k_vals = []
        max_rel_diffs = []
        for k in k_values:
            if k in _test_results[config]:
                k_vals.append(k)
                max_rel_diffs.append(_test_results[config][k]["max_rel_diff"])

        plt.plot(k_vals, max_rel_diffs, marker="o", label=config, linewidth=2)

    plt.xlabel("K (Dot Product Length)", fontsize=12)
    plt.ylabel("Max Relative Difference", fontsize=12)
    plt.title("MatMul Accuracy: Max Relative Difference vs K\n(M=32, N=32)", fontsize=14, fontweight="bold")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3, linestyle="--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    rel_diff_path = os.path.join(output_dir, "matmul_max_rel_diff_vs_k.png")
    plt.savefig(rel_diff_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved chart: {rel_diff_path}")
    plt.close()

    # Chart 2: Max Absolute Difference vs K
    plt.figure(figsize=(12, 7))
    for config in configs:
        k_vals = []
        max_diffs = []
        for k in k_values:
            if k in _test_results[config]:
                k_vals.append(k)
                max_diffs.append(_test_results[config][k]["max_diff"])

        plt.plot(k_vals, max_diffs, marker="o", label=config, linewidth=2)

    plt.xlabel("K (Dot Product Length)", fontsize=12)
    plt.ylabel("Max Absolute Difference", fontsize=12)
    plt.title("MatMul Accuracy: Max Absolute Difference vs K\n(M=32, N=32)", fontsize=14, fontweight="bold")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3, linestyle="--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    max_diff_path = os.path.join(output_dir, "matmul_max_abs_diff_vs_k.png")
    plt.savefig(max_diff_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved chart: {max_diff_path}")
    plt.close()

    # Also create a summary table
    summary_path = os.path.join(output_dir, "matmul_accuracy_summary.txt")
    with open(summary_path, "w") as f:
        f.write("MatMul Accuracy Test Results Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Matrix dimensions: M=32, N=32, K varies\n\n")

        for config in configs:
            f.write(f"\nConfiguration: {config}\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'K':>6} | {'Max Abs Diff':>14} | {'Mean Abs Diff':>14} | "
                f"{'Max Rel Diff':>14} | {'Mean Rel Diff':>14}\n"
            )
            f.write("-" * 80 + "\n")

            for k in k_values:
                if k in _test_results[config]:
                    results = _test_results[config][k]
                    f.write(
                        f"{k:>6} | {results['max_diff']:>14.6e} | {results['mean_diff']:>14.6e} | "
                        f"{results['max_rel_diff']:>14.6e} | {results['mean_rel_diff']:>14.6e}\n"
                    )

    logger.info(f"Saved summary: {summary_path}")


@pytest.fixture(scope="session", autouse=True)
def generate_charts_after_tests(request):
    """Pytest fixture to generate charts after all tests complete."""
    yield
    # This runs after all tests in the session
    generate_charts()
