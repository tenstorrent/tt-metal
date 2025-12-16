# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test suite for distributed layer norm and RMS norm using torch.allclose as the pass/fail metric.

This file contains test cases that call the reusable test functions from distributed_norm_test_utils.

HOW TO ADD NEW TESTS:
=====================
1. Import the run_distributed_norm_test function (already done below)
2. Create a new test function with pytest decorators
3. Call run_distributed_norm_test() with your desired parameters
4. Assert on the returned passes_allclose value

EXAMPLE - Adding a test for a specific configuration:
------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_my_custom_config(mesh_device):
    passes, max_abs_diff, max_rel_diff, mean_rel_diff = run_distributed_norm_test(
        mesh_device=mesh_device,
        batch_size=2,           # Your batch size
        seq_len=1024,           # Your sequence length
        hidden_dim=4096,        # Your hidden dimension
        eps=1e-6,               # Epsilon value
        norm_type="layer_norm", # "layer_norm" or "rms_norm"
        mean=0,                 # Input distribution mean
        var=1,                  # Input distribution variance
        outlier_pct=0,          # Percentage of outliers (0-1)
        outlier_var=0,          # Variance of outliers
        use_legacy=False,       # Use legacy reduction/rsqrt
        use_high_precision=True,# Use high precision compute
        verbose=False,          # Minimal output
    )
    assert passes, (
        f"TEST FAILED: Average relative difference {mean_rel_diff*100:.2f}% exceeds 5% threshold | "
        f"max_abs_diff={max_abs_diff:.6e} | "
        f"max_rel_diff={max_rel_diff:.6e}"
    )

See examples at the bottom of this file for more patterns!
"""

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.fused.distributed_norm_test_utils import run_distributed_norm_test


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [512, 1024, 2048])
@pytest.mark.parametrize("hidden_dim", [2048, 4096, 8192])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize(
    "mean, var, outlier_pct, outlier_var",
    [
        (0, 1, 0, 0),
        (0, 10, 0, 0),
        (-10, 10, 0, 0),
        (0, 1, 0.01, 10),
    ],
    ids=[
        "standard_normal",
        "high_variance",
        "shifted_mean",
        "with_outliers",
    ],
)
@pytest.mark.parametrize(
    "norm_type, use_legacy, use_welford",
    [
        ("layer_norm", False, False),  # LayerNorm with new reduction (non-Welford)
        ("layer_norm", False, True),  # LayerNorm with Welford
        ("rms_norm", False, False),  # RMSNorm with new reduction (non-Welford)
    ],
    ids=["layer_norm_new_reduction", "layer_norm_welford", "rms_norm_new_reduction"],
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
def test_distributed_norm_allclose(
    mesh_device,
    batch_size,
    seq_len,
    hidden_dim,
    eps,
    mean,
    var,
    outlier_pct,
    outlier_var,
    norm_type,
    use_legacy,
    use_welford,
):
    """
    Test distributed layer norm and RMS norm.

    Test passes if average relative difference is under 5%.

    Tests 3 configurations:
    1. LayerNorm with new reduction (non-Welford)
    2. LayerNorm with Welford algorithm
    3. RMSNorm with new reduction (non-Welford)
    """
    # Run the test using the utility function
    passes, max_abs_diff, max_rel_diff, mean_rel_diff = run_distributed_norm_test(
        mesh_device=mesh_device,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        eps=eps,
        norm_type=norm_type,
        mean=mean,
        var=var,
        outlier_pct=outlier_pct,
        outlier_var=outlier_var,
        use_legacy=use_legacy,
        use_high_precision=True,
        verbose=False,
        use_welford=use_welford,
    )

    # Assert - test passes only if average relative diff < 5%
    assert passes, (
        f"TEST FAILED: Average relative difference {mean_rel_diff*100:.2f}% exceeds 5% threshold | "
        f"max_abs_diff={max_abs_diff:.6e} | "
        f"max_rel_diff={max_rel_diff:.6e}"
    )


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("hidden_dim", [4096])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize(
    "norm_type, use_welford",
    [
        ("layer_norm", True),  # LayerNorm with Welford
        ("rms_norm", False),  # RMSNorm without Welford
    ],
    ids=[
        "layer_norm_welford",
        "rms_norm_no_welford"
    ],
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
def test_smoke(mesh_device, batch_size, seq_len, hidden_dim, eps, norm_type, use_welford):
    """
    Smoke test for distributed layer norm and RMS norm with standard parameters.
    Note: RMS norm only runs with use_welford=False.
    """
    # Run the test using the utility function
    passes, max_abs_diff, max_rel_diff, mean_rel_diff = run_distributed_norm_test(
        mesh_device=mesh_device,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        eps=eps,
        norm_type=norm_type,
        mean=0,
        var=1,
        outlier_pct=0,
        outlier_var=0,
        use_legacy=False,
        use_high_precision=True,
        verbose=False,
        use_welford=use_welford,
    )

    assert passes, (
        f"TEST FAILED: Average relative difference {mean_rel_diff*100:.2f}% exceeds 5% threshold | "
        f"max_abs_diff={max_abs_diff:.6e} | "
        f"max_rel_diff={max_rel_diff:.6e}"
    )


# ============================================================================
# Memory Layout Tests - Testing different layouts for gamma (weight) and beta (bias)
# ============================================================================


@pytest.mark.parametrize("norm_type", ["layer_norm"])
@pytest.mark.parametrize(
    "weight_layout, bias_layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    ],
    ids=[
        "row_major_both",
        "tile_both",
        "row_major_weight_tile_bias",
        "tile_weight_row_major_bias",
    ],
)
@pytest.mark.parametrize("use_welford", [True, False], ids=["welford", "no_welford"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
def test_distributed_layernorm_memory_layouts(mesh_device, norm_type, weight_layout, bias_layout, use_welford):
    """
    Test distributed layer norm with different memory layouts for weight and bias.
    Tests combinations of ROW_MAJOR and TILE layouts with both Welford and non-Welford algorithms.
    """
    passes, max_abs_diff, max_rel_diff, mean_rel_diff = run_distributed_norm_test(
        mesh_device=mesh_device,
        batch_size=1,
        seq_len=1024,
        hidden_dim=4096,
        eps=1e-6,
        norm_type=norm_type,
        mean=0,
        var=1,
        outlier_pct=0,
        outlier_var=0,
        use_legacy=False,
        use_high_precision=True,
        verbose=False,
        weight_layout=weight_layout,
        bias_layout=bias_layout,
        use_welford=use_welford,
    )

    assert passes, (
        f"TEST FAILED: Average relative difference {mean_rel_diff*100:.2f}% exceeds 5% threshold | "
        f"weight_layout={weight_layout} | bias_layout={bias_layout} | use_welford={use_welford} | "
        f"max_abs_diff={max_abs_diff:.6e} | "
        f"max_rel_diff={max_rel_diff:.6e}"
    )


@pytest.mark.parametrize("norm_type", ["rms_norm"])
@pytest.mark.parametrize(
    "weight_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
    ids=[
        "row_major_weight",
        "tile_weight",
    ],
)
@pytest.mark.parametrize(
    "use_welford",
    [False],
    ids=[
        "no_welford",
    ],
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
def test_distributed_rmsnorm_memory_layouts(mesh_device, norm_type, weight_layout, use_welford):
    """
    Test distributed RMS norm with different memory layouts for weight.
    RMS norm only has weight (no bias).
    Tests both Welford and non-Welford algorithms.
    """
    passes, max_abs_diff, max_rel_diff, mean_rel_diff = run_distributed_norm_test(
        mesh_device=mesh_device,
        batch_size=1,
        seq_len=1024,
        hidden_dim=4096,
        eps=1e-6,
        norm_type=norm_type,
        mean=0,
        var=1,
        outlier_pct=0,
        outlier_var=0,
        use_legacy=False,
        use_high_precision=True,
        verbose=False,
        weight_layout=weight_layout,
        use_welford=use_welford,
    )

    assert passes, (
        f"TEST FAILED: Average relative difference {mean_rel_diff*100:.2f}% exceeds 5% threshold | "
        f"weight_layout={weight_layout} | use_welford={use_welford} | "
        f"max_abs_diff={max_abs_diff:.6e} | "
        f"max_rel_diff={max_rel_diff:.6e}"
    )


# ============================================================================
# Example: How to easily add new custom tests
# ============================================================================
# Simply add new test functions with different parameters!
# The run_distributed_norm_test utility handles all the complexity.


@pytest.mark.parametrize(
    "norm_type",
    [
        "layer_norm",
    ],
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize("use_welford", [True, False], ids=["welford", "no_welford"])
def test_distributed_norm_large_batch(mesh_device, norm_type, use_welford):
    """
    Example: Test with larger batch size.
    This shows how easy it is to add a new test case.
    """
    passes, max_abs_diff, max_rel_diff, mean_rel_diff = run_distributed_norm_test(
        mesh_device=mesh_device,
        batch_size=4,  # Larger batch
        seq_len=2048,
        hidden_dim=4096,
        eps=1e-6,
        norm_type=norm_type,
        mean=0,
        var=1,
        outlier_pct=0,
        outlier_var=0,
        use_legacy=False,
        use_high_precision=True,
        verbose=False,
        use_welford=use_welford,
    )

    assert passes, (
        f"TEST FAILED: Average relative difference {mean_rel_diff*100:.2f}% exceeds 5% threshold | "
        f"max_abs_diff={max_abs_diff:.6e} | "
        f"max_rel_diff={max_rel_diff:.6e}"
    )


@pytest.mark.parametrize("hidden_dim", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
def test_distributed_layernorm_sweep_hidden_dim(mesh_device, hidden_dim):
    """
    Example: Sweep over different hidden dimensions for layer norm only.
    This shows how to test a specific configuration across multiple parameters.
    """
    passes, max_abs_diff, max_rel_diff, mean_rel_diff = run_distributed_norm_test(
        mesh_device=mesh_device,
        batch_size=1,
        seq_len=512,
        hidden_dim=hidden_dim,
        eps=1e-6,
        norm_type="layer_norm",
        mean=0,
        var=1,
        outlier_pct=0,
        outlier_var=0,
        use_legacy=False,
        use_high_precision=True,
        verbose=False,
    )

    assert passes, (
        f"TEST FAILED: Average relative difference {mean_rel_diff*100:.2f}% exceeds 5% threshold | "
        f"max_abs_diff={max_abs_diff:.6e} | "
        f"max_rel_diff={max_rel_diff:.6e}"
    )
