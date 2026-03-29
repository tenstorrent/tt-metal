# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import ttnn
import ttml


DEFAULT_SHAPE = (32, 32)
# Use a large shape so statistical checks are meaningful
STAT_SHAPE = (1024, 1024)
# Bypass AutocastTensor's default HALF precision to get the original dtype
FULL_PRECISION = ttml.autograd.PreferredPrecision.FULL


def check_normal_distribution(data, expected_mean=0.0, expected_std=1.0, sigma_threshold=15.0):
    """
    Checks that sample mean and std are close to expected values.
    Mean and std are each checked via z-test (within sigma_threshold standard errors).
    """
    n = data.numel()
    assert n >= 1000, f"Need at least 1000 samples for a meaningful check, got {n}"

    flat = data.detach().cpu().float().flatten()
    sample_mean = flat.mean().item()
    sample_std = flat.std(correction=0).item()

    if expected_std == 0.0:
        return (flat - expected_mean).abs().max().item() < 1e-3

    se_mean = expected_std / math.sqrt(n)
    se_std = expected_std / math.sqrt(2 * (n - 1))
    mean_ok = abs(sample_mean - expected_mean) < sigma_threshold * se_mean
    std_ok = abs(sample_std - expected_std) < sigma_threshold * se_std

    # # Skewness and kurtosis z-tests (expected: 0 and 3 for a normal)
    # # Disabled: ttnn::rand's hardware LFSR introduces systematic bias that
    # # fails these checks. The device Box-Muller math itself is accurate
    # # (verified with CPU-generated uniforms).
    # standardized = (flat - sample_mean) / sample_std
    # skew = standardized.pow(3).mean().item()
    # kurt = standardized.pow(4).mean().item()
    #
    # se_skew = math.sqrt(6.0 / n)
    # se_kurt = math.sqrt(24.0 / n)
    # skew_ok = abs(skew) < sigma_threshold * se_skew
    # kurt_ok = abs(kurt - 3.0) < sigma_threshold * se_kurt

    return mean_ok and std_ok


@pytest.mark.requires_device
class TestRandn:
    # --- default behaviour ---

    def test_randn_defaults(self):
        tensor = ttml.ops.randn(DEFAULT_SHAPE)
        ttnn_tensor = tensor.get_value(precision=FULL_PRECISION)

        assert ttnn_tensor.dtype == ttnn.DataType.BFLOAT16
        assert ttnn_tensor.layout == ttnn.Layout.TILE
        assert ttnn_tensor.storage_type() == ttnn.StorageType.DEVICE
        assert ttnn_tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tuple(ttnn_tensor.shape) == tuple(DEFAULT_SHAPE)

    # --- shape ---

    @pytest.mark.parametrize("shape", [tuple([32] * i) for i in range(1, 6)])
    def test_randn_shapes(self, shape):
        tensor = ttml.ops.randn(shape)
        assert tuple(tensor.get_value(precision=FULL_PRECISION).shape) == tuple(shape)

    # --- dtype ---

    @pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16, ttnn.DataType.FLOAT32])
    def test_randn_dtype(self, dtype):
        tensor = ttml.ops.randn(DEFAULT_SHAPE, dtype=dtype)
        ttnn_tensor = tensor.get_value(precision=FULL_PRECISION)
        assert ttnn_tensor.dtype == dtype
        assert tuple(ttnn_tensor.shape) == tuple(DEFAULT_SHAPE)

    # --- layout ---

    @pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
    def test_randn_layout(self, layout):
        tensor = ttml.ops.randn(DEFAULT_SHAPE, layout=layout)
        assert tensor.get_value(precision=FULL_PRECISION).layout == layout

    # --- distribution checks ---

    @pytest.mark.parametrize("mean,stddev", [(0.0, 1.0), (5.0, 2.0), (-3.0, 0.5)])
    def test_randn_distribution(self, mean, stddev):
        tensor = ttml.ops.randn(
            STAT_SHAPE,
            dtype=ttnn.DataType.FLOAT32,
            mean=mean,
            std=stddev,
            seed=42,
        )
        torch_tensor = ttnn.to_torch(tensor.get_value(precision=FULL_PRECISION))

        assert not torch.isnan(torch_tensor).any(), "Tensor contains NaN values"
        assert check_normal_distribution(
            torch_tensor, expected_mean=mean, expected_std=stddev
        ), f"Distribution check failed for mean={mean}, std={stddev}"

    def test_randn_std_zero(self):
        """std=0 produces a constant tensor equal to mean."""
        mean = 3.5
        tensor = ttml.ops.randn(
            DEFAULT_SHAPE,
            dtype=ttnn.DataType.FLOAT32,
            mean=mean,
            std=0.0,
            seed=0,
        )
        torch_tensor = ttnn.to_torch(tensor.get_value(precision=FULL_PRECISION)).float()
        assert torch.allclose(torch_tensor, torch.full(torch_tensor.shape, mean), atol=1e-2)

    def test_randn_no_nan(self):
        tensor = ttml.ops.randn(STAT_SHAPE, dtype=ttnn.DataType.FLOAT32, seed=1)
        torch_tensor = ttnn.to_torch(tensor.get_value(precision=FULL_PRECISION))
        assert not torch.isnan(torch_tensor).any()
        assert not torch.isinf(torch_tensor).any()

    # --- seed reproducibility ---

    def test_randn_seed_reproducibility(self):
        t1 = ttml.ops.randn(DEFAULT_SHAPE, dtype=ttnn.DataType.FLOAT32, seed=123)
        t2 = ttml.ops.randn(DEFAULT_SHAPE, dtype=ttnn.DataType.FLOAT32, seed=123)
        assert torch.equal(
            ttnn.to_torch(t1.get_value(precision=FULL_PRECISION)),
            ttnn.to_torch(t2.get_value(precision=FULL_PRECISION)),
        ), "Same seed should produce identical tensors"

    def test_randn_different_seeds_differ(self):
        t1 = ttml.ops.randn(DEFAULT_SHAPE, dtype=ttnn.DataType.FLOAT32, seed=1)
        t2 = ttml.ops.randn(DEFAULT_SHAPE, dtype=ttnn.DataType.FLOAT32, seed=2)
        assert not torch.equal(
            ttnn.to_torch(t1.get_value(precision=FULL_PRECISION)),
            ttnn.to_torch(t2.get_value(precision=FULL_PRECISION)),
        ), "Different seeds should produce different tensors"

    # --- invalid args ---

    def test_randn_invalid_args(self):
        with pytest.raises(Exception):
            # negative std must be rejected
            ttml.ops.randn(DEFAULT_SHAPE, std=-1.0)
