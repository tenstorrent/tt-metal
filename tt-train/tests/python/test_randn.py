# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttml
import ttnn
import torch
import math


DEFAULT_SHAPE = (32, 32)
# Use a large shape so statistical checks are meaningful
STAT_SHAPE = (1024, 1024)


def check_normal_distribution(
    data, expected_mean=0.0, expected_std=1.0, sigma_threshold=5.0, tolerance_pct=5.0
):
    """
    Checks that sample mean and std are close to expected values.
    Mean is checked via z-test (within sigma_threshold standard errors).
    Std is checked via relative error (within tolerance_pct percent).
    """
    n = data.numel()
    assert n >= 1000, f"Need at least 1000 samples for a meaningful check, got {n}"

    flat = data.detach().cpu().float().flatten()
    sample_mean = flat.mean().item()
    sample_std = flat.std().item()

    if expected_std == 0.0:
        return (flat - expected_mean).abs().max().item() < 1e-3

    # z-test: sample mean should be within sigma_threshold standard errors
    se_mean = expected_std / math.sqrt(n)
    mean_ok = abs(sample_mean - expected_mean) < sigma_threshold * se_mean
    std_ok = abs(sample_std - expected_std) / expected_std * 100 < tolerance_pct

    return mean_ok and std_ok


@pytest.mark.requires_device
class TestRandn:
    @pytest.fixture(autouse=True)
    def setup_device(self):
        auto_ctx = ttml.autograd.AutoContext.get_instance()
        auto_ctx.open_device()
        yield
        auto_ctx.close_device()

    # --- default behaviour ---

    def test_randn_defaults(self):
        tensor = ttml.ops.randn(DEFAULT_SHAPE)
        ttnn_tensor = tensor.get_value()

        assert ttnn_tensor.dtype == ttnn.DataType.BFLOAT16
        assert ttnn_tensor.layout == ttnn.Layout.TILE
        assert ttnn_tensor.storage_type() == ttnn.StorageType.DEVICE
        assert ttnn_tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tuple(ttnn_tensor.shape) == tuple(DEFAULT_SHAPE)

    # --- shape ---

    @pytest.mark.parametrize("shape", [tuple([32] * i) for i in range(1, 6)])
    def test_randn_shapes(self, shape):
        tensor = ttml.ops.randn(shape)
        assert tuple(tensor.get_value().shape) == tuple(shape)

    # --- dtype ---

    @pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16, ttnn.DataType.FLOAT32])
    def test_randn_dtype(self, dtype):
        tensor = ttml.ops.randn(DEFAULT_SHAPE, dtype=dtype)
        ttnn_tensor = tensor.get_value()
        assert ttnn_tensor.dtype == dtype
        assert tuple(ttnn_tensor.shape) == tuple(DEFAULT_SHAPE)

    # --- layout ---

    @pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
    def test_randn_layout(self, layout):
        tensor = ttml.ops.randn(DEFAULT_SHAPE, layout=layout)
        assert tensor.get_value().layout == layout

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
        torch_tensor = ttnn.to_torch(tensor.get_value())

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
        torch_tensor = ttnn.to_torch(tensor.get_value()).float()
        assert torch.allclose(
            torch_tensor, torch.full(torch_tensor.shape, mean), atol=1e-2
        )

    def test_randn_no_nan(self):
        tensor = ttml.ops.randn(STAT_SHAPE, dtype=ttnn.DataType.FLOAT32, seed=1)
        torch_tensor = ttnn.to_torch(tensor.get_value())
        assert not torch.isnan(torch_tensor).any()
        assert not torch.isinf(torch_tensor).any()

    # --- seed reproducibility ---

    def test_randn_seed_reproducibility(self):
        t1 = ttml.ops.randn(DEFAULT_SHAPE, dtype=ttnn.DataType.FLOAT32, seed=123)
        t2 = ttml.ops.randn(DEFAULT_SHAPE, dtype=ttnn.DataType.FLOAT32, seed=123)
        assert torch.equal(
            ttnn.to_torch(t1.get_value()), ttnn.to_torch(t2.get_value())
        ), "Same seed should produce identical tensors"

    def test_randn_different_seeds_differ(self):
        t1 = ttml.ops.randn(DEFAULT_SHAPE, dtype=ttnn.DataType.FLOAT32, seed=1)
        t2 = ttml.ops.randn(DEFAULT_SHAPE, dtype=ttnn.DataType.FLOAT32, seed=2)
        assert not torch.equal(
            ttnn.to_torch(t1.get_value()), ttnn.to_torch(t2.get_value())
        ), "Different seeds should produce different tensors"

    # --- invalid args ---

    def test_randn_invalid_args(self):
        with pytest.raises(TypeError):
            # shape must be a list or tuple
            ttml.ops.randn(5)

        with pytest.raises(TypeError):
            # layout must be ttnn.Layout
            ttml.ops.randn(DEFAULT_SHAPE, layout="TILE")

        with pytest.raises(TypeError):
            # dtype must be ttnn.DataType
            ttml.ops.randn(DEFAULT_SHAPE, dtype="bfloat16")

        with pytest.raises(Exception):
            # negative std must be rejected
            ttml.ops.randn(DEFAULT_SHAPE, std=-1.0)
