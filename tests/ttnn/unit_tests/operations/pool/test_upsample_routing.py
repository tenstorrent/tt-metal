# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

"""
Test routing logic for merged upsample operation.

Tests verify:
1. Integer scales route to optimized integer path
2. Float scales route to general float path
3. Fallback behavior when integer path doesn't support tensor config
4. Error handling for unsupported configurations
"""


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestUpsampleRouting:
    """Tests for upsample routing logic between integer and float paths."""

    def test_integer_scale_routes_to_integer_path(self, device):
        """Integer scale with ROW_MAJOR interleaved should use integer path."""
        input_shape = (1, 32, 32, 64)
        scale_factor = 2

        input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            input_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output = ttnn.upsample(input_tensor, scale_factor, mode="nearest")
        output_torch = ttnn.to_torch(output)

        # Verify output shape
        assert output_torch.shape == (1, 64, 64, 64)

        # Verify correctness with torch
        torch_upsample = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
        expected = torch_upsample(input_nhwc.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        assert_with_pcc(output_torch, expected, pcc=0.9999)

    def test_float_scale_routes_to_float_path(self, device):
        """Float scale should use float path."""
        input_shape = (1, 32, 32, 64)
        scale_factor = 2.5

        input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            input_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output = ttnn.upsample(input_tensor, scale_factor, mode="nearest")
        output_torch = ttnn.to_torch(output)

        # Verify output shape (floor of scale)
        expected_h = int(32 * 2.5)  # 80
        expected_w = int(32 * 2.5)  # 80
        assert output_torch.shape == (1, expected_h, expected_w, 64)

        # Verify correctness
        torch_upsample = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
        expected = torch_upsample(input_nhwc.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        assert_with_pcc(output_torch, expected, pcc=0.9999)

    def test_float_as_integer_routes_to_integer_path(self, device):
        """Float value 2.0 should route to integer path."""
        input_shape = (1, 32, 32, 64)
        scale_factor = 2.0  # Exact integer as float

        input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            input_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output = ttnn.upsample(input_tensor, scale_factor, mode="nearest")
        output_torch = ttnn.to_torch(output)

        assert output_torch.shape == (1, 64, 64, 64)

        torch_upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
        expected = torch_upsample(input_nhwc.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        assert_with_pcc(output_torch, expected, pcc=0.9999)

    def test_array_integer_scale(self, device):
        """Array of integers should route to integer path."""
        input_shape = (1, 32, 32, 64)
        scale_factor = [2, 3]

        input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            input_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output = ttnn.upsample(input_tensor, scale_factor, mode="nearest")
        output_torch = ttnn.to_torch(output)

        assert output_torch.shape == (1, 64, 96, 64)

        torch_upsample = torch.nn.Upsample(scale_factor=(2, 3), mode="nearest")
        expected = torch_upsample(input_nhwc.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        assert_with_pcc(output_torch, expected, pcc=0.9999)

    def test_array_float_scale(self, device):
        """Array of floats should route to float path."""
        input_shape = (1, 32, 32, 64)
        scale_factor = [2.5, 1.5]

        input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            input_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output = ttnn.upsample(input_tensor, scale_factor, mode="nearest")
        output_torch = ttnn.to_torch(output)

        expected_h = int(32 * 2.5)  # 80
        expected_w = int(32 * 1.5)  # 48
        assert output_torch.shape == (1, expected_h, expected_w, 64)

        torch_upsample = torch.nn.Upsample(scale_factor=(2.5, 1.5), mode="nearest")
        expected = torch_upsample(input_nhwc.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        assert_with_pcc(output_torch, expected, pcc=0.9999)

    def test_bilinear_integer_only(self, device):
        """Bilinear mode requires integer scales."""
        input_shape = (1, 32, 32, 64)

        input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            input_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Integer scale should work
        output = ttnn.upsample(input_tensor, 2, mode="bilinear")
        assert list(output.shape) == [1, 64, 64, 64]

        # Float scale should fail
        with pytest.raises(RuntimeError, match="bilinear mode requires integer scale factors"):
            ttnn.upsample(input_tensor, 2.5, mode="bilinear")

    def test_tile_layout_interleaved_uses_integer_path(self, device):
        """TILE layout with interleaved memory should use integer path."""
        # Use tile-aligned dimensions (multiples of 32)
        input_shape = (1, 32, 32, 64)  # NHWC format
        scale_factor = 2

        input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

        input_tensor = ttnn.from_torch(
            input_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Should work with integer path
        output = ttnn.upsample(input_tensor, scale_factor, mode="nearest")
        assert output.shape[1] == 64  # Output height doubled (32 * 2)
        assert output.shape[2] == 64  # Output width doubled (32 * 2)

    def test_width_sharded_fallback_to_float(self, device):
        """Width sharded with integer scale should fallback to float path (not supported by integer path)."""
        input_shape = (1, 32, 64, 128)
        scale_factor = 2  # Integer, but width sharded not supported by integer path

        input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)

        # Create width sharded input
        shard_width = input_shape[3]
        shard_spec = ttnn.create_sharded_memory_config(
            shape=(input_shape[0] * input_shape[1] * input_shape[2], shard_width),
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        input_tensor = ttnn.from_torch(
            input_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=shard_spec,
        )

        # Should fallback to float path silently (width sharded not supported by integer path)
        # Use explicit interleaved output since we're testing INPUT path selection
        output = ttnn.upsample(input_tensor, scale_factor, mode="nearest", memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output_torch = ttnn.to_torch(output)

        assert output_torch.shape == (1, 64, 128, 128)

    def test_invalid_mode_error(self, device):
        """Invalid mode should produce clear error."""
        input_shape = (1, 32, 32, 64)

        input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            input_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        with pytest.raises(RuntimeError, match="mode must be 'nearest' or 'bilinear'"):
            ttnn.upsample(input_tensor, 2, mode="invalid")

    def test_fractional_downscale(self, device):
        """Fractional scale < 1.0 should work with float path."""
        input_shape = (1, 64, 64, 64)
        scale_factor = 0.5

        input_nhwc = torch.randn(input_shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            input_nhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output = ttnn.upsample(input_tensor, scale_factor, mode="nearest")
        output_torch = ttnn.to_torch(output)

        # Output should be 32x32
        assert output_torch.shape == (1, 32, 32, 64)

        torch_upsample = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
        expected = torch_upsample(input_nhwc.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        assert_with_pcc(output_torch, expected, pcc=0.9999)
