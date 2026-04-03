# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for GitHub issue #32306: matmul with BLOCK_SHARDED output on 1D grid.

This test suite validates the fix for issue #32306, which caused matmul to hang
when BLOCK_SHARDED output memory config was requested with a 1D core grid
(single row or column of cores).

The fix:
1. For unbatched B tensor: Routes to 1D multicast program with proper per_core values
2. For batched B tensor: Throws a clear error message explaining the limitation

Test cases:
- test_matmul_block_sharded_1d_column_grid_batched_b: Error case (batched B on 1D column)
- test_matmul_block_sharded_1d_row_grid_batched_b: Error case (batched B on 1D row)
- test_matmul_block_sharded_1d_column_grid_unbatched_b: Success case (unbatched B on 1D column)
- test_matmul_block_sharded_1d_row_grid_unbatched_b: Success case (unbatched B on 1D row)
- test_matmul_block_sharded_2d_grid: Success case (2D grid)
- test_matmul_no_sharding_sanity: Sanity check
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


class TestMatmulBlockSharded1DGrid:
    """Tests for BLOCK_SHARDED output with 1D core grids (issue #32306)."""

    @pytest.mark.parametrize(
        "batch_a, batch_b, M, K, N, grid_start, grid_end",
        [
            # Original issue case: [5, 400, 32] @ [5, 32, 400] on 1x5 column grid
            (5, 5, 416, 32, 416, (0, 0), (0, 4)),
            # Smaller batched case on 1D column grid
            (2, 2, 64, 64, 64, (0, 0), (0, 1)),
        ],
    )
    def test_matmul_block_sharded_1d_column_grid_batched_b(
        self, device, batch_a, batch_b, M, K, N, grid_start, grid_end
    ):
        """
        Test that BLOCK_SHARDED on 1D column grid with batched B tensor raises an error.

        This configuration is unsupported because the 2D multicast program doesn't handle
        degenerate 1D grids correctly. The fix should raise TT_FATAL with a clear message.
        """
        torch.manual_seed(0)

        # Create batched input tensors
        torch_a = torch.randn((batch_a, M, K), dtype=torch.bfloat16)
        torch_b = torch.randn((batch_b, K, N), dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        # BLOCK_SHARDED memory config with 1D column grid (single column of cores)
        num_cores_y = grid_end[1] - grid_start[1] + 1
        shard_height = (batch_a * M + num_cores_y - 1) // num_cores_y
        # Round up to tile size
        shard_height = ((shard_height + 31) // 32) * 32
        shard_width = N

        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*grid_start), ttnn.CoreCoord(*grid_end))}),
                (shard_height, shard_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # Should raise RuntimeError with message about BLOCK_SHARDED on 1D grid
        with pytest.raises(RuntimeError) as excinfo:
            _ = ttnn.matmul(tt_a, tt_b, memory_config=memory_config)

        assert "BLOCK_SHARDED" in str(excinfo.value)
        assert "1D grid" in str(excinfo.value) or "single row or column" in str(excinfo.value)

    @pytest.mark.parametrize(
        "batch_a, batch_b, M, K, N, grid_start, grid_end",
        [
            # 1D row grid (5x1) with batched tensors
            (5, 5, 416, 32, 416, (0, 0), (4, 0)),
            # Smaller batched case on 1D row grid
            (2, 2, 64, 64, 64, (0, 0), (1, 0)),
        ],
    )
    def test_matmul_block_sharded_1d_row_grid_batched_b(self, device, batch_a, batch_b, M, K, N, grid_start, grid_end):
        """
        Test that BLOCK_SHARDED on 1D row grid with batched B tensor raises an error.

        Similar to column grid case, this configuration is unsupported with batched tensors.
        """
        torch.manual_seed(0)

        # Create batched input tensors
        torch_a = torch.randn((batch_a, M, K), dtype=torch.bfloat16)
        torch_b = torch.randn((batch_b, K, N), dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        # BLOCK_SHARDED memory config with 1D row grid (single row of cores)
        num_cores_x = grid_end[0] - grid_start[0] + 1
        shard_height = batch_a * M
        shard_width = (N + num_cores_x - 1) // num_cores_x
        # Round up to tile size
        shard_width = ((shard_width + 31) // 32) * 32

        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*grid_start), ttnn.CoreCoord(*grid_end))}),
                (shard_height, shard_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # Should raise RuntimeError with message about BLOCK_SHARDED on 1D grid
        with pytest.raises(RuntimeError) as excinfo:
            _ = ttnn.matmul(tt_a, tt_b, memory_config=memory_config)

        assert "BLOCK_SHARDED" in str(excinfo.value)
        assert "1D grid" in str(excinfo.value) or "single row or column" in str(excinfo.value)

    @pytest.mark.parametrize(
        "M, K, N, num_cores_y",
        [
            # Various sizes with unbatched B on 1D column grid
            # Use sizes that work well with the 1D mcast program constraints
            (128, 128, 128, 4),
            (256, 128, 128, 4),
        ],
    )
    def test_matmul_block_sharded_1d_column_grid_unbatched_b(self, device, M, K, N, num_cores_y):
        """
        Test that BLOCK_SHARDED on 1D column grid works correctly with unbatched B tensor.

        When B is unbatched (batch_size=1), the fix routes this to the 1D multicast program
        with HEIGHT_SHARDED-like behavior. This should complete without hanging.
        """
        torch.manual_seed(0)

        # Use 4D tensors: [1, 1, M, K] @ [1, 1, K, N] = [1, 1, M, N]
        torch_a = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
        torch_b = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
        torch_expected = torch.matmul(torch_a, torch_b)

        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        # BLOCK_SHARDED memory config with 1D column grid
        shard_height = M // num_cores_y
        shard_width = N

        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores_y - 1))}),
                (shard_height, shard_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # Should complete without hanging
        output = ttnn.matmul(tt_a, tt_b, memory_config=memory_config)
        ttnn.synchronize_device(device)

        output_torch = ttnn.to_torch(output)
        assert_with_pcc(torch_expected, output_torch, pcc=0.99)

    @pytest.mark.parametrize(
        "M, K, N, num_cores_x",
        [
            # Various sizes with unbatched B on 1D row grid
            # Use sizes that work well with the 1D mcast program constraints
            (128, 128, 128, 4),
            (128, 128, 256, 4),
        ],
    )
    def test_matmul_block_sharded_1d_row_grid_unbatched_b(self, device, M, K, N, num_cores_x):
        """
        Test that BLOCK_SHARDED on 1D row grid works correctly with unbatched B tensor.

        When B is unbatched (batch_size=1), the fix routes this to the 1D multicast program
        with WIDTH_SHARDED-like behavior. This should complete without hanging.
        """
        torch.manual_seed(0)

        # Use 4D tensors: [1, 1, M, K] @ [1, 1, K, N] = [1, 1, M, N]
        torch_a = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
        torch_b = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
        torch_expected = torch.matmul(torch_a, torch_b)

        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        # BLOCK_SHARDED memory config with 1D row grid
        shard_height = M
        shard_width = N // num_cores_x

        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, 0))}),
                (shard_height, shard_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # Should complete without hanging
        output = ttnn.matmul(tt_a, tt_b, memory_config=memory_config)
        ttnn.synchronize_device(device)

        output_torch = ttnn.to_torch(output)
        assert_with_pcc(torch_expected, output_torch, pcc=0.99)

    @pytest.mark.parametrize(
        "batch, M, K, N, grid_x, grid_y",
        [
            # 2D grid cases (should work normally)
            (1, 128, 64, 128, 2, 2),
            (1, 256, 128, 256, 4, 2),
        ],
    )
    def test_matmul_block_sharded_2d_grid(self, device, batch, M, K, N, grid_x, grid_y):
        """
        Test that BLOCK_SHARDED on 2D grid continues to work correctly.

        This is a regression test to ensure the fix doesn't break the normal 2D grid case.
        """
        torch.manual_seed(0)

        torch_a = torch.randn((batch, M, K), dtype=torch.bfloat16)
        torch_b = torch.randn((batch, K, N), dtype=torch.bfloat16)
        torch_expected = torch.bmm(torch_a, torch_b)

        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        # BLOCK_SHARDED memory config with 2D grid
        shard_height = (batch * M) // grid_y
        shard_width = N // grid_x
        # Round up to tile size
        shard_height = ((shard_height + 31) // 32) * 32
        shard_width = ((shard_width + 31) // 32) * 32

        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
                (shard_height, shard_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        output = ttnn.matmul(tt_a, tt_b, memory_config=memory_config)
        ttnn.synchronize_device(device)

        output_torch = ttnn.to_torch(output)
        assert_with_pcc(torch_expected, output_torch, pcc=0.99)

    def test_matmul_no_sharding_sanity(self, device):
        """
        Sanity test: basic matmul without sharding should work.

        This verifies the device and basic matmul functionality are working.
        """
        torch.manual_seed(0)

        # Use the exact shapes from the original issue
        torch_a = torch.randn((5, 400, 32), dtype=torch.bfloat16)
        torch_b = torch.randn((5, 32, 400), dtype=torch.bfloat16)
        torch_expected = torch.bmm(torch_a, torch_b)

        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        # matmul without memory_config should work
        output = ttnn.matmul(tt_a, tt_b)
        ttnn.synchronize_device(device)

        output_torch = ttnn.to_torch(output)
        assert_with_pcc(torch_expected, output_torch, pcc=0.99)


class TestMatmulBlockSharded1DGridOriginalIssue:
    """Tests that specifically reproduce the original issue #32306 scenario."""

    def test_original_issue_exact_shapes(self, device):
        """
        Reproduce the EXACT scenario from GitHub issue #32306.

        Original shapes: [5, 400, 32] @ [5, 32, 400] = [5, 400, 400]
        With BLOCK_SHARDED output on a 1x5 column grid, shard shape (416, 416).

        This should now raise a clear error instead of hanging.
        """
        # Create tensors with exact shapes from the issue
        v2 = ttnn.ones(
            shape=ttnn.Shape([5, 400, 32]),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=device,
        )
        v3 = ttnn.ones(
            shape=ttnn.Shape([5, 32, 400]),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=device,
        )

        # Exact memory config from the original issue
        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 4))}),
                (416, 416),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # With the fix, this should raise RuntimeError instead of hanging
        with pytest.raises(RuntimeError) as excinfo:
            _ = ttnn.matmul(v2, v3, memory_config=memory_config)

        error_msg = str(excinfo.value)
        # Verify the error message is helpful
        assert "BLOCK_SHARDED" in error_msg
        # Should suggest workarounds
        assert "HEIGHT_SHARDED" in error_msg or "WIDTH_SHARDED" in error_msg or "2D" in error_msg

    def test_original_issue_shapes_batched_error(self, device):
        """
        Reproduce the exact scenario from GitHub issue #32306 using torch tensors.

        Original shapes: [5, 400, 32] @ [5, 32, 400] = [5, 400, 400]
        With BLOCK_SHARDED output on a 1x5 column grid.

        This should now raise a clear error instead of hanging.
        """
        torch.manual_seed(0)

        # Exact shapes from the issue (padded to tile alignment)
        torch_a = torch.randn((5, 416, 32), dtype=torch.bfloat16)
        torch_b = torch.randn((5, 32, 416), dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        # Memory config from the original issue: 1x5 column grid
        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 4))}),
                (416, 416),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # With the fix, this should raise RuntimeError instead of hanging
        with pytest.raises(RuntimeError) as excinfo:
            _ = ttnn.matmul(tt_a, tt_b, memory_config=memory_config)

        error_msg = str(excinfo.value)
        # Verify the error message is helpful
        assert "BLOCK_SHARDED" in error_msg
        # Should suggest workarounds
        assert "HEIGHT_SHARDED" in error_msg or "WIDTH_SHARDED" in error_msg or "2D" in error_msg


class TestMatmulBlockSharded1DGridEdgeCases:
    """Edge cases and boundary conditions for the fix."""

    def test_single_core_grid(self, device):
        """
        Test BLOCK_SHARDED on a single core (1x1 grid).

        A single core is technically both a "1D column" and "1D row" grid.
        This should work when B is unbatched.
        """
        torch.manual_seed(0)

        M, K, N = 64, 64, 64

        # Use 4D tensors with batch=1
        torch_a = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
        torch_b = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
        torch_expected = torch.matmul(torch_a, torch_b)

        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        # Single core grid
        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                (M, N),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        output = ttnn.matmul(tt_a, tt_b, memory_config=memory_config)
        ttnn.synchronize_device(device)

        output_torch = ttnn.to_torch(output)
        assert_with_pcc(torch_expected, output_torch, pcc=0.99)

    def test_workaround_height_sharded_explicit(self, device):
        """
        Test the workaround: use HEIGHT_SHARDED explicitly instead of BLOCK_SHARDED on 1D column.

        This demonstrates the recommended workaround for users hitting issue #32306.
        """
        torch.manual_seed(0)

        # Use simpler shapes that divide evenly
        M, K, N = 256, 128, 128
        num_cores = 4

        torch_a = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
        torch_b = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
        torch_expected = torch.matmul(torch_a, torch_b)

        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        # Workaround: use HEIGHT_SHARDED explicitly on a 1D column grid
        shard_height = M // num_cores

        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))}),
                (shard_height, N),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        output = ttnn.matmul(tt_a, tt_b, memory_config=memory_config)
        ttnn.synchronize_device(device)

        output_torch = ttnn.to_torch(output)
        assert_with_pcc(torch_expected, output_torch, pcc=0.99)

    def test_workaround_width_sharded_explicit(self, device):
        """
        Test the workaround: use WIDTH_SHARDED explicitly instead of BLOCK_SHARDED on 1D row.

        This demonstrates another recommended workaround for users hitting issue #32306.
        """
        torch.manual_seed(0)

        # Use simpler shapes that divide evenly
        M, K, N = 128, 128, 256
        num_cores = 4

        torch_a = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
        torch_b = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
        torch_expected = torch.matmul(torch_a, torch_b)

        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        # Workaround: use WIDTH_SHARDED explicitly on a 1D row grid
        shard_width = N // num_cores

        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))}),
                (M, shard_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        output = ttnn.matmul(tt_a, tt_b, memory_config=memory_config)
        ttnn.synchronize_device(device)

        output_torch = ttnn.to_torch(output)
        assert_with_pcc(torch_expected, output_torch, pcc=0.99)
