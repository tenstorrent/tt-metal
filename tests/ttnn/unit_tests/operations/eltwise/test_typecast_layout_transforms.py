# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for typecast with layout and memory config transforms.

Validates that ttnn.typecast can transparently handle arbitrary combinations of:
  - Input/output layout (TILE <-> ROW_MAJOR)
  - Input/output memory config (interleaved <-> sharded, L1 <-> DRAM)
  - Input/output dtype (all supported typecast pairs)

The user passes input tensor and desired output configuration; typecast composes
the necessary internal transformations (to_layout, to_memory_config, prim::typecast).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

TILE_HEIGHT = 32
TILE_WIDTH = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_torch_input(shape, pt_dtype, low=0, high=100):
    """Create a torch tensor with the given shape and dtype."""
    if pt_dtype in (torch.int, torch.int32):
        return torch.randint(low, high, shape, dtype=pt_dtype)
    elif pt_dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=pt_dtype)
    else:
        return (torch.rand(shape) * (high - low) + low).to(pt_dtype)


def _make_sharded_mem_config(shard_layout, tensor_shape):
    """Build a simple L1 sharded memory config for a given tensor shape."""
    h, w = tensor_shape[-2], tensor_shape[-1]
    if shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        num_cores = max(1, h // TILE_HEIGHT)
        num_cores = min(num_cores, 8)
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
        shard_h = (h + num_cores - 1) // num_cores
        # Round up to tile height for TILE layout compatibility
        shard_h = ((shard_h + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
        shard_shape = [shard_h, w]
    elif shard_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        num_cores = max(1, w // TILE_WIDTH)
        num_cores = min(num_cores, 8)
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
        shard_w = (w + num_cores - 1) // num_cores
        shard_w = ((shard_w + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH
        shard_shape = [h, shard_w]
    else:  # BLOCK_SHARDED
        num_cores_h = max(1, min(h // TILE_HEIGHT, 2))
        num_cores_w = max(1, min(w // TILE_WIDTH, 2))
        core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1))}
        )
        shard_h = ((h + num_cores_h - 1) // num_cores_h + TILE_HEIGHT - 1) // TILE_HEIGHT * TILE_HEIGHT
        shard_w = ((w + num_cores_w - 1) // num_cores_w + TILE_WIDTH - 1) // TILE_WIDTH * TILE_WIDTH
        shard_shape = [shard_h, shard_w]

    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)


# ---------------------------------------------------------------------------
# Dtype pairs (representative subset covering float, int, mixed)
# Only dtypes that work in both TILE and ROW_MAJOR layouts.
# ---------------------------------------------------------------------------
DTYPE_PAIRS_CROSS_LAYOUT = [
    (torch.bfloat16, ttnn.bfloat16, ttnn.float32, 0.99),
    (torch.float32, ttnn.float32, ttnn.bfloat16, 0.99),
    (torch.float32, ttnn.float32, ttnn.int32, 0.99),
    (torch.int, ttnn.int32, ttnn.float32, 0.99),
    (torch.bfloat16, ttnn.bfloat16, ttnn.uint16, 0.99),
    (torch.int, ttnn.uint16, ttnn.bfloat16, 0.99),
]


# ---------------------------------------------------------------------------
# 1. Layout transforms: TILE <-> ROW_MAJOR (interleaved memory, dtype change)
# ---------------------------------------------------------------------------
class TestTypecastLayoutTransforms:
    """Typecast with layout change (TILE<->RM) on interleaved tensors."""

    @pytest.mark.parametrize(
        "pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc",
        DTYPE_PAIRS_CROSS_LAYOUT,
    )
    @pytest.mark.parametrize("input_shape", [[1, 1, 32, 32], [1, 1, 128, 128]])
    @pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
    def test_tile_to_row_major(
        self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, input_shape, mem_config
    ):
        """TILE input -> ROW_MAJOR output with dtype change."""
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_input_dtype)

        tt_input = ttnn.from_torch(
            torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
        )

        tt_output = ttnn.typecast(
            tt_input, dtype=tt_output_dtype, memory_config=mem_config, output_layout=ttnn.ROW_MAJOR_LAYOUT
        )

        assert tt_output.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tt_output.dtype == tt_output_dtype

        # Golden: typecast on host then compare
        cpu_input = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT)
        cpu_cast = ttnn.typecast(cpu_input, dtype=tt_output_dtype)
        torch_golden = ttnn.to_torch(cpu_cast)
        torch_output = ttnn.to_torch(tt_output)
        assert_with_pcc(torch_golden, torch_output, pcc=pcc)

    @pytest.mark.parametrize(
        "pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc",
        DTYPE_PAIRS_CROSS_LAYOUT,
    )
    @pytest.mark.parametrize("input_shape", [[1, 1, 32, 32], [1, 1, 128, 128]])
    @pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
    def test_row_major_to_tile(
        self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, input_shape, mem_config
    ):
        """ROW_MAJOR input -> TILE output with dtype change."""
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_input_dtype)

        tt_input = ttnn.from_torch(
            torch_input, dtype=tt_input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
        )

        tt_output = ttnn.typecast(
            tt_input, dtype=tt_output_dtype, memory_config=mem_config, output_layout=ttnn.TILE_LAYOUT
        )

        assert tt_output.layout == ttnn.TILE_LAYOUT
        assert tt_output.dtype == tt_output_dtype

        cpu_input = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT)
        cpu_cast = ttnn.typecast(cpu_input, dtype=tt_output_dtype)
        torch_golden = ttnn.to_torch(cpu_cast)
        torch_output = ttnn.to_torch(tt_output)
        assert_with_pcc(torch_golden, torch_output, pcc=pcc)

    @pytest.mark.parametrize("input_shape", [[1, 1, 32, 32], [1, 1, 128, 128]])
    def test_layout_change_same_dtype(self, device, input_shape):
        """Layout change without dtype change - pure layout transform via typecast."""
        torch.manual_seed(0)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        tt_output = ttnn.typecast(tt_input, dtype=ttnn.bfloat16, output_layout=ttnn.ROW_MAJOR_LAYOUT)

        assert tt_output.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tt_output.dtype == ttnn.bfloat16

        torch_output = ttnn.to_torch(tt_output)
        assert_with_pcc(torch_input, torch_output, pcc=0.9999)


# ---------------------------------------------------------------------------
# 2. Memory config transforms: interleaved <-> sharded (same layout, dtype change)
# ---------------------------------------------------------------------------
class TestTypecastMemoryConfigTransforms:
    """Typecast with memory config change (interleaved<->sharded)."""

    @pytest.mark.parametrize(
        "pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc",
        [
            (torch.bfloat16, ttnn.bfloat16, ttnn.float32, 0.99),
            (torch.float32, ttnn.float32, ttnn.int32, 0.99),
        ],
    )
    @pytest.mark.parametrize(
        "shard_layout",
        [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ],
    )
    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_interleaved_to_sharded(
        self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, shard_layout, input_shape
    ):
        """Interleaved input -> sharded output with dtype change."""
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_input_dtype)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=tt_input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output_mem_config = _make_sharded_mem_config(shard_layout, input_shape)

        tt_output = ttnn.typecast(tt_input, dtype=tt_output_dtype, memory_config=output_mem_config)

        assert tt_output.dtype == tt_output_dtype
        assert tt_output.is_sharded()

        cpu_input = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT)
        cpu_cast = ttnn.typecast(cpu_input, dtype=tt_output_dtype)
        torch_golden = ttnn.to_torch(cpu_cast)
        torch_output = ttnn.to_torch(tt_output)
        assert_with_pcc(torch_golden, torch_output, pcc=pcc)

    @pytest.mark.parametrize(
        "pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc",
        [
            (torch.bfloat16, ttnn.bfloat16, ttnn.float32, 0.99),
            (torch.float32, ttnn.float32, ttnn.int32, 0.99),
        ],
    )
    @pytest.mark.parametrize(
        "shard_layout",
        [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ],
    )
    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_sharded_to_interleaved(
        self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, shard_layout, input_shape
    ):
        """Sharded input -> interleaved output with dtype change."""
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_input_dtype)

        input_mem_config = _make_sharded_mem_config(shard_layout, input_shape)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=tt_input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_mem_config,
        )

        tt_output = ttnn.typecast(tt_input, dtype=tt_output_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        assert tt_output.dtype == tt_output_dtype
        assert not tt_output.is_sharded()

        cpu_input = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT)
        cpu_cast = ttnn.typecast(cpu_input, dtype=tt_output_dtype)
        torch_golden = ttnn.to_torch(cpu_cast)
        torch_output = ttnn.to_torch(tt_output)
        assert_with_pcc(torch_golden, torch_output, pcc=pcc)

    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_memory_config_change_same_dtype(self, device, input_shape):
        """Memory config change without dtype change."""
        torch.manual_seed(0)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output_mem_config = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)

        tt_output = ttnn.typecast(tt_input, dtype=ttnn.bfloat16, memory_config=output_mem_config)

        assert tt_output.dtype == ttnn.bfloat16
        assert tt_output.is_sharded()

        torch_output = ttnn.to_torch(tt_output)
        assert_with_pcc(torch_input, torch_output, pcc=0.9999)


# ---------------------------------------------------------------------------
# 3. Combined transforms: layout + memory config + dtype (all combos)
# ---------------------------------------------------------------------------
COMBINED_DTYPE_PAIRS = [
    (torch.bfloat16, ttnn.bfloat16, ttnn.float32, 0.99),
    (torch.float32, ttnn.float32, ttnn.bfloat16, 0.99),
    (torch.float32, ttnn.float32, ttnn.int32, 0.99),
    (torch.int, ttnn.int32, ttnn.float32, 0.99),
]


class TestTypecastCombinedTransforms:
    """All layout × memory direction combos with dtype change."""

    @pytest.mark.parametrize("pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc", COMBINED_DTYPE_PAIRS)
    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_tile_interleaved_to_rm_sharded(
        self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, input_shape
    ):
        """TILE interleaved -> RM sharded + dtype change."""
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_input_dtype)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=tt_input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output_mem_config = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)
        tt_output = ttnn.typecast(
            tt_input,
            dtype=tt_output_dtype,
            memory_config=output_mem_config,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        assert tt_output.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tt_output.dtype == tt_output_dtype
        assert tt_output.is_sharded()
        cpu_cast = ttnn.typecast(
            ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT), dtype=tt_output_dtype
        )
        assert_with_pcc(ttnn.to_torch(cpu_cast), ttnn.to_torch(tt_output), pcc=pcc)

    @pytest.mark.parametrize("pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc", COMBINED_DTYPE_PAIRS)
    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_tile_sharded_to_rm_interleaved(
        self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, input_shape
    ):
        """TILE sharded -> RM interleaved + dtype change."""
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_input_dtype)
        input_mem_config = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=tt_input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_mem_config,
        )
        tt_output = ttnn.typecast(
            tt_input,
            dtype=tt_output_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        assert tt_output.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tt_output.dtype == tt_output_dtype
        assert not tt_output.is_sharded()
        cpu_cast = ttnn.typecast(
            ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT), dtype=tt_output_dtype
        )
        assert_with_pcc(ttnn.to_torch(cpu_cast), ttnn.to_torch(tt_output), pcc=pcc)

    @pytest.mark.parametrize("pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc", COMBINED_DTYPE_PAIRS)
    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_rm_interleaved_to_tile_sharded(
        self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, input_shape
    ):
        """RM interleaved -> TILE sharded + dtype change."""
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_input_dtype)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=tt_input_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output_mem_config = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)
        tt_output = ttnn.typecast(
            tt_input,
            dtype=tt_output_dtype,
            memory_config=output_mem_config,
            output_layout=ttnn.TILE_LAYOUT,
        )
        assert tt_output.layout == ttnn.TILE_LAYOUT
        assert tt_output.dtype == tt_output_dtype
        assert tt_output.is_sharded()
        cpu_cast = ttnn.typecast(
            ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT), dtype=tt_output_dtype
        )
        assert_with_pcc(ttnn.to_torch(cpu_cast), ttnn.to_torch(tt_output), pcc=pcc)

    @pytest.mark.parametrize("pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc", COMBINED_DTYPE_PAIRS)
    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_rm_sharded_to_tile_interleaved(
        self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, input_shape
    ):
        """RM sharded -> TILE interleaved + dtype change."""
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_input_dtype)
        input_mem_config = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=tt_input_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_mem_config,
        )
        tt_output = ttnn.typecast(
            tt_input,
            dtype=tt_output_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
        )
        assert tt_output.layout == ttnn.TILE_LAYOUT
        assert tt_output.dtype == tt_output_dtype
        assert not tt_output.is_sharded()
        cpu_cast = ttnn.typecast(
            ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT), dtype=tt_output_dtype
        )
        assert_with_pcc(ttnn.to_torch(cpu_cast), ttnn.to_torch(tt_output), pcc=pcc)


# ---------------------------------------------------------------------------
# 4. Regression: existing behavior preserved when output_layout is None
# ---------------------------------------------------------------------------
class TestTypecastRegressionSameLayout:
    """Ensure existing behavior is unchanged when output_layout is not specified."""

    @pytest.mark.parametrize(
        "pt_input_dtype, tt_input_dtype, tt_output_dtype",
        [
            (torch.bfloat16, ttnn.bfloat16, ttnn.float32),
            (torch.float32, ttnn.float32, ttnn.int32),
        ],
    )
    @pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
    @pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
    def test_no_layout_change(self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, layout, mem_config):
        """No output_layout specified -> layout stays the same (existing behavior)."""
        torch.manual_seed(0)
        input_shape = [1, 1, 32, 32]
        torch_input = _make_torch_input(input_shape, pt_input_dtype)

        tt_input = ttnn.from_torch(
            torch_input, dtype=tt_input_dtype, layout=layout, device=device, memory_config=mem_config
        )

        tt_output = ttnn.typecast(tt_input, dtype=tt_output_dtype, memory_config=mem_config)

        assert tt_output.layout == layout
        assert tt_output.dtype == tt_output_dtype

        cpu_input = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT)
        cpu_cast = ttnn.typecast(cpu_input, dtype=tt_output_dtype)
        torch_golden = ttnn.to_torch(cpu_cast)
        torch_output = ttnn.to_torch(tt_output)
        assert_with_pcc(torch_golden, torch_output, pcc=0.99)


# ---------------------------------------------------------------------------
# 5. Host tensor path with output_layout
# ---------------------------------------------------------------------------
class TestTypecastHostTensorLayoutTransform:
    """Typecast on host tensors should respect output_layout."""

    @pytest.mark.parametrize(
        "input_layout, output_layout",
        [
            (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
            (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        ],
    )
    def test_host_layout_change_with_dtype(self, input_layout, output_layout):
        """Host tensor typecast with layout change."""
        torch.manual_seed(0)
        input_shape = [1, 1, 32, 32]
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

        cpu_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=input_layout)
        cpu_output = ttnn.typecast(cpu_input, dtype=ttnn.float32, output_layout=output_layout)

        assert cpu_output.layout == output_layout
        assert cpu_output.dtype == ttnn.float32


# ---------------------------------------------------------------------------
# 6. BFP types with layout transform (TILE-only types)
# ---------------------------------------------------------------------------
class TestTypecastBfpLayoutTransform:
    """BFP8/BFP4 types are tile-only. Typecast from BFP TILE input to RM non-BFP output."""

    @pytest.mark.parametrize(
        "pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc",
        [
            (torch.bfloat16, ttnn.bfloat8_b, ttnn.float32, 0.99),
            (torch.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, 0.99),
            (torch.bfloat16, ttnn.bfloat4_b, ttnn.float32, 0.98),
        ],
    )
    @pytest.mark.parametrize("input_shape", [[1, 1, 32, 32], [1, 1, 128, 128]])
    def test_bfp_tile_to_rm(self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, input_shape):
        """BFP TILE input -> non-BFP ROW_MAJOR output."""
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_input_dtype)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=tt_input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        tt_output = ttnn.typecast(
            tt_input,
            dtype=tt_output_dtype,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        assert tt_output.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tt_output.dtype == tt_output_dtype

        cpu_input = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT)
        cpu_cast = ttnn.typecast(cpu_input, dtype=tt_output_dtype)
        torch_golden = ttnn.to_torch(cpu_cast)
        torch_output = ttnn.to_torch(tt_output)
        assert_with_pcc(torch_golden, torch_output, pcc=pcc)

    @pytest.mark.parametrize(
        "pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc",
        [
            (torch.float32, ttnn.float32, ttnn.bfloat8_b, 0.99),
            (torch.bfloat16, ttnn.bfloat16, ttnn.bfloat8_b, 0.99),
        ],
    )
    @pytest.mark.parametrize("input_shape", [[1, 1, 32, 32], [1, 1, 128, 128]])
    def test_rm_to_bfp_tile(self, device, pt_input_dtype, tt_input_dtype, tt_output_dtype, pcc, input_shape):
        """Non-BFP ROW_MAJOR input -> BFP TILE output."""
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_input_dtype)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=tt_input_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        tt_output = ttnn.typecast(
            tt_input,
            dtype=tt_output_dtype,
            output_layout=ttnn.TILE_LAYOUT,
        )

        assert tt_output.layout == ttnn.TILE_LAYOUT
        assert tt_output.dtype == tt_output_dtype

        cpu_input = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT)
        cpu_cast = ttnn.typecast(cpu_input, dtype=tt_output_dtype)
        torch_golden = ttnn.to_torch(cpu_cast)
        torch_output = ttnn.to_torch(tt_output)
        assert_with_pcc(torch_golden, torch_output, pcc=pcc)


# ---------------------------------------------------------------------------
# 7. Reshard: sharded -> differently sharded with dtype change
# ---------------------------------------------------------------------------
class TestTypecastReshardTransform:
    """Typecast between different sharding strategies with dtype change."""

    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_height_sharded_to_width_sharded(self, device, input_shape):
        """HEIGHT_SHARDED -> WIDTH_SHARDED with dtype change."""
        torch.manual_seed(0)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

        input_mem_config = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)
        output_mem_config = _make_sharded_mem_config(ttnn.TensorMemoryLayout.WIDTH_SHARDED, input_shape)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_mem_config,
        )

        tt_output = ttnn.typecast(
            tt_input,
            dtype=ttnn.float32,
            memory_config=output_mem_config,
        )

        assert tt_output.dtype == ttnn.float32
        assert tt_output.is_sharded()

        cpu_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        cpu_cast = ttnn.typecast(cpu_input, dtype=ttnn.float32)
        torch_golden = ttnn.to_torch(cpu_cast)
        torch_output = ttnn.to_torch(tt_output)
        assert_with_pcc(torch_golden, torch_output, pcc=0.99)


# ---------------------------------------------------------------------------
# 8. Same-dtype layout and memory transforms
# ---------------------------------------------------------------------------
class TestTypecastSameDtypeTransforms:
    """Same dtype with layout and/or memory config changes."""

    @pytest.mark.parametrize(
        "pt_dtype, tt_dtype",
        [
            (torch.bfloat16, ttnn.bfloat16),
            (torch.float32, ttnn.float32),
            (torch.int, ttnn.int32),
        ],
    )
    @pytest.mark.parametrize("input_shape", [[1, 1, 32, 32], [1, 1, 128, 128]])
    def test_tile_to_rm_same_dtype(self, device, pt_dtype, tt_dtype, input_shape):
        """TILE -> RM without dtype change."""
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_dtype)
        tt_input = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.typecast(tt_input, dtype=tt_dtype, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        assert tt_output.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tt_output.dtype == tt_dtype
        assert_with_pcc(torch_input, ttnn.to_torch(tt_output), pcc=0.9999)

    @pytest.mark.parametrize(
        "pt_dtype, tt_dtype",
        [
            (torch.bfloat16, ttnn.bfloat16),
            (torch.float32, ttnn.float32),
            (torch.int, ttnn.int32),
        ],
    )
    @pytest.mark.parametrize("input_shape", [[1, 1, 32, 32], [1, 1, 128, 128]])
    def test_rm_to_tile_same_dtype(self, device, pt_dtype, tt_dtype, input_shape):
        """RM -> TILE without dtype change."""
        torch.manual_seed(0)
        torch_input = _make_torch_input(input_shape, pt_dtype)
        tt_input = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        tt_output = ttnn.typecast(tt_input, dtype=tt_dtype, output_layout=ttnn.TILE_LAYOUT)
        assert tt_output.layout == ttnn.TILE_LAYOUT
        assert tt_output.dtype == tt_dtype
        assert_with_pcc(torch_input, ttnn.to_torch(tt_output), pcc=0.9999)

    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_sharded_to_interleaved_same_dtype(self, device, input_shape):
        """Sharded -> interleaved without dtype change."""
        torch.manual_seed(0)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        input_mem_config = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)
        tt_input = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_mem_config
        )
        tt_output = ttnn.typecast(tt_input, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        assert not tt_output.is_sharded()
        assert_with_pcc(torch_input, ttnn.to_torch(tt_output), pcc=0.9999)

    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_tile_sharded_to_rm_interleaved_same_dtype(self, device, input_shape):
        """TILE sharded -> RM interleaved without dtype change (layout + memory)."""
        torch.manual_seed(0)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        input_mem_config = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)
        tt_input = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_mem_config
        )
        tt_output = ttnn.typecast(
            tt_input, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_layout=ttnn.ROW_MAJOR_LAYOUT
        )
        assert tt_output.layout == ttnn.ROW_MAJOR_LAYOUT
        assert not tt_output.is_sharded()
        assert_with_pcc(torch_input, ttnn.to_torch(tt_output), pcc=0.9999)

    @pytest.mark.parametrize("input_shape", [[1, 1, 128, 128]])
    def test_rm_interleaved_to_tile_sharded_same_dtype(self, device, input_shape):
        """RM interleaved -> TILE sharded without dtype change (layout + memory)."""
        torch.manual_seed(0)
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
        output_mem_config = _make_sharded_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, input_shape)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_output = ttnn.typecast(
            tt_input, dtype=ttnn.bfloat16, memory_config=output_mem_config, output_layout=ttnn.TILE_LAYOUT
        )
        assert tt_output.layout == ttnn.TILE_LAYOUT
        assert tt_output.is_sharded()
        assert_with_pcc(torch_input, ttnn.to_torch(tt_output), pcc=0.9999)


# ---------------------------------------------------------------------------
# 9. Negative cases: invalid output format/layout combinations
# ---------------------------------------------------------------------------
class TestTypecastNegativeCases:
    """Tests that invalid combinations are properly rejected."""

    def test_bfp8_to_bfp8_rm_rejected(self, device):
        """BFP8_B TILE -> BFP8_B RM should be rejected (tile-only format)."""
        torch_input = torch.randn([1, 1, 32, 32], dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        with pytest.raises(RuntimeError):
            ttnn.typecast(tt_input, dtype=ttnn.bfloat8_b, output_layout=ttnn.ROW_MAJOR_LAYOUT)

    def test_bfp4_to_bfp4_rm_rejected(self, device):
        """BFP4_B TILE -> BFP4_B RM should be rejected (tile-only format)."""
        torch_input = torch.randn([1, 1, 32, 32], dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
        with pytest.raises(RuntimeError):
            ttnn.typecast(tt_input, dtype=ttnn.bfloat4_b, output_layout=ttnn.ROW_MAJOR_LAYOUT)

    def test_rm_to_bfp8_rm_rejected(self, device):
        """RM float32 -> BFP8_B without layout change should fail (BFP can't be RM)."""
        torch_input = torch.randn([1, 1, 32, 32], dtype=torch.float32)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        with pytest.raises(RuntimeError):
            ttnn.typecast(tt_input, dtype=ttnn.bfloat8_b)

    def test_rm_to_bfp4_rm_rejected(self, device):
        """RM float32 -> BFP4_B without layout change should fail (BFP can't be RM)."""
        torch_input = torch.randn([1, 1, 32, 32], dtype=torch.float32)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        with pytest.raises(RuntimeError):
            ttnn.typecast(tt_input, dtype=ttnn.bfloat4_b)

    def test_bf16_tile_to_bfp8_rm_rejected(self, device):
        """bf16 TILE -> bfp8_b RM should be rejected (bfp8 can't be RM output)."""
        torch_input = torch.randn([1, 1, 32, 32], dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        with pytest.raises(RuntimeError):
            ttnn.typecast(tt_input, dtype=ttnn.bfloat8_b, output_layout=ttnn.ROW_MAJOR_LAYOUT)

    def test_bf16_tile_to_bfp4_rm_rejected(self, device):
        """bf16 TILE -> bfp4_b RM should be rejected (bfp4 can't be RM output)."""
        torch_input = torch.randn([1, 1, 32, 32], dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        with pytest.raises(RuntimeError):
            ttnn.typecast(tt_input, dtype=ttnn.bfloat4_b, output_layout=ttnn.ROW_MAJOR_LAYOUT)

    def test_fp32_tile_to_bfp8_rm_rejected(self, device):
        """fp32 TILE -> bfp8_b RM should be rejected (bfp8 can't be RM output)."""
        torch_input = torch.randn([1, 1, 32, 32], dtype=torch.float32)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        with pytest.raises(RuntimeError):
            ttnn.typecast(tt_input, dtype=ttnn.bfloat8_b, output_layout=ttnn.ROW_MAJOR_LAYOUT)

    def test_int32_tile_to_bfp8_rm_rejected(self, device):
        """int32 TILE -> bfp8_b RM should be rejected (bfp8 can't be RM output)."""
        torch_input = torch.randint(0, 100, [1, 1, 32, 32], dtype=torch.int)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
        with pytest.raises(RuntimeError):
            ttnn.typecast(tt_input, dtype=ttnn.bfloat8_b, output_layout=ttnn.ROW_MAJOR_LAYOUT)
