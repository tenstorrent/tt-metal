# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import re
import pytest
import torch
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_equal

# the block-interleaved (width+height) factory is chosen only when num_tiles_per_row > 32
_WH_TILE_THRESHOLD = 32
_TILE = 32
_CORE_RANGE_RE = re.compile(r"\[(\d+)-(\d+) - (\d+)-(\d+)\]")


def _num_cores_from_graph(captured_graph):
    """Count unique cores that received a CB or dataflow buffer during graph capture."""
    cores = set()
    for node in captured_graph:
        if node["node_type"] not in ("circular_buffer_allocate", "dataflow_buffer_allocate"):
            continue
        core_range_set = node.get("params", {}).get("core_range_set", "")
        for x1, y1, x2, y2 in _CORE_RANGE_RE.findall(core_range_set):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            for x in range(x1, x2 + 1):
                for y in range(y1, y2 + 1):
                    cores.add((x, y))
    return len(cores)


def _height_only_core_count(shape):
    """Cores used if the op only split along height (one core per tile-row)."""
    return shape[-2] // _TILE


@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_untilize_with_padded_input(mem_config, device):
    """Regression test: untilize on a padded TILE tensor must discard padding.

    Issue #36765: untilize kept the padded values in the buffer (padded_shape
    stayed larger than logical_shape), silently corrupting downstream ops. After
    the fix, untilize routes to untilize_with_unpadding so the output buffer holds
    only the logical data (padded_shape == logical_shape).
    """
    torch_tensor = torch.randn(1, 1, 33, 33, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, memory_config=mem_config, device=device)

    logger.debug(f"Input logical_shape: {tt_tensor.shape}")
    logger.debug(f"Input padded_shape:  {tt_tensor.padded_shape}")

    untilized = ttnn.untilize(tt_tensor)

    logger.debug(f"After untilize:")
    logger.debug(f"  logical_shape: {untilized.shape}")
    logger.debug(f"  padded_shape:  {untilized.padded_shape}")
    logger.debug(f"  layout:        {untilized.layout}")

    if untilized.padded_shape == untilized.shape:
        torch_output = ttnn.to_torch(untilized)
        logger.debug(f"  to_torch() shape: {torch_output.shape}")
        logger.debug("   No padding in output buffer (clean data)")
    else:
        raise AssertionError("Output has padding in buffer")

    assert_equal(torch_tensor, ttnn.to_torch(untilized))


@pytest.mark.parametrize(
    "shape, output_end, expect_width_parallel",
    [
        # 100 tiles wide, 4 tile-rows. Height-only would use 4 cores.
        ([1, 1, 128, 3200], [0, 0, 119, 3167], True),
        # 48 tiles wide, one tile-row. Height-only would use 1 core.
        ([1, 1, 32, 1536], [0, 0, 31, 1535], True),
        # Just above threshold_row_block (33 tiles).
        ([1, 1, 32, (_WH_TILE_THRESHOLD + 1) * _TILE], [0, 0, 31, 1039], True),
        # At the cutoff (32 tiles): height-only path, 1 core.
        ([1, 1, 32, _WH_TILE_THRESHOLD * _TILE], [0, 0, 31, 991], False),
    ],
)
def test_untilize_with_unpadding_wide(device, shape, output_end, expect_width_parallel):
    """Wide untilize_with_unpadding must use more cores than a height-only split (#17537)."""
    torch.manual_seed(42)
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)

    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    tile_tensor = ttnn.from_torch(
        torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        untilized = ttnn.untilize_with_unpadding(
            tile_tensor, output_tensor_end=output_end, memory_config=output_memory_config
        )
    finally:
        captured_graph = ttnn.graph.end_graph_capture()

    num_cores = _num_cores_from_graph(captured_graph)
    height_only_cores = _height_only_core_count(shape)
    logger.info(
        f"untilize_with_unpadding shape={shape} used {num_cores} cores " f"(height-only would use {height_only_cores})"
    )
    assert num_cores > 0, f"graph capture did not report any cores for shape={shape}"
    if expect_width_parallel:
        assert num_cores > height_only_cores, (
            f"shape={shape} used {num_cores} cores; width-parallel factory should beat "
            f"the height-only split of {height_only_cores} cores"
        )
    else:
        assert num_cores == height_only_cores, (
            f"shape={shape} is at the {_WH_TILE_THRESHOLD}-tile cutoff and should stay "
            f"on the height-only path ({height_only_cores} cores), got {num_cores}"
        )

    result = ttnn.to_torch(untilized)
    slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
    assert_equal(torch_tensor[slices], result)
