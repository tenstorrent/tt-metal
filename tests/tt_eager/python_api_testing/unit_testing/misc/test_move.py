# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn

from models.common.utility_functions import is_wormhole_b0
import ttnn
from models.common.utility_functions import (
    comp_pcc,
)
import torch


def run_move_op(
    test_id, shape, layout, dtype, in0_mem_config, output_mem_config, device, move_op=ttnn.move, dummy_shape=None
):
    """
    For non_overlap, multi-core is run for num_tiles > 1.
    """
    torch.manual_seed(1234)

    # Dummy tensor to shift input tensor in memory
    dummy_layout = layout
    if dummy_shape is not None:
        dummy_layout = ttnn.ROW_MAJOR_LAYOUT
    elif test_id == 0:
        dummy_shape = [1, 1, 32, 32]
    elif test_id == 1:
        dummy_shape = shape  # This will allow output and input buffers to not overlap
    else:
        raise NotImplementedError(f"Unknown test id: {test_id}!")

    # Create tensor with appropriate dtype
    if dtype == ttnn.uint8:
        dummy_tensor = torch.randint(0, 256, dummy_shape, dtype=torch.uint8)
        torch_tensor = torch.randint(0, 256, shape, dtype=torch.uint8)
    elif dtype == ttnn.uint16:
        dummy_tensor = torch.randint(0, 2**16, dummy_shape, dtype=torch.uint16)
        torch_tensor = torch.randint(0, 2**16, shape, dtype=torch.uint16)
    elif dtype == ttnn.uint32:
        dummy_tensor = torch.randint(0, 2**32, dummy_shape, dtype=torch.uint32)
        torch_tensor = torch.randint(0, 2**32, shape, dtype=torch.uint32)
    elif dtype == ttnn.int32:
        dummy_tensor = torch.randint(-(2**31), 2**31, dummy_shape, dtype=torch.int32)
        torch_tensor = torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    else:
        dummy_tensor = torch.randn(dummy_shape)
        torch_tensor = torch.randn(shape)

    tt_dummy_tensor = ttnn.Tensor(dummy_tensor, dtype).to(dummy_layout).to(device, in0_mem_config)
    tt_tensor = ttnn.Tensor(torch_tensor, dtype).to(layout).to(device, in0_mem_config)

    # Free up dummy tensor from memory to make available to move
    tt_dummy_tensor.deallocate()

    output = move_op(tt_tensor, memory_config=output_mem_config)

    tt_host_rm = output.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    pyt_got_back_rm = tt_host_rm.to_torch()

    passing_pcc, output_pcc = comp_pcc(pyt_got_back_rm, torch_tensor, 0.99)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc


shapes = [
    [1, 1, 32, 32],
    [1, 3, 320, 384],
]
if is_wormhole_b0():
    del shapes[1:]


@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "output_mem_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.uint8,
        ttnn.uint16,
        ttnn.uint32,
        ttnn.int32,
    ),
    ids=["BFLOAT4_B", "BFLOAT8_B", "BFLOAT16", "FLOAT32", "UINT8", "UINT16", "UINT32", "INT32"],
)
@pytest.mark.parametrize("layout", (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT), ids=["TILE", "RM"])
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("test_id", (0, 1), ids=["overlap", "non_overlap"])
def test_move_op(test_id, shape, layout, dtype, in0_mem_config, output_mem_config, device):
    if in0_mem_config.buffer_type != ttnn.BufferType.L1:
        pytest.skip("Skipping test for non-L1 buffer type")
    run_move_op(test_id, shape, layout, dtype, in0_mem_config, output_mem_config, device)


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32, ttnn.uint8, ttnn.uint16, ttnn.uint32, ttnn.int32],
)
def test_move_op_with_program_cache(dtype, device):
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    layout = ttnn.TILE_LAYOUT
    shape = [1, 3, 320, 384]
    dummy_shape = [1, 1, 32, 32]

    # Single core because of overlap
    for _ in range(2):
        run_move_op(0, shape, layout, dtype, mem_config, mem_config, device)
        # Create dummy tensor with appropriate dtype
        if dtype == ttnn.uint8:
            py_dummy_tensor = torch.randint(0, 256, dummy_shape, dtype=torch.uint8)
        elif dtype == ttnn.uint16:
            py_dummy_tensor = torch.randint(0, 2**16, dummy_shape, dtype=torch.uint16)
        elif dtype == ttnn.uint32:
            py_dummy_tensor = torch.randint(0, 2**32, dummy_shape, dtype=torch.uint32)
        elif dtype == ttnn.int32:
            py_dummy_tensor = torch.randint(-(2**31), 2**31, dummy_shape, dtype=torch.int32)
        else:
            py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    # Multi-core
    for _ in range(2):
        run_move_op(1, shape, layout, dtype, mem_config, mem_config, device)
        # Create dummy tensor with appropriate dtype
        if dtype == ttnn.uint8:
            py_dummy_tensor = torch.randint(0, 256, dummy_shape, dtype=torch.uint8)
        elif dtype == ttnn.uint16:
            py_dummy_tensor = torch.randint(0, 2**16, dummy_shape, dtype=torch.uint16)
        elif dtype == ttnn.uint32:
            py_dummy_tensor = torch.randint(0, 2**32, dummy_shape, dtype=torch.uint32)
        elif dtype == ttnn.int32:
            py_dummy_tensor = torch.randint(-(2**31), 2**31, dummy_shape, dtype=torch.int32)
        else:
            py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 2


MOVE_OPS = [ttnn.move, ttnn.experimental.quasar.move]
MOVE_OP_IDS = ["ttnn", "quasar"]
# Freeing a tiny-page dummy makes the output overlap the input, forcing the overlap path for few tiles.
TINY_DUMMY = [1, 1, 1, 8]


@pytest.mark.parametrize("move_op", MOVE_OPS, ids=MOVE_OP_IDS)
@pytest.mark.parametrize("shape", [[1, 1, 160, 32], [1, 1, 32, 32]], ids=["single_column", "single_core"])
def test_move_op_overlap_narrow_core_range(device, move_op, shape):
    # Regression: a single-column or single-core grid built an invalid CoreRange.
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    run_move_op(0, shape, ttnn.TILE_LAYOUT, ttnn.bfloat16, mem_config, mem_config, device, move_op, TINY_DUMMY)


@pytest.mark.parametrize("move_op", MOVE_OPS, ids=MOVE_OP_IDS)
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM], ids=["L1", "DRAM"])
def test_move_op_overlap_row_major_unaligned_page_size(device, move_op, buffer_type):
    # Regression: a 48 B page (16 mod 32) was rounded differently from the CB's total size.
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)
    run_move_op(0, [1, 1, 64, 24], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, mem_config, mem_config, device, move_op)
