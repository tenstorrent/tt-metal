# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import re

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc
from tests.ttnn.unit_tests.base_functionality.test_narrow import assert_quality
from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    UnreadableDocstring,
    check_readable,
    docstring_dtype_layout_params,
    docstring_interleaved_buffer_type_params,
    interleaved_memory_config,
)
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact, full_range_input


def _docstring_sharded_output_params(op):
    """One ``pytest.param(memory_layout, buffer_type)`` per shard layout and buffer type on the ``- Output:`` item of
    the op's Memory Support list, which reads like ``HEIGHT_SHARDED and BLOCK_SHARDED in L1; HEIGHT_SHARDED in DRAM``.

    Every word before ``in`` other than ``and`` must name a ``ttnn.TensorMemoryLayout``, and the word after it a
    ``ttnn.BufferType``. If the item can't be read, a single case fails with the reason.
    """
    try:
        item = None
        for line in op.__doc__.split("Memory Support:", 1)[1].splitlines():
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if item is None:
                if stripped.startswith("- Output:"):
                    item, item_indent = stripped[len("- Output:") :], indent
            elif stripped and not stripped.startswith("- ") and indent > item_indent:
                item += " " + stripped
            else:
                break
        assert item, "no '- Output:' Memory Support item"
        params = []
        for clause in item.split(";"):
            match = re.fullmatch(r"\s*(.+?)\s+in\s+(\w+)\s*", clause)
            assert match, f"no '<layouts> in <buffer type>' in {clause!r}"
            layout_names, buffer_name = match.groups()
            buffer_type = getattr(ttnn.BufferType, buffer_name, None)
            assert isinstance(buffer_type, ttnn.BufferType), f"not a ttnn.BufferType: {buffer_name}"
            names = [name for name in re.findall(r"[^\s,]+", layout_names) if name != "and"]
            assert names, f"no shard layouts in {clause!r}"
            for name in names:
                memory_layout = getattr(ttnn.TensorMemoryLayout, name, None)
                assert isinstance(memory_layout, ttnn.TensorMemoryLayout), f"not a ttnn.TensorMemoryLayout: {name}"
                params.append(pytest.param(memory_layout, buffer_type, id=f"{name}-{buffer_name}"))
        return params
    except Exception as error:
        reason = UnreadableDocstring(f"could not read the Output Memory Support item from the {op} docstring: {error}")
        return [pytest.param(reason, reason, id="docstring-unreadable")]


def _docstring_output_dtype_pairs(op):
    """One ``pytest.param(input_dtype, output_dtype)`` per ordered pair of the dtypes that the op's ``output_dtype``
    line lists in "input of <dtypes> can be converted": the dtypes it says an input converts between. Each name must be
    a ``ttnn.DataType``."""
    try:
        line = next(line for line in op.__doc__.splitlines() if line.strip().startswith("output_dtype ("))
        match = re.search(r"input of (.+?) can be converted", line)
        assert match, f"no 'input of <dtypes> can be converted' on: {line.strip()!r}"
        names = re.split(r",\s*|\s+or\s+", match.group(1))
        unknown = [name for name in names if not isinstance(getattr(ttnn.DataType, name, None), ttnn.DataType)]
        assert not unknown, f"not a ttnn.DataType: {unknown} on: {line.strip()!r}"
        assert len(names) > 1, f"fewer than two dtypes on: {line.strip()!r}"
        return [
            pytest.param(getattr(ttnn.DataType, a), getattr(ttnn.DataType, b), id=f"{a}-to-{b}")
            for a in names
            for b in names
            if a != b
        ]
    except Exception as error:
        reason = UnreadableDocstring(f"could not read the output_dtype line from the {op} docstring: {error}")
        return [pytest.param(reason, reason, id="docstring-unreadable")]


# Fraction bits of a block float value that has its block's exponent.
_BLOCK_FLOAT_FRACTION_BITS = {ttnn.bfloat8_b: 6, ttnn.bfloat4_b: 2}


def _convertible_input(shape, input_dtype, output_dtype):
    """Values that both dtypes hold exactly, so converting between them keeps every value, and no NaN, as the
    output_dtype line asks. With a block float, the values are +-(1 + k / 2**bits): they share one exponent and fit
    the block float's fraction bits."""
    bits = [_BLOCK_FLOAT_FRACTION_BITS[d] for d in (input_dtype, output_dtype) if d in _BLOCK_FLOAT_FRACTION_BITS]
    if not bits:
        return full_range_input(shape, ttnn.bfloat16, nan=False).float()
    torch.manual_seed(0)
    magnitude = 1 + torch.randint(0, 2 ** min(bits), shape) / 2 ** min(bits)
    return torch.where(torch.randint(0, 2, shape, dtype=torch.bool), -magnitude, magnitude)


@pytest.mark.parametrize(
    "first_dtype, second_dtype",
    [
        (ttnn.bfloat8_b, ttnn.bfloat16),
        (ttnn.bfloat8_b, ttnn.float32),
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat8_b),
        (ttnn.float32, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize("input_in_l1", [True, False])
@pytest.mark.parametrize("keep_l1_aligned", [True, False])
def test_interleaved_to_sharded_hash(device, first_dtype, second_dtype, input_in_l1, keep_l1_aligned):
    # Sample tensor size and shard config
    input_tensor_shape = (1, 1, 512, 512)
    input_memory_config = ttnn.L1_MEMORY_CONFIG if input_in_l1 else ttnn.DRAM_MEMORY_CONFIG

    # L1 grid and calculations
    l1_core_grid_x = 4
    l1_core_grid_y = 4
    l1_shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(l1_core_grid_x - 1, l1_core_grid_y - 1))}
    )
    l1_h, l1_w = input_tensor_shape[2] // l1_core_grid_x, input_tensor_shape[3] // l1_core_grid_y

    # Shard spec and mem config
    shard_spec = ttnn.ShardSpec(l1_shard_grid, [l1_h, l1_w], ttnn.ShardOrientation.ROW_MAJOR)

    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    # Create input tensors and send to device
    input_tensor_torch = torch.randn(input_tensor_shape)
    input_tensor = ttnn.from_torch(input_tensor_torch, first_dtype, layout=ttnn.TILE_LAYOUT)
    input_tensor_device = ttnn.allocate_tensor_on_device(
        input_tensor.shape, input_tensor.dtype, input_tensor.layout, device, input_memory_config
    )
    ttnn.copy_host_to_device_tensor(input_tensor, input_tensor_device)

    # Do interleaved to sharded op on device several times to load the program from cache
    for iter in range(0, 5):
        output_tensor = ttnn.interleaved_to_sharded(
            input_tensor_device, sharded_mem_config, first_dtype, keep_l1_aligned=keep_l1_aligned
        )
        pcc_passed_a, pcc_message_a = assert_with_pcc(input_tensor_torch, ttnn.to_torch(output_tensor), pcc=0.9999)

        output_tensor = ttnn.interleaved_to_sharded(
            input_tensor_device, sharded_mem_config, second_dtype, keep_l1_aligned=keep_l1_aligned
        )
        pcc_passed_b, pcc_message_b = assert_with_pcc(input_tensor_torch, ttnn.to_torch(output_tensor), pcc=0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, shard_grid",
    [
        [
            [2, 2, 128, 64],
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        [
            [1, 1, 416, 64],
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_interleaved_to_dram_height_sharded(
    device, dtype, layout, tensor_shape, shard_shape, shard_grid, shard_orientation
):
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b not supported for i2s row-major")

    # Output memory config
    output_shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, output_shard_spec
    )

    # Test
    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=layout)
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)
    ttnn_output_tensor = ttnn.interleaved_to_sharded(ttnn_input_tensor, output_mem_config)

    assert_quality(torch_input_tensor, ttnn.to_torch(ttnn_output_tensor), dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, shard_grid",
    [
        [
            [2, 1, 32, 512],
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        [
            [1, 1, 64, 416],
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_interleaved_to_dram_width_sharded(
    device, dtype, layout, tensor_shape, shard_shape, shard_grid, shard_orientation
):
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b not supported for i2s row-major")

    # Output memory config
    output_shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, output_shard_spec
    )

    # Test
    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=layout)
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)
    ttnn_output_tensor = ttnn.interleaved_to_sharded(ttnn_input_tensor, output_mem_config)

    assert_quality(torch_input_tensor, ttnn.to_torch(ttnn_output_tensor), dtype)


@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        [ttnn.bfloat8_b, ttnn.float32],
        [ttnn.float32, ttnn.bfloat8_b],
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "tensor_shape, shard_type, shard_shape, shard_grid",
    [
        [
            [1, 1, 416, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        [
            [1, 1, 64, 416],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_interleaved_to_dram_sharded_convert_dtype(
    device, in_dtype, out_dtype, layout, tensor_shape, shard_type, shard_shape, shard_grid, shard_orientation
):
    # Output memory config
    output_shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(shard_type, ttnn.BufferType.DRAM, output_shard_spec)

    # Test
    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=in_dtype, layout=layout)
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)
    ttnn_output_tensor = ttnn.interleaved_to_sharded(ttnn_input_tensor, output_mem_config, out_dtype)

    effective_dtype = ttnn.bfloat8_b if (in_dtype == ttnn.bfloat8_b or out_dtype == ttnn.bfloat8_b) else out_dtype
    assert_quality(torch_input_tensor, ttnn.to_torch(ttnn_output_tensor), effective_dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "tensor_shape, shard_type, shard_shape, shard_grid",
    [
        [
            [1, 1, 416, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (128, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        [
            [1, 1, 64, 416],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_interleaved_to_dram_sharded_via_to_memory_layout(
    device, dtype, layout, input_mem_config, tensor_shape, shard_type, shard_shape, shard_grid, shard_orientation
):
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b not supported for i2s row-major")

    # Output memory config
    output_shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    output_mem_config = ttnn.MemoryConfig(shard_type, ttnn.BufferType.DRAM, output_shard_spec)

    # Test
    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=dtype, layout=layout, device=device, memory_config=input_mem_config
    )
    ttnn_output_tensor = ttnn.to_memory_config(ttnn_input_tensor, output_mem_config)

    assert_quality(torch_input_tensor, ttnn.to_torch(ttnn_output_tensor), dtype)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "tensor_shape, nd_shard_shape, shard_grid",
    [
        # HEIGHT_SHARDED equivalent: shard covers full width, 4 shards along height
        [
            [1, 1, 128, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        # WIDTH_SHARDED equivalent: shard covers full height, 4 shards along width
        [
            [1, 1, 64, 256],
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        # BLOCK_SHARDED equivalent: shard splits both dims, 2x2 grid
        [
            [1, 1, 128, 128],
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ],
        # 3D ND shard folding into HEIGHT_SHARDED: batch dim folds into shard height
        [
            [4, 32, 64],
            [2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ],
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
def test_interleaved_to_sharded_nd_with_equivalent_2d(
    device, layout, tensor_shape, nd_shard_shape, shard_grid, shard_orientation
):
    nd_shard_spec = ttnn.NdShardSpec(nd_shard_shape, shard_grid, orientation=shard_orientation)
    output_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)

    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=layout)
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)
    ttnn_output_tensor = ttnn.interleaved_to_sharded(ttnn_input_tensor, output_mem_config)

    assert_equal(torch_input_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize(
    "tensor_shape, nd_shard_shape, shard_grid",
    [
        # Sharding across non-adjacent dims: volume mismatch prevents 2D flattening
        [
            [6, 32, 64],
            [3, 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        # Sharding batch and spatial with non-trivial batch shard: can't fold to 2D
        [
            [4, 3, 32, 64],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
        ],
    ],
)
def test_interleaved_to_sharded_rejects_pure_nd(device, tensor_shape, nd_shard_shape, shard_grid, expect_error):
    nd_shard_spec = ttnn.NdShardSpec(nd_shard_shape, shard_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)

    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)

    with expect_error(RuntimeError, r"interleaved_to_sharded does not support ND sharding"):
        ttnn.interleaved_to_sharded(ttnn_input_tensor, output_mem_config)


@pytest.mark.parametrize(
    "tensor_shape, nd_shard_shape, nd_shard_grid, output_shard_shape, output_shard_grid",
    [
        # Same pure-ND config as above, paired with a legal 2D-sharded pre-allocated output:
        # 6*32 = 192 rows of 64, height-sharded 32 rows per core across 6 cores.
        [
            [6, 32, 64],
            [3, 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
        ],
    ],
)
def test_interleaved_to_sharded_rejects_pure_nd_with_preallocated_output(
    device, tensor_shape, nd_shard_shape, nd_shard_grid, output_shard_shape, output_shard_grid, expect_error
):
    """A pure-ND output config must be rejected even when a pre-allocated output is supplied.

    validate_inputs used to read the ND check off the caller's own spec, so this was wrongly accepted.
    """
    nd_shard_spec = ttnn.NdShardSpec(nd_shard_shape, nd_shard_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)

    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)

    # Valid on its own terms, so the ND output config is the only thing left to object to.
    preallocated_output = ttnn.allocate_tensor_on_device(
        ttnn_input_tensor.shape,
        ttnn_input_tensor.dtype,
        ttnn_input_tensor.layout,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(output_shard_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )

    with expect_error(RuntimeError, r"interleaved_to_sharded does not support ND sharding"):
        ttnn.interleaved_to_sharded(ttnn_input_tensor, output_mem_config, preallocated_output=preallocated_output)


def _single_core_l1(layout, shard_shape):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    return ttnn.MemoryConfig(
        layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )


def _zero_width_tile(device):
    return ttnn.from_torch(torch.zeros(1, 1, 32, 0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)


# Each of these used to reach an integer division by the input width (or a truncated num_slices) on the host.
@pytest.mark.parametrize(
    "run, message",
    [
        (
            lambda device: ttnn.interleaved_to_sharded(
                _zero_width_tile(device),
                ttnn.CoreCoord(1, 1),
                [32, 32],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
            "Input tensor width must be greater than 0",
        ),
        (
            lambda device: ttnn.interleaved_to_sharded(
                _zero_width_tile(device), _single_core_l1(ttnn.TensorMemoryLayout.WIDTH_SHARDED, (32, 32))
            ),
            "Input tensor width must be greater than 0",
        ),
        (
            lambda device: ttnn.interleaved_to_sharded_partial(
                _zero_width_tile(device),
                ttnn.CoreCoord(1, 1),
                [32, 32],
                1,
                0,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
            "Input tensor width must be greater than 0",
        ),
        (
            lambda device: ttnn.interleaved_to_sharded_partial(
                ttnn.from_torch(torch.randn(1, 1, 64, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device),
                ttnn.CoreCoord(1, 1),
                [32, 64],
                2**32,
                0,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
            "num_slices must be at most UINT32_MAX",
        ),
        (
            lambda device: ttnn.sharded_to_interleaved_partial(
                ttnn.from_torch(
                    torch.randn(1, 1, 32, 32).bfloat16(),
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=_single_core_l1(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (32, 32)),
                ),
                _zero_width_tile(device),
                1,
                0,
            ),
            "Cache tensor width must be greater than 0",
        ),
    ],
    ids=[
        "i2s_zero_width",
        "i2s_memory_config_zero_width",
        "i2s_partial_zero_width",
        "i2s_partial_num_slices_wraps",
        "s2i_partial_zero_width_cache",
    ],
)
def test_sharding_rejects_zero_width_and_wrapping_slices(device, expect_error, run, message):
    with expect_error(RuntimeError, message):
        run(device)


# _I2S_PARTIAL_SHAPE has 2 * 3 * 64 = 384 rows of width 128, copied in 3 slices of 128 rows that cross the batch and
# channel boundaries. Shard shape and core grid size (x, y) of each shard layout, for one 128 x 128 slice.
_I2S_PARTIAL_SHAPE = (2, 3, 64, 128)
_I2S_PARTIAL_NUM_SLICES = 3
_I2S_PARTIAL_SHARDS = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: ((32, 128), (2, 2)),
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: ((128, 64), (2, 1)),
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: ((64, 64), (2, 2)),
}


def _i2s_partial_slices(input_tensor, memory_layout, buffer_type, output_dtype=None):
    """Copies every slice of ``input_tensor`` with ttnn.interleaved_to_sharded_partial, checks each against its rows of
    the input, and returns the slices. When ``output_dtype`` changes the dtype, zeros are compared by value."""
    shard_shape, (grid_x, grid_y) = _I2S_PARTIAL_SHARDS[memory_layout]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    memory_config = ttnn.MemoryConfig(
        memory_layout, buffer_type, ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )
    rows = ttnn.to_torch(input_tensor).reshape(-1, _I2S_PARTIAL_SHAPE[-1])
    slice_height = rows.shape[0] // _I2S_PARTIAL_NUM_SLICES
    slices = []
    for slice_index in range(_I2S_PARTIAL_NUM_SLICES):
        output = ttnn.interleaved_to_sharded_partial(
            input_tensor,
            ttnn.CoreCoord(grid_x, grid_y),
            shard_shape,
            _I2S_PARTIAL_NUM_SLICES,
            slice_index,
            memory_layout,
            ttnn.ShardOrientation.ROW_MAJOR,
            output_dtype=output_dtype,
        )

        assert output.dtype == (input_tensor.dtype if output_dtype is None else output_dtype)
        assert output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == memory_config
        expected = rows[slice_index * slice_height : (slice_index + 1) * slice_height]
        assert_exact(
            expected.reshape(1, 1, slice_height, -1),
            ttnn.to_torch(output),
            signed_zeros=output_dtype in (None, input_tensor.dtype),
        )
        slices.append(output)
    return slices


@pytest.mark.parametrize(
    "memory_layout, output_buffer_type", _docstring_sharded_output_params(ttnn.interleaved_to_sharded_partial)
)
@pytest.mark.parametrize(
    "buffer_type", docstring_interleaved_buffer_type_params(ttnn.interleaved_to_sharded_partial, label="Input")
)
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.interleaved_to_sharded_partial))
def test_interleaved_to_sharded_partial_docstring_supported_inputs(
    dtype, layout, buffer_type, memory_layout, output_buffer_type, device
):
    """Copies every slice of an input of each dtype, layout and interleaved buffer type the
    ttnn.interleaved_to_sharded_partial docstring lists into each sharded output it lists, then writes the slices back
    into a tensor of the input's shape with ttnn.sharded_to_interleaved_partial."""
    check_readable(dtype, layout, buffer_type, memory_layout, output_buffer_type)
    memory_config = interleaved_memory_config(buffer_type)
    input_tensor = ttnn.from_torch(
        full_range_input(_I2S_PARTIAL_SHAPE, dtype),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )
    output_tensor = ttnn.from_torch(
        full_range_input(_I2S_PARTIAL_SHAPE, dtype, seed=1),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )
    for slice_index, sharded_slice in enumerate(_i2s_partial_slices(input_tensor, memory_layout, output_buffer_type)):
        ttnn.sharded_to_interleaved_partial(sharded_slice, output_tensor, _I2S_PARTIAL_NUM_SLICES, slice_index)

    assert_exact(ttnn.to_torch(input_tensor), ttnn.to_torch(output_tensor))


@pytest.mark.parametrize(
    "buffer_type", docstring_interleaved_buffer_type_params(ttnn.interleaved_to_sharded_partial, label="Input")
)
@pytest.mark.parametrize(
    "input_dtype, output_dtype", _docstring_output_dtype_pairs(ttnn.interleaved_to_sharded_partial)
)
def test_interleaved_to_sharded_partial_docstring_output_dtype(input_dtype, output_dtype, buffer_type, device):
    """Copies every slice of a TILE input into every other dtype the ttnn.interleaved_to_sharded_partial output_dtype
    line names. The input values are exact in both dtypes, so every slice keeps them."""
    check_readable(input_dtype, output_dtype, buffer_type)
    input_tensor = ttnn.from_torch(
        _convertible_input(_I2S_PARTIAL_SHAPE, input_dtype, output_dtype),
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=interleaved_memory_config(buffer_type),
    )
    _i2s_partial_slices(input_tensor, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_dtype)
