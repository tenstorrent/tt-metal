# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def _is_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


def _get_torch_data_type(tt_dtype) -> torch.dtype:
    mapping = {
        ttnn.bfloat4_b: torch.bfloat16,  # approximate fallback
        ttnn.bfloat8_b: torch.bfloat16,  # approximate fallback
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.uint8: torch.int32,
        ttnn.uint16: torch.int32,  # torch has limited uint16 support
        ttnn.uint32: torch.int64,  # torch has no uint32, fallback
        ttnn.int32: torch.int32,
    }

    if tt_dtype == ttnn.DataType.INVALID:
        raise ValueError("INVALID data type provided.")

    torch_dtype = mapping.get(tt_dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported or unknown TTNN data type: {tt_dtype}")

    return torch_dtype


def _create_random_torch_tensors(tensor_shape: tuple, tt_dtype, num_tensors: int):
    torch.manual_seed(0)
    torch_dtype = _get_torch_data_type(tt_dtype)

    results = []
    for _ in range(num_tensors):
        if _is_float_type(tt_dtype):
            t = torch.rand(tensor_shape, dtype=torch_dtype)
        else:
            t = torch.randint(0, 100, tensor_shape, dtype=torch_dtype)
        results.append(t)

    return tuple(results)


def _convert_torch_to_ttnn(
    torch_tensors: tuple,
    device,
    tt_dtype,
    layout,
    mem_config,
):
    ttnn_results = []
    for t in torch_tensors:
        tt_tensor = ttnn.from_torch(
            t,
            layout=layout,
            dtype=tt_dtype,
            memory_config=mem_config,
            device=device,
        )
        tt_tensor = ttnn.to_device(tt_tensor, device)
        ttnn_results.append(tt_tensor)

    return tuple(ttnn_results)


def _ttt_where_test_impl(
    device, tensor_shape: tuple, tt_dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, mem_config=None
):
    torch_inputs = _create_random_torch_tensors(tensor_shape, tt_dtype, 3)

    condition_torch, true_torch, false_torch = torch_inputs
    golden_fn = ttnn.get_golden_function(ttnn.where)
    torch_output_tensor = golden_fn(condition_torch.to(torch.bool), true_torch, false_torch)

    condition, true_values, false_values = _convert_torch_to_ttnn(torch_inputs, device, tt_dtype, layout, mem_config)

    output_tensor = ttnn.experimental.where(condition, true_values, false_values)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [16, 32, 64, 65, 1024])
@pytest.mark.parametrize("w", [16, 32, 64, 65, 1024])
def test_ttt_where(device, h, w):
    _ttt_where_test_impl(device, (h, w))


@pytest.mark.parametrize("d5", [16])
@pytest.mark.parametrize("d4", [16])
@pytest.mark.parametrize("d3", [16])
@pytest.mark.parametrize("h", [16])
@pytest.mark.parametrize("w", [16])
def test_ttt_where_multidim(device, d5, d4, d3, h, w):
    _ttt_where_test_impl(device, (d5, d4, d3, h, w))


@pytest.mark.parametrize("tt_dtype", [ttnn.uint8, ttnn.uint16, ttnn.int32, ttnn.uint32])
def test_ttt_where_int_types(device, tt_dtype):
    _ttt_where_test_impl(device, (64, 64), tt_dtype=tt_dtype)


@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_ttt_where_float_types(device, tt_dtype):
    _ttt_where_test_impl(device, (64, 64), tt_dtype=tt_dtype)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_ttt_where_layouts(device, layout):
    # Missing parameters in from_torch() for layout test coverage: tile, pad_value
    _ttt_where_test_impl(device, (64, 64), layout=layout)


# TODO: Too many parameters in sharded memory config test — should be moved out into its own file.
# block_sharded_mem_config = ttnn.create_sharded_memory_config(
#     shape=x.shape,
#     core_grid=ttnn.CoreGrid(y=8, x=5),
#     strategy=ttnn.ShardStrategy.BLOCK,
#     orientation=ttnn.ShardOrientation.ROW_MAJOR,
#     use_height_and_width_as_shard_shape=False,
# )
# Sharded configurations can’t be used directly because they don’t include a Shard spec.
# @pytest.mark.parametrize(
#     "mem_config",
#     [
#         ttnn.DRAM_MEMORY_CONFIG,
#         ttnn.L1_MEMORY_CONFIG,
#         ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
#         ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
#         ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
#     ],
# )
# def test_ttt_where_mem_config(device, mem_config):
#     _ttt_where_test_impl(
#         device, (64, 64), tt_dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, mem_config=mem_config
#     )
