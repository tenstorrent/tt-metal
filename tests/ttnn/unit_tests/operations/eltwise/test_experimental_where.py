# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import is_watcher_enabled
from tests.ttnn.utils_for_testing import assert_with_pcc, tt_dtype_to_torch_dtype


DEFAULT_SHAPE = (64, 64)


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


def _ttt_where_test_impl(
    device, tensor_shape: tuple, tt_dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, mem_config=None
):
    torch.manual_seed(0)

    condition_torch = torch.rand(tensor_shape, dtype=torch.float) > 0.5

    torch_dtype = tt_dtype_to_torch_dtype[tt_dtype]
    true_torch, false_torch = [
        torch.rand(tensor_shape, dtype=torch_dtype)
        if is_ttnn_float_type(tt_dtype)
        else torch.randint(torch.iinfo(torch_dtype).min, torch.iinfo(torch_dtype).max, tensor_shape, dtype=torch_dtype)
        for _ in range(2)
    ]

    golden_fn = ttnn.get_golden_function(ttnn.where)
    torch_output_tensor = golden_fn(condition_torch, true_torch, false_torch)

    condition, true_values, false_values = [
        ttnn.from_torch(
            tensor,
            layout=layout,
            dtype=tt_dtype,
            memory_config=mem_config,
            device=device,
        )
        for tensor in (condition_torch, true_torch, false_torch)
    ]

    output_tensor = ttnn.experimental.where(condition, true_values, false_values)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


def test_ttt_where_0d(device):
    if is_watcher_enabled():
        pytest.skip("Skipping test_ttt_where_0d with watcher enabled, github issue #37048")
    _ttt_where_test_impl(device, ())


@pytest.mark.parametrize("h", [16, 32, 64, 65, 1024])
def test_ttt_where_1d(device, h):
    if is_watcher_enabled():
        pytest.skip("Skipping test_ttt_where_1d with watcher enabled, github issue #37048")
    _ttt_where_test_impl(device, (h))


@pytest.mark.parametrize("h", [0, 16, 32, 64, 65, 1024])
@pytest.mark.parametrize("w", [0, 16, 32, 64, 65, 1024])
def test_ttt_where_2d(device, h, w):
    if is_watcher_enabled():
        pytest.skip("Skipping test_ttt_where_2d with watcher enabled, github issue #37048")
    _ttt_where_test_impl(device, (h, w))


@pytest.mark.parametrize("d4", [16])
@pytest.mark.parametrize("d3", [16])
@pytest.mark.parametrize("h", [16])
@pytest.mark.parametrize("w", [16])
def test_ttt_where_4d(device, d4, d3, h, w):
    _ttt_where_test_impl(device, (d4, d3, h, w))


@pytest.mark.parametrize("d5", [16])
@pytest.mark.parametrize("d4", [16])
@pytest.mark.parametrize("d3", [16])
@pytest.mark.parametrize("h", [16])
@pytest.mark.parametrize("w", [16])
def test_ttt_where_5d(device, d5, d4, d3, h, w):
    _ttt_where_test_impl(device, (d5, d4, d3, h, w))


@pytest.mark.parametrize("shape", [tuple([32] * i) for i in range(6)])
def test_ttt_where_shapes(device, shape):
    if is_watcher_enabled():
        pytest.skip("Skipping test_ttt_where_shapes with watcher enabled, github issue #37048")
    _ttt_where_test_impl(device, shape)


@pytest.mark.xfail(reason="Integer data types are not yet supported.")
@pytest.mark.parametrize("tt_dtype", [ttnn.uint8, ttnn.uint16, ttnn.int32, ttnn.uint32])
def test_ttt_where_int_types(device, tt_dtype):
    if is_watcher_enabled():
        pytest.skip("Skipping test_ttt_where_float_types with watcher enabled, github issue #37048")
    _ttt_where_test_impl(device, DEFAULT_SHAPE, tt_dtype=tt_dtype)


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.bfloat8_b,
        pytest.param(
            ttnn.bfloat4_b,
            marks=pytest.mark.xfail(reason="ttnn.bfloat4_b data type is not yet supported."),
        ),
    ],
)
def test_ttt_where_float_types(device, tt_dtype):
    if is_watcher_enabled():
        pytest.skip("Skipping test_ttt_where_float_types with watcher enabled, github issue #37048")
    _ttt_where_test_impl(device, DEFAULT_SHAPE, tt_dtype=tt_dtype)


@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            marks=pytest.mark.xfail(reason="ROW_MAJOR_LAYOUT is not yet supported."),
        ),
    ],
)
def test_ttt_where_layouts(device, layout):
    if is_watcher_enabled():
        pytest.skip("Skipping test_ttt_where_float_types with watcher enabled, github issue #37048")
    # Missing parameters in from_torch() for layout test coverage: tile, pad_value
    _ttt_where_test_impl(device, DEFAULT_SHAPE, layout=layout)
