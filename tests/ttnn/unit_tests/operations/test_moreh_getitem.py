# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.utility_functions import comp_allclose_and_pcc, is_blackhole, skip_for_blackhole
from loguru import logger

from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import to_npu


def to_output_5d_shape(shape, index_dims, index_size):
    output_5d_shape = list(shape)
    for index_dim in index_dims:
        output_5d_shape[index_dim] = 1

    output_5d_shape[index_dims[-1]] = index_size

    return output_5d_shape


@skip_for_blackhole("Mismatching on Blackhole, see #12349")
@pytest.mark.parametrize(
    "shape_index_dim",
    [
        [[10, 70], 0],
        [[10, 5, 70], 1],
        [[10, 5, 7, 70], 2],
        [[10, 2, 5, 7, 70], 3],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.bfloat16,
    ],
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    [
        4,
        100,
    ],
)
def test_getitem_RAW_MJOR_one_index(shape_index_dim, dtype, index_size, device):
    shape, index_dim = shape_index_dim
    torch.manual_seed(2)

    if dtype == torch.int32:
        tt_dtype = ttnn.int32
    if dtype == torch.bfloat16:
        tt_dtype = ttnn.bfloat16

    x = torch.randint(low=0, high=10, size=shape).to(dtype)
    dev_x = ttnn.Tensor(x, tt_dtype).to(device)

    idx_value_max = shape[index_dim] - 1
    idx = torch.randint(-idx_value_max - 1, idx_value_max, (index_size,))
    dev_idx = ttnn.Tensor(idx, ttnn.int32).to(device)

    if index_dim == 0:
        tt_cpu = x[idx]
    elif index_dim == 1:
        tt_cpu = x[:, idx]
    elif index_dim == 2:
        tt_cpu = x[:, :, idx]
    elif index_dim == 3:
        tt_cpu = x[:, :, :, idx]
    elif index_dim == 4:
        tt_cpu = x[:, :, :, :, idx]

    tt_npu = ttnn.operations.moreh.getitem(dev_x, [dev_idx], [index_dim])

    assert list(tt_npu.shape.with_tile_padding()) == list(tt_cpu.shape)
    tt_dev = tt_npu.cpu().to_torch()

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@skip_for_blackhole("Mismatching on Blackhole, see #12349")
@pytest.mark.parametrize(
    "shape_index_dims",
    [
        [[10, 3, 5, 7, 80], [0, 1]],
        [[10, 3, 5, 7, 80], [1, 2]],
        [[10, 3, 5, 7, 80], [2, 3]],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.bfloat16,
    ],
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    [
        4,
        100,
    ],
)
def test_getitem_RAW_MAJOR_two_indices(shape_index_dims, dtype, index_size, device):
    shape, index_dims = shape_index_dims
    torch.manual_seed(1)

    if dtype == torch.int32:
        tt_dtype = ttnn.int32
    if dtype == torch.bfloat16:
        tt_dtype = ttnn.bfloat16

    x = torch.randint(low=0, high=10, size=shape).to(dtype)
    dev_x = ttnn.Tensor(x, tt_dtype).to(device)

    indices = []
    dev_indices = []
    for index_dim in index_dims:
        idx_value_max = shape[index_dim] - 1
        idx = torch.randint(-idx_value_max - 1, idx_value_max, (index_size,))
        dev_idx = ttnn.Tensor(idx, ttnn.int32).to(device)
        indices.append(idx)
        dev_indices.append(dev_idx)

    if index_dims == [0, 1]:
        tt_cpu = x[indices[0], indices[1]]
    if index_dims == [1, 2]:
        tt_cpu = x[:, indices[0], indices[1]]
    if index_dims == [2, 3]:
        tt_cpu = x[:, :, indices[0], indices[1]]
    tt_npu = ttnn.operations.moreh.getitem(dev_x, dev_indices, index_dims)

    assert list(tt_npu.shape.with_tile_padding()) == list(tt_cpu.shape)
    tt_dev = tt_npu.cpu().to_torch()

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@skip_for_blackhole("Mismatching on Blackhole, see #12349")
@pytest.mark.parametrize(
    "shape_index_dims",
    [
        [[10, 15, 7, 80], [0, 1, 2]],
        [[10, 3, 5, 7, 80], [0, 1, 2]],
        [[10, 3, 5, 7, 80], [1, 2, 3]],
    ],
)
@pytest.mark.parametrize(
    "index_size",
    [
        4,
        100,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.bfloat16,
    ],
    ids=["int32", "bfloat16"],
)
def test_getitem_RAW_MAJOR_three_indices(shape_index_dims, dtype, index_size, device):
    shape, index_dims = shape_index_dims
    torch.manual_seed(1)

    if dtype == torch.int32:
        tt_dtype = ttnn.int32
    if dtype == torch.bfloat16:
        tt_dtype = ttnn.bfloat16

    x = torch.randint(low=0, high=10, size=shape).to(dtype)
    dev_x = ttnn.Tensor(x, tt_dtype).to(device)

    indices = []
    dev_indices = []
    for index_dim in index_dims:
        idx_value_max = shape[index_dim] - 1
        idx = torch.randint(-idx_value_max - 1, idx_value_max, (index_size,))
        dev_idx = ttnn.Tensor(idx, ttnn.int32).to(device)
        indices.append(idx)
        dev_indices.append(dev_idx)

    if index_dims == [0, 1, 2]:
        tt_cpu = x[indices[0], indices[1], indices[2]]
    if index_dims == [1, 2, 3]:
        tt_cpu = x[:, indices[0], indices[1], indices[2]]
    tt_npu = ttnn.operations.moreh.getitem(dev_x, dev_indices, index_dims)

    assert list(tt_npu.shape.with_tile_padding()) == list(tt_cpu.shape)
    tt_dev = tt_npu.cpu().to_torch()

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


def run_getitem_RAW_MAJOR(shape_index_dim, dtype, index_size, device):
    shape, index_dim = shape_index_dim
    torch.manual_seed(2)

    if dtype == torch.int32:
        tt_dtype = ttnn.int32
    if dtype == torch.bfloat16:
        tt_dtype = ttnn.bfloat16

    x = torch.randint(low=0, high=10, size=shape).to(dtype)
    dev_x = ttnn.Tensor(x, tt_dtype).to(device)

    idx_value_max = shape[index_dim] - 1
    idx = torch.randint(-idx_value_max - 1, idx_value_max, (index_size,))
    dev_idx = ttnn.Tensor(idx, ttnn.int32).to(device)

    if index_dim == 0:
        tt_cpu = x[idx]
    elif index_dim == 1:
        tt_cpu = x[:, idx]
    elif index_dim == 2:
        tt_cpu = x[:, :, idx]
    elif index_dim == 3:
        tt_cpu = x[:, :, :, idx]
    elif index_dim == 4:
        tt_cpu = x[:, :, :, :, idx]

    tt_npu = ttnn.operations.moreh.getitem(dev_x, [dev_idx], [index_dim])

    assert list(tt_npu.shape.with_tile_padding()) == list(tt_cpu.shape)
    tt_dev = tt_npu.cpu().to_torch()

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@skip_for_blackhole("Mismatching on Blackhole, see #12349")
@pytest.mark.parametrize(
    "shape_index_dim",
    [
        [[10, 70], 0],
        [[10, 5, 70], 1],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.bfloat16,
    ],
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    [
        4,
        100,
    ],
)
def test_getitem_RAW_MAJOR_callback(shape_index_dim, dtype, index_size, device, use_program_cache):
    torch.manual_seed(2024)

    for _ in range(2):
        run_getitem_RAW_MAJOR(shape_index_dim, dtype, index_size, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_npu(torch_dummy, device)


@skip_for_blackhole("Mismatching on Blackhole, see #12349")
@pytest.mark.parametrize(
    "shape_index_dim",
    [
        [[7, 70], 0],
        [[5, 64], 1],
        [[10, 7, 70], 0],
        [[10, 5, 64], 1],
        [[10, 15, 70], 2],
        [[10, 5, 7, 70], 0],
        [[10, 5, 5, 64], 1],
        [[10, 5, 15, 70], 2],
        [[10, 5, 15, 70], 3],
        [[10, 3, 5, 7, 70], 0],
        [[1, 10, 3, 5, 64], 1],
        [[1, 1, 10, 15, 70], 2],
        [[1, 1, 10, 15, 70], 3],
        [[1, 1, 1, 2, 6], 4],
        [[1, 1, 1, 1, 30], 4],
        [[1, 5, 7, 3, 80], 4],
    ],
    ids=[
        "2d_W",
        "2d_H",
        "3d_D",
        "3d_H",
        "3d_W",
        "4d_C",
        "4d_D",
        "4d_H",
        "4d_W",
        "5d_N",
        "5d_C",
        "5d_D",
        "5d_H",
        "5d_W1",
        "5d_W2",
        "5d_W_LARGE",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.bfloat16,
    ],
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    [
        4,
        100,
    ],
)
@pytest.mark.parametrize(
    "row_major_index",
    [
        True,
        False,
    ],
)
def test_getitem_tilized_one_index(shape_index_dim, dtype, index_size, row_major_index, device):
    shape, index_dim = shape_index_dim
    torch.manual_seed(2)

    if dtype == torch.int32:
        tt_dtype = ttnn.int32
    if dtype == torch.bfloat16:
        tt_dtype = ttnn.bfloat16

    x = torch.randint(low=0, high=10, size=shape).to(dtype)

    dev_x = ttnn.Tensor(x, tt_dtype).reshape(shape).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    idx_value_max = shape[index_dim] - 1
    idx = torch.randint(-idx_value_max - 1, idx_value_max, (index_size,))
    if row_major_index:
        dev_idx = ttnn.Tensor(idx, ttnn.int32).to(device)
    else:
        dev_idx = (
            ttnn.Tensor(idx, ttnn.int32)
            .reshape([1, index_size])
            .pad_to_tile(float("nan"))
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )

    if index_dim == 0:
        tt_cpu = x[idx]
    elif index_dim == 1:
        tt_cpu = x[:, idx]
    elif index_dim == 2:
        tt_cpu = x[:, :, idx]
    elif index_dim == 3:
        tt_cpu = x[:, :, :, idx]
    elif index_dim == 4:
        tt_cpu = x[:, :, :, :, idx]

    tt_npu = ttnn.operations.moreh.getitem(dev_x, [dev_idx], [index_dim])
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    cpu_5d_shape = to_output_5d_shape(shape, [index_dim], index_size)

    tt_npu = tt_npu.unpad_from_tile(cpu_5d_shape)
    tt_dev = tt_npu.to_torch().reshape(tt_cpu.shape).to(dtype)

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@skip_for_blackhole("Mismatching on Blackhole, see #12349")
@pytest.mark.parametrize(
    "shape_index_dims",
    [
        [[7, 70], [0, 1]],
        [[10, 7, 70], [0, 1]],
        [[10, 5, 64], [1, 2]],
        [[10, 5, 7, 70], [0, 1]],
        [[10, 5, 5, 64], [1, 2]],
        [[10, 5, 15, 70], [2, 3]],
        [[10, 3, 5, 7, 70], [0, 1]],
        [[10, 3, 5, 7, 70], [1, 2]],
        [[10, 3, 5, 7, 70], [2, 3]],
        [[10, 3, 5, 7, 70], [3, 4]],
    ],
    ids=["2d_HW", "3d_DH", "3d_HW", "4d_CD", "4d_DH", "4d_HW", "5d_NC", "5d_CD", "5d_DH", "5d_HW"],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.bfloat16,
    ],
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    [
        4,
        100,
    ],
)
@pytest.mark.parametrize(
    "row_major_index",
    [
        True,
        False,
    ],
)
def test_getitem_tilized_two_indices(shape_index_dims, dtype, index_size, row_major_index, device):
    shape, index_dims = shape_index_dims
    torch.manual_seed(2)

    if dtype == torch.int32:
        tt_dtype = ttnn.int32
    if dtype == torch.bfloat16:
        tt_dtype = ttnn.bfloat16

    x = torch.randint(low=0, high=10, size=shape).to(dtype)

    dev_x = ttnn.Tensor(x, tt_dtype).reshape(shape).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    indices = []
    dev_indices = []
    for index_dim in index_dims:
        idx_value_max = shape[index_dim] - 1
        idx = torch.randint(-idx_value_max - 1, idx_value_max, (index_size,))
        if row_major_index:
            dev_idx = ttnn.Tensor(idx, ttnn.int32).to(device)
        else:
            dev_idx = (
                ttnn.Tensor(idx, ttnn.int32)
                .reshape([1, index_size])
                .pad_to_tile(float("nan"))
                .to(ttnn.TILE_LAYOUT)
                .to(device)
            )
        indices.append(idx)
        dev_indices.append(dev_idx)

    if index_dims == [0, 1]:
        tt_cpu = x[indices[0], indices[1]]
    if index_dims == [1, 2]:
        tt_cpu = x[:, indices[0], indices[1]]
    if index_dims == [2, 3]:
        tt_cpu = x[:, :, indices[0], indices[1]]
    if index_dims == [3, 4]:
        tt_cpu = x[:, :, :, indices[0], indices[1]]

    tt_npu = ttnn.operations.moreh.getitem(dev_x, dev_indices, index_dims)
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    output_5d_shape = to_output_5d_shape(shape, index_dims, index_size)

    tt_npu = tt_npu.unpad_from_tile(output_5d_shape)
    tt_dev = tt_npu.to_torch().reshape(tt_cpu.shape).to(dtype)

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@skip_for_blackhole("Mismatching on Blackhole, see #12349")
@pytest.mark.parametrize(
    "shape_index_dims",
    [
        [[5, 7, 70], [0, 1, 2]],
        [[3, 5, 7, 70], [0, 1, 2]],
        [[3, 5, 7, 70], [1, 2, 3]],
        [[10, 3, 5, 7, 70], [0, 1, 2]],
        [[10, 3, 5, 7, 70], [1, 2, 3]],
        [[10, 3, 5, 7, 70], [2, 3, 4]],
    ],
    ids=["3d_DHW", "4d_CDH", "4d_DHW", "5d_NCD", "5d_CDH", "5d_DHW"],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.bfloat16,
    ],
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    [
        4,
        100,
    ],
)
@pytest.mark.parametrize(
    "row_major_index",
    [
        True,
        False,
    ],
)
def test_getitem_tilized_three_indices(shape_index_dims, dtype, index_size, row_major_index, device):
    shape, index_dims = shape_index_dims
    torch.manual_seed(2)

    if dtype == torch.int32:
        tt_dtype = ttnn.int32
    if dtype == torch.bfloat16:
        tt_dtype = ttnn.bfloat16

    x = torch.randint(low=0, high=10, size=shape).to(dtype)

    dev_x = ttnn.Tensor(x, tt_dtype).reshape(shape).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    indices = []
    dev_indices = []
    for index_dim in index_dims:
        idx_value_max = shape[index_dim] - 1
        idx = torch.randint(-idx_value_max - 1, idx_value_max, (index_size,))
        if row_major_index:
            dev_idx = ttnn.Tensor(idx, ttnn.int32).to(device)
        else:
            dev_idx = (
                ttnn.Tensor(idx, ttnn.int32)
                .reshape([1, index_size])
                .pad_to_tile(float("nan"))
                .to(ttnn.TILE_LAYOUT)
                .to(device)
            )
        indices.append(idx)
        dev_indices.append(dev_idx)

    if index_dims == [0, 1, 2]:
        tt_cpu = x[indices[0], indices[1], indices[2]]
    if index_dims == [1, 2, 3]:
        tt_cpu = x[:, indices[0], indices[1], indices[2]]
    if index_dims == [2, 3, 4]:
        tt_cpu = x[:, :, indices[0], indices[1], indices[2]]

    tt_npu = ttnn.operations.moreh.getitem(dev_x, dev_indices, index_dims)
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    output_5d_shape = to_output_5d_shape(shape, index_dims, index_size)

    tt_npu = tt_npu.unpad_from_tile(output_5d_shape)
    tt_dev = tt_npu.to_torch().reshape(tt_cpu.shape).to(dtype)

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@skip_for_blackhole("Mismatching on Blackhole, see #12349")
@pytest.mark.parametrize(
    "shape_index_dims",
    [
        [[3, 5, 7, 80], [0, 1, 2, 3]],
        [[10, 3, 5, 7, 80], [0, 1, 2, 3]],
        [[10, 3, 5, 7, 80], [1, 2, 3, 4]],
    ],
    ids=["4d_CDHW", "5d_NCDH", "5d_CDHW"],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.bfloat16,
    ],
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    [
        4,
        100,
    ],
)
@pytest.mark.parametrize(
    "row_major_index",
    [
        True,
        False,
    ],
)
def test_getitem_tilized_four_indices(shape_index_dims, dtype, index_size, row_major_index, device):
    shape, index_dims = shape_index_dims
    torch.manual_seed(2)

    if dtype == torch.int32:
        tt_dtype = ttnn.int32
    if dtype == torch.bfloat16:
        tt_dtype = ttnn.bfloat16

    x = torch.randint(low=0, high=10, size=shape).to(dtype)

    dev_x = ttnn.Tensor(x, tt_dtype).reshape(shape).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    indices = []
    dev_indices = []
    for index_dim in index_dims:
        idx_value_max = shape[index_dim] - 1
        idx = torch.randint(-idx_value_max - 1, idx_value_max, (index_size,))
        if row_major_index:
            dev_idx = ttnn.Tensor(idx, ttnn.int32).to(device)
        else:
            dev_idx = (
                ttnn.Tensor(idx, ttnn.int32)
                .reshape([1, index_size])
                .pad_to_tile(float("nan"))
                .to(ttnn.TILE_LAYOUT)
                .to(device)
            )
        indices.append(idx)
        dev_indices.append(dev_idx)

    if index_dims == [0, 1, 2, 3]:
        tt_cpu = x[indices[0], indices[1], indices[2], indices[3]]
    if index_dims == [1, 2, 3, 4]:
        tt_cpu = x[:, indices[0], indices[1], indices[2], indices[3]]

    tt_npu = ttnn.operations.moreh.getitem(dev_x, dev_indices, index_dims)
    tt_npu = tt_npu.cpu().to(ttnn.Layout.ROW_MAJOR)

    output_5d_shape = to_output_5d_shape(shape, index_dims, index_size)

    tt_npu = tt_npu.unpad_from_tile(output_5d_shape)

    tt_dev = tt_npu.to_torch().reshape(tt_cpu.shape).to(dtype)

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@skip_for_blackhole("Mismatching on Blackhole, see #12349")
@pytest.mark.parametrize(
    "shape_index_dims",
    [
        [[10, 3, 5, 7, 80], [0, 1, 2, 3, 4]],
    ],
    ids=["NCDHW"],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.bfloat16,
    ],
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    [
        4,
        100,
    ],
)
@pytest.mark.parametrize(
    "row_major_index",
    [
        True,
        False,
    ],
)
def test_getitem_tilized_five_indices(shape_index_dims, dtype, index_size, row_major_index, device):
    shape, index_dims = shape_index_dims
    torch.manual_seed(2)

    if dtype == torch.int32:
        tt_dtype = ttnn.int32
    if dtype == torch.bfloat16:
        tt_dtype = ttnn.bfloat16

    x = torch.randint(low=0, high=10, size=shape).to(dtype)

    dev_x = ttnn.Tensor(x, tt_dtype).reshape(shape).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    indices = []
    dev_indices = []
    for index_dim in index_dims:
        idx_value_max = shape[index_dim] - 1
        idx = torch.randint(-idx_value_max - 1, idx_value_max, (index_size,))
        if row_major_index:
            dev_idx = ttnn.Tensor(idx, ttnn.int32).to(device)
        else:
            dev_idx = (
                ttnn.Tensor(idx, ttnn.int32)
                .reshape([1, index_size])
                .pad_to_tile(float("nan"))
                .to(ttnn.TILE_LAYOUT)
                .to(device)
            )
        indices.append(idx)
        dev_indices.append(dev_idx)

    tt_cpu = x[indices[0], indices[1], indices[2], indices[3], indices[4]]

    tt_npu = ttnn.operations.moreh.getitem(dev_x, dev_indices, index_dims)
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    output_5d_shape = to_output_5d_shape(shape, index_dims, index_size)

    tt_npu = tt_npu.unpad_from_tile(output_5d_shape)

    tt_dev = tt_npu.to_torch().reshape(tt_cpu.shape).to(dtype)

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


def run_moreh_geitem_tilized_one_index(shape_index_dim, dtype, index_size, row_major_index, device):
    shape, index_dim = shape_index_dim
    torch.manual_seed(2)

    if dtype == torch.int32:
        tt_dtype = ttnn.int32
    if dtype == torch.bfloat16:
        tt_dtype = ttnn.bfloat16

    x = torch.randint(low=0, high=10, size=shape).to(dtype)

    dev_x = ttnn.Tensor(x, tt_dtype).reshape(shape).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    idx_value_max = shape[index_dim] - 1
    idx = torch.randint(-idx_value_max - 1, idx_value_max, (index_size,))
    if row_major_index:
        dev_idx = ttnn.Tensor(idx, ttnn.int32).to(device)
    else:
        dev_idx = (
            ttnn.Tensor(idx, ttnn.int32)
            .reshape([1, index_size])
            .pad_to_tile(float("nan"))
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )

    if index_dim == 0:
        tt_cpu = x[idx]
    elif index_dim == 1:
        tt_cpu = x[:, idx]
    elif index_dim == 2:
        tt_cpu = x[:, :, idx]
    elif index_dim == 3:
        tt_cpu = x[:, :, :, idx]
    elif index_dim == 4:
        tt_cpu = x[:, :, :, :, idx]

    tt_npu = ttnn.operations.moreh.getitem(dev_x, [dev_idx], [index_dim])
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    cpu_5d_shape = to_output_5d_shape(shape, [index_dim], index_size)

    tt_npu = tt_npu.unpad_from_tile(cpu_5d_shape)
    tt_dev = tt_npu.to_torch().reshape(tt_cpu.shape).to(dtype)

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@skip_for_blackhole("Mismatching on Blackhole, see #12349")
@pytest.mark.parametrize(
    "shape_index_dim",
    [
        [[7, 70], 0],
    ],
    ids=[
        "2d_W",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.bfloat16,
    ],
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    [
        4,
    ],
)
@pytest.mark.parametrize(
    "row_major_index",
    [
        False,
    ],
)
def test_getitem_tilized_one_index_callback(
    shape_index_dim, dtype, index_size, row_major_index, device, use_program_cache
):
    torch.manual_seed(2024)
    for _ in range(2):
        run_moreh_geitem_tilized_one_index(shape_index_dim, dtype, index_size, row_major_index, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_npu(torch_dummy, device)
