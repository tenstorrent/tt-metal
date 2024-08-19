# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger


def to_output_4d_shape(shape, index_dims, index_size):
    output_4d_shape = list(shape)
    for index_dim in index_dims:
        output_4d_shape[index_dim] = 1

    output_4d_shape[index_dims[-1]] = index_size

    return output_4d_shape


@pytest.mark.parametrize(
    "shape_index_dim",
    (
        ((10, 70), 0),
        ((10, 5, 70), 1),
        ((10, 5, 7, 70), 2),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    (
        torch.int32,
        torch.bfloat16,
    ),
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    (
        4,
        100,
    ),
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

    tt_npu = ttnn.experimental.operations.primary.moreh_getitem(dev_x, [dev_idx], [index_dim])

    assert list(tt_npu.get_legacy_shape()) == list(tt_cpu.shape)
    tt_dev = tt_npu.cpu().to_torch()

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@pytest.mark.parametrize(
    "shape_index_dims",
    (
        ((10, 15, 7, 80), (0, 1)),
        ((10, 15, 7, 80), (1, 2)),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    (
        torch.int32,
        torch.bfloat16,
    ),
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    (
        4,
        100,
    ),
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

    if index_dims == (0, 1):
        tt_cpu = x[indices[0], indices[1]]
    if index_dims == (1, 2):
        tt_cpu = x[:, indices[0], indices[1]]
    tt_npu = ttnn.experimental.operations.primary.moreh_getitem(dev_x, dev_indices, index_dims)

    assert list(tt_npu.get_legacy_shape()) == list(tt_cpu.shape)
    tt_dev = tt_npu.cpu().to_torch()

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@pytest.mark.parametrize(
    "shape_index_dims",
    (((10, 15, 7, 80), (0, 1, 2)),),
)
@pytest.mark.parametrize(
    "index_size",
    (
        4,
        100,
    ),
)
@pytest.mark.parametrize(
    "dtype",
    (
        torch.int32,
        torch.bfloat16,
    ),
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

    if index_dims == (0, 1, 2):
        tt_cpu = x[indices[0], indices[1], indices[2]]
    tt_npu = ttnn.experimental.operations.primary.moreh_getitem(dev_x, dev_indices, index_dims)

    assert list(tt_npu.get_legacy_shape()) == list(tt_cpu.shape)
    tt_dev = tt_npu.cpu().to_torch()

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@pytest.mark.parametrize(
    "shape_index_dim",
    (
        ((10, 5, 7, 70), 0),
        ((1, 10, 5, 64), 1),
        ((1, 1, 10, 70), 2),
        ((1, 1, 2, 6), 3),
        ((1, 1, 1, 30), 3),
        ((5, 7, 3, 80), 3),
    ),
    ids=[
        "N",
        "C",
        "H",
        "W1",
        "W2",
        "W_LARGE",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    (
        torch.int32,
        torch.bfloat16,
    ),
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    (
        4,
        100,
    ),
)
@pytest.mark.parametrize(
    "row_major_index",
    (
        True,
        False,
    ),
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
            .reshape(1, 1, 1, index_size)
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

    tt_npu = ttnn.experimental.operations.primary.moreh_getitem(dev_x, [dev_idx], [index_dim])
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    cpu_4d_shape = to_output_4d_shape(shape, [index_dim], index_size)

    tt_npu = tt_npu.unpad_from_tile(cpu_4d_shape)
    tt_dev = tt_npu.to_torch().reshape(tt_cpu.shape).to(dtype)

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@pytest.mark.parametrize(
    "shape_index_dims",
    (
        ((10, 15, 7, 70), (0, 1)),
        ((10, 15, 7, 70), (1, 2)),
        ((10, 15, 7, 70), (2, 3)),
    ),
    ids=["NC", "CH", "HW"],
)
@pytest.mark.parametrize(
    "dtype",
    (
        torch.int32,
        torch.bfloat16,
    ),
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    (
        4,
        100,
    ),
)
@pytest.mark.parametrize(
    "row_major_index",
    (
        True,
        False,
    ),
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
                .reshape(1, 1, 1, index_size)
                .pad_to_tile(float("nan"))
                .to(ttnn.TILE_LAYOUT)
                .to(device)
            )
        indices.append(idx)
        dev_indices.append(dev_idx)

    if index_dims == (0, 1):
        tt_cpu = x[indices[0], indices[1]]
    if index_dims == (1, 2):
        tt_cpu = x[:, indices[0], indices[1]]
    if index_dims == (2, 3):
        tt_cpu = x[:, :, indices[0], indices[1]]

    tt_npu = ttnn.experimental.operations.primary.moreh_getitem(dev_x, dev_indices, index_dims)
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    output_4d_shape = to_output_4d_shape(shape, index_dims, index_size)

    tt_npu = tt_npu.unpad_from_tile(output_4d_shape)
    tt_dev = tt_npu.to_torch().reshape(tt_cpu.shape).to(dtype)

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@pytest.mark.parametrize(
    "shape_index_dims",
    (
        ((10, 15, 7, 70), (0, 1, 2)),
        ((10, 15, 7, 70), (1, 2, 3)),
    ),
    ids=["NC", "CH"],
)
@pytest.mark.parametrize(
    "dtype",
    (
        torch.int32,
        torch.bfloat16,
    ),
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    (
        4,
        100,
    ),
)
@pytest.mark.parametrize(
    "row_major_index",
    (
        True,
        False,
    ),
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
                .reshape(1, 1, 1, index_size)
                .pad_to_tile(float("nan"))
                .to(ttnn.TILE_LAYOUT)
                .to(device)
            )
        indices.append(idx)
        dev_indices.append(dev_idx)

    if index_dims == (0, 1, 2):
        tt_cpu = x[indices[0], indices[1], indices[2]]
    if index_dims == (1, 2, 3):
        tt_cpu = x[:, indices[0], indices[1], indices[2]]

    tt_npu = ttnn.experimental.operations.primary.moreh_getitem(dev_x, dev_indices, index_dims)
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    output_4d_shape = to_output_4d_shape(shape, index_dims, index_size)

    tt_npu = tt_npu.unpad_from_tile(output_4d_shape)
    tt_dev = tt_npu.to_torch().reshape(tt_cpu.shape).to(dtype)

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing


@pytest.mark.parametrize(
    "shape_index_dims",
    (((10, 15, 7, 80), (0, 1, 2, 3)),),
    ids=["NCHW"],
)
@pytest.mark.parametrize(
    "dtype",
    (
        torch.int32,
        torch.bfloat16,
    ),
    ids=["int32", "bfloat16"],
)
@pytest.mark.parametrize(
    "index_size",
    (
        4,
        100,
    ),
)
@pytest.mark.parametrize(
    "row_major_index",
    (
        True,
        False,
    ),
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
                .reshape(1, 1, 1, index_size)
                .pad_to_tile(float("nan"))
                .to(ttnn.TILE_LAYOUT)
                .to(device)
            )
        indices.append(idx)
        dev_indices.append(dev_idx)

    tt_cpu = x[indices[0], indices[1], indices[2], indices[3]]

    tt_npu = ttnn.experimental.operations.primary.moreh_getitem(dev_x, dev_indices, index_dims)
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    output_4d_shape = to_output_4d_shape(shape, index_dims, index_size)

    tt_npu = tt_npu.unpad_from_tile(output_4d_shape)

    tt_dev = tt_npu.to_torch().reshape(tt_cpu.shape).to(dtype)

    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev)
    logger.info(out)

    assert passing
