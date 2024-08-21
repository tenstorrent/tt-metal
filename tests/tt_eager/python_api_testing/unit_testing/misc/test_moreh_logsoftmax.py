# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger
import torch.nn.functional as F
from models.utility_functions import is_wormhole_b0

from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((50, 32), 1),  # single tile
        ((3, 32, 32 * 5), 2),  # mutiple tile with dim W
        ((5, 6, 32, 32), 3),  # multiple cores
        ((10, 20, 32 * 3, 32 * 5), 3),  # multiple tiles per core
        ((32, 32), 0),  # single tile
        ((3, 32 * 5, 32), 1),  # mutiple tile with dim H
        ((5, 6, 32, 32), 2),  # multiple cores
        ((10, 20, 32 * 3, 32 * 5), 2),  # multiple tiles per core
    ),
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_for_dim_hw(shape_dim, compute_kernel_options, device):
    device.enable_program_cache()

    shape, dim = shape_dim
    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    x = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16) + 100

    dev_x = ttnn.Tensor(x, ttnn.bfloat16).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    tt_cpu = F.log_softmax(x, dim)
    tt_npu = ttnn.experimental.operations.primary.moreh_logsoftmax(
        dev_x, dim, compute_kernel_config=compute_kernel_config
    )

    tt_dev = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(shape)
    assert list(tt_dev.get_legacy_shape()) == list(tt_cpu.shape)
    tt_dev = tt_dev.to_torch().to(torch.bfloat16)

    rtol = atol = 0.1
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.debug(out)
    assert passing


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((2, 3, 32 * 4, 32 * 5), 3),
        ((2, 3, 32 * 4, 32 * 5), 2),
    ),
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_large_algorithm_for_dim_hw(shape_dim, compute_kernel_options, device):
    device.enable_program_cache()

    shape, dim = shape_dim
    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    x = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16) + 100

    dev_x = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    tt_cpu = F.log_softmax(x, dim)
    strategy = (
        ttnn.experimental.operations.primary.MorehSoftmaxOpParallelizationStrategy.LARGE_W
        if dim == 3
        else ttnn.experimental.operations.primary.MorehSoftmaxOpParallelizationStrategy.LARGE_H
    )
    tt_npu = ttnn.experimental.operations.primary.moreh_logsoftmax(
        dev_x, dim, None, strategy, compute_kernel_config=compute_kernel_config
    )

    assert list(tt_npu.get_legacy_shape()) == list(tt_cpu.shape)
    tt_dev = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)

    rtol = atol = 0.1
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.debug(out)
    assert passing


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((1, 1, 10, 15), 3),  # single tile
        ((1, 1, 10, 32 * 2 + 10), 3),  # mutiple tile with dim
        ((1, 1, 15, 10), 2),  # single tile
        ((1, 1, 32 * 2 + 10, 32), 2),  # mutiple tile with dim
    ),
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_not_multiple_of_32_for_dim_hw(shape_dim, compute_kernel_options, device):
    device.enable_program_cache()
    shape, dim = shape_dim
    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    x = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16)

    dev_x = ttnn.Tensor(x, ttnn.bfloat16).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    tt_cpu = F.log_softmax(x, dim)
    tt_npu = ttnn.experimental.operations.primary.moreh_logsoftmax(
        dev_x, dim, compute_kernel_config=compute_kernel_config
    )
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(shape)

    assert list(tt_npu.get_legacy_shape()) == list(tt_cpu.shape)
    tt_dev = tt_npu.to_torch().to(torch.bfloat16)

    rtol = atol = 0.1
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.debug(out)
    assert passing


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((1, 15, 32, 32), 1),  # single tile c
        ((1, 15, 32 * 7, 32 * 5), 1),  # mutiple cores
        ((109, 15, 32, 32), 1),  # mutiple tiles per cores
        ((15, 1, 32, 32), 0),  # single tile n
        ((15, 1, 32 * 7, 32 * 5), 0),  # mutiple cores
        ((15, 109, 32 * 2, 32 * 2), 0),  # mutiple tiles per cores
    ),
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_for_dim_nc(shape_dim, compute_kernel_options, device):
    device.enable_program_cache()
    shape, dim = shape_dim
    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    x = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16) + 100

    dev_x = ttnn.Tensor(x, ttnn.bfloat16).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    tt_cpu = F.log_softmax(x, dim)
    tt_npu = ttnn.experimental.operations.primary.moreh_logsoftmax(
        dev_x, dim, compute_kernel_config=compute_kernel_config
    )
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(shape)

    assert list(tt_npu.get_legacy_shape()) == list(tt_cpu.shape)
    tt_dev = tt_npu.to_torch().to(torch.bfloat16)

    rtol = atol = 0.1
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.debug(out)
    assert passing


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((32, 32), 1),  # single tile
        ((3, 32, 32 * 2), 2),  # mutiple tile with dim W
        ((5, 6, 32, 32), 3),  # multiple cores
        ((10, 20, 32 * 3, 32 * 5), 3),  # multiple tiles per core
        ((32, 32), 0),  # single tile
        ((3, 32 * 5, 32), 1),  # mutiple tile with dim H
        ((5, 6, 32, 32), 2),  # multiple cores
        ((10, 20, 32 * 5, 32), 2),  # multiple tiles per core
    ),
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_backward_for_dim_hw(shape_dim, compute_kernel_options, device):
    device.enable_program_cache()
    shape, dim = shape_dim
    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    x = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16).requires_grad_(True)

    y = F.log_softmax(x, dim)
    dev_y = ttnn.Tensor(y, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    dy = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16)
    dev_dy = ttnn.Tensor(dy, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    y.backward(dy)
    tt_npu = ttnn.experimental.operations.primary.moreh_logsoftmax_backward(
        dev_y, dev_dy, dim, compute_kernel_config=compute_kernel_config
    )

    assert list(tt_npu.get_legacy_shape()) == list(x.grad.shape)
    tt_dev = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)

    rtol = atol = 0.5
    passing, out = comp_allclose_and_pcc(x.grad, tt_dev, rtol=rtol, atol=atol)
    logger.debug(out)
    assert passing


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((2, 3, 32 * 4, 32 * 5), 3),
        ((2, 3, 32 * 4, 32 * 5), 2),
    ),
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_backward_large_algorithm_for_dim_hw(shape_dim, compute_kernel_options, device):
    device.enable_program_cache()
    shape, dim = shape_dim
    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    x = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16).requires_grad_(True)

    y = F.log_softmax(x, dim)
    dev_y = ttnn.Tensor(y, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    dy = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16)
    dev_dy = ttnn.Tensor(dy, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    y.backward(dy)
    strategy = (
        ttnn.experimental.operations.primary.MorehSoftmaxBackwardOpParallelizationStrategy.LARGE_W
        if dim == 3
        else ttnn.experimental.operations.primary.MorehSoftmaxBackwardOpParallelizationStrategy.LARGE_H
    )
    tt_npu = ttnn.experimental.operations.primary.moreh_logsoftmax_backward(
        dev_y, dev_dy, dim, None, strategy, compute_kernel_config=compute_kernel_config
    )

    assert list(tt_npu.get_legacy_shape()) == list(x.grad.shape)
    tt_dev = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)

    rtol = atol = 0.5
    passing, out = comp_allclose_and_pcc(x.grad, tt_dev, rtol=rtol, atol=atol)
    logger.debug(out)
    assert passing


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((1, 1, 10, 15), 3),  # single tile
        ((1, 1, 10, 32 * 2 + 10), 3),  # mutiple tile with dim
        ((1, 1, 15, 10), 2),  # single tile
        ((1, 1, 32 * 2 + 10, 32), 2),  # mutiple tile with dim
    ),
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_backward_not_multiple_of_32_for_dim_hw(shape_dim, compute_kernel_options, device):
    device.enable_program_cache()
    shape, dim = shape_dim
    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    x = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16).requires_grad_(True)

    y = F.log_softmax(x, dim)
    dev_y = ttnn.Tensor(y, ttnn.bfloat16).pad_to_tile(float("10")).to(ttnn.TILE_LAYOUT).to(device)

    dy = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16)
    dev_dy = ttnn.Tensor(dy, ttnn.bfloat16).pad_to_tile(float("200")).to(ttnn.TILE_LAYOUT).to(device)

    y.backward(dy)
    tt_npu = ttnn.experimental.operations.primary.moreh_logsoftmax_backward(
        dev_y, dev_dy, dim, compute_kernel_config=compute_kernel_config
    )
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(shape)

    assert list(tt_npu.get_legacy_shape()) == list(x.grad.shape)
    tt_dev = tt_npu.to_torch().to(torch.bfloat16)

    rtol = atol = 0.1
    passing, out = comp_allclose_and_pcc(x.grad, tt_dev, rtol=rtol, atol=atol)
    logger.debug(out)
    assert passing


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((1, 15, 32, 32), 1),  # single tile c
        ((1, 15, 32 * 7, 32 * 5), 1),  # mutiple cores
        ((109, 15, 32, 32), 1),  # mutiple tiles per cores
        ((15, 1, 32, 32), 0),  # single tile n
        ((15, 1, 32 * 7, 32 * 5), 0),  # mutiple cores
        ((15, 109, 32 * 2, 32 * 2), 0),  # mutiple tiles per cores
    ),
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_logsoftmax_backward_for_dim_nc(shape_dim, compute_kernel_options, device):
    device.enable_program_cache()
    shape, dim = shape_dim
    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    x = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16).requires_grad_(True)

    y = F.log_softmax(x, dim)
    dev_y = ttnn.Tensor(y, ttnn.bfloat16).pad_to_tile(float("10")).to(ttnn.TILE_LAYOUT).to(device)

    dy = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16)
    dev_dy = ttnn.Tensor(dy, ttnn.bfloat16).pad_to_tile(float("10")).to(ttnn.TILE_LAYOUT).to(device)

    y.backward(dy)
    tt_npu = ttnn.experimental.operations.primary.moreh_logsoftmax_backward(
        dev_y, dev_dy, dim, compute_kernel_config=compute_kernel_config
    )
    tt_npu = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(shape)
    assert list(tt_npu.get_legacy_shape()) == list(x.grad.shape)
    tt_dev = tt_npu.cpu().to_torch().to(torch.bfloat16)

    rtol = atol = 0.5
    passing, out = comp_allclose_and_pcc(x.grad, tt_dev, rtol=rtol, atol=atol)
    logger.debug(out)
    assert passing


@pytest.mark.parametrize(
    "shape_dim",
    (((32, 32), 1),),  # single tile
)
@pytest.mark.parametrize(
    "optional_output_tensor",
    (True, False),
)
def test_logsoftmax_optional_output_tensor(shape_dim, optional_output_tensor, device):
    device.enable_program_cache()

    shape, dim = shape_dim
    torch.manual_seed(0)

    x = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16)

    # cpu calculation
    tt_cpu = F.log_softmax(x, dim)

    # npu calculation
    dev_x = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    if optional_output_tensor:
        dev_y = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

        tt_npu = ttnn.experimental.operations.primary.moreh_logsoftmax(dev_x, dim, dev_y)
    else:
        tt_npu = ttnn.experimental.operations.primary.moreh_logsoftmax(dev_x, dim)

    assert list(tt_npu.get_legacy_shape()) == list(tt_cpu.shape)
    tt_dev = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.info(out)
    assert passing


@pytest.mark.parametrize(
    "shape_dim",
    (((32, 32), 1),),  # single tile
)
@pytest.mark.parametrize(
    "optional_output_tensor",
    (True, False),
)
def test_logsoftmax_backward_optional_output_tensor(shape_dim, optional_output_tensor, device):
    device.enable_program_cache()
    shape, dim = shape_dim
    torch.manual_seed(0)

    # cpu calculation
    x = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16).requires_grad_(True)

    y = F.log_softmax(x, dim)
    dy = torch.randint(low=0, high=4, size=shape).to(torch.bfloat16)
    y.backward(dy)

    # npu calculation
    dev_y = ttnn.Tensor(y, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    dev_dy = ttnn.Tensor(dy, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    if optional_output_tensor:
        dev_dx = ttnn.Tensor(dy, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
        tt_npu = ttnn.experimental.operations.primary.moreh_logsoftmax_backward(dev_y, dev_dy, dim, dev_dx)
    else:
        tt_npu = ttnn.experimental.operations.primary.moreh_logsoftmax_backward(dev_y, dev_dy, dim)

    assert list(tt_npu.get_legacy_shape()) == list(x.grad.shape)
    tt_dev = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(x.grad, tt_dev, rtol=rtol, atol=atol)
    logger.info(out)
    assert passing
