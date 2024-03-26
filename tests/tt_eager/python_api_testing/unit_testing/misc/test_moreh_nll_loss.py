# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
import pytest
from models.utility_functions import (
    comp_allclose_and_pcc,
)
from loguru import logger


def get_torch_tensors_4d(shape, device):
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    cpu_dtype = torch.float32
    cpu_index_dtype = torch.long

    torch_data = torch.randn([N, C, H, W], dtype=cpu_dtype)
    torch_input = torch.nn.functional.log_softmax(torch_data, dim=1).requires_grad_()
    torch_target = torch.randint(0, C, [N, H, W], dtype=cpu_index_dtype)
    torch_weight = torch.rand(C, dtype=cpu_dtype)
    torch_divisor = torch.tensor([0], dtype=cpu_dtype)
    torch_output = torch.tensor([0], dtype=cpu_dtype)

    return torch_input, torch_target, torch_weight, torch_divisor, torch_output


def get_tt_tensors_4d(torch_input, torch_target, torch_weight, torch_divisor, torch_output, device):
    torch.manual_seed(0)

    N = torch_input.shape[0]
    C = torch_input.shape[1]
    H = torch_input.shape[2]
    W = torch_input.shape[3]

    npu_dtype = ttl.tensor.DataType.BFLOAT16
    npu_index_dtype = ttl.tensor.DataType.UINT32
    npu_layout = ttl.tensor.Layout.TILE
    npu_weight_layout = ttl.tensor.Layout.ROW_MAJOR

    tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_target = (
        ttl.tensor.Tensor(torch_target, npu_index_dtype).reshape(N, 1, H, W).pad_to_tile(C).to(npu_layout).to(device)
    )
    tt_weight = ttl.tensor.Tensor(torch_weight, npu_dtype).to(npu_weight_layout).to(device)
    tt_divisor = (
        ttl.tensor.Tensor(torch_divisor, npu_dtype)
        .reshape(1, 1, 1, 1)
        .pad_to_tile(float("nan"))
        .to(npu_layout)
        .to(device)
    )
    tt_output = (
        ttl.tensor.Tensor(torch_output, npu_dtype)
        .reshape(1, 1, 1, 1)
        .pad_to_tile(float("nan"))
        .to(npu_layout)
        .to(device)
    )

    return tt_input, tt_target, tt_weight, tt_divisor, tt_output


def get_torch_tensors_2d(shape, device):
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]

    cpu_dtype = torch.bfloat16
    cpu_index_dtype = torch.long

    torch_data = torch.randint(-2, 3, [N, C], dtype=cpu_dtype)
    torch_input = torch.nn.functional.log_softmax(torch_data, dim=1).requires_grad_()
    torch_target = torch.randint(0, C, [N], dtype=cpu_index_dtype)
    torch_weight = torch.rand(C, dtype=cpu_dtype)
    torch_divisor = torch.tensor([0], dtype=cpu_dtype)
    torch_output = torch.tensor([0], dtype=cpu_dtype)

    return torch_input, torch_target, torch_weight, torch_divisor, torch_output


def get_tt_tensors_2d(torch_input, torch_target, torch_weight, torch_divisor, torch_output, device):
    torch.manual_seed(0)

    N = torch_input.shape[0]
    C = torch_input.shape[1]
    H = 1
    W = 1

    npu_dtype = ttl.tensor.DataType.BFLOAT16
    npu_index_dtype = ttl.tensor.DataType.UINT32
    npu_layout = ttl.tensor.Layout.TILE
    npu_weight_layout = ttl.tensor.Layout.ROW_MAJOR

    tt_input = (
        ttl.tensor.Tensor(torch_input, npu_dtype)
        .reshape(N, C, H, W)
        .pad_to_tile(float("nan"))
        .to(npu_layout)
        .to(device)
    )
    tt_target = (
        ttl.tensor.Tensor(torch_target, npu_index_dtype).reshape(N, 1, H, W).pad_to_tile(C).to(npu_layout).to(device)
    )
    tt_weight = ttl.tensor.Tensor(torch_weight, npu_dtype).to(npu_weight_layout).to(device)
    tt_divisor = (
        ttl.tensor.Tensor(torch_divisor, npu_dtype)
        .reshape(1, 1, 1, 1)
        .pad_to_tile(float("nan"))
        .to(npu_layout)
        .to(device)
    )
    tt_output = (
        ttl.tensor.Tensor(torch_output, npu_dtype)
        .reshape(1, 1, 1, 1)
        .pad_to_tile(float("nan"))
        .to(npu_layout)
        .to(device)
    )

    return tt_input, tt_target, tt_weight, tt_divisor, tt_output


@pytest.mark.parametrize(
    "shape",
    (
        [1, 2, 2, 2],
        [1, 2, 32, 32],
        [3, 4, 32 * 5, 32 * 6],
    ),
)
@pytest.mark.parametrize("ignore_index", [0, -1])
@pytest.mark.parametrize("reduction_mean", [True, False])
def test_moreh_nll_loss_4d(shape, ignore_index, reduction_mean, device):
    device.enable_program_cache()

    (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors_4d(shape, device)
    nll_loss = torch.nn.NLLLoss(
        weight=torch_weight, ignore_index=ignore_index, reduction="mean" if reduction_mean else "sum"
    )
    torch_loss = torch.tensor([nll_loss(torch_input, torch_target)])

    (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors_4d(
        torch_input, torch_target, torch_weight, torch_divisor, torch_output, device
    )
    tt_loss = ttl.operations.primary.moreh_nll_loss(
        tt_input, tt_target, tt_weight, tt_divisor, tt_output, ignore_index, reduction_mean
    )
    tt_loss_to_cpu = tt_loss.cpu().to(ttl.tensor.Layout.ROW_MAJOR).unpad_from_tile([1, 1, 1, 1]).to_torch().reshape([1])

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_loss, tt_loss_to_cpu, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing


@pytest.mark.parametrize("shape", ([1, 2], [3, 4], [12, 6]))
@pytest.mark.parametrize("ignore_index", [0, -1])
@pytest.mark.parametrize("reduction_mean", [True, False])
def test_moreh_nll_loss_2d(shape, ignore_index, reduction_mean, device):
    device.enable_program_cache()

    (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors_2d(shape, device)
    nll_loss = torch.nn.NLLLoss(
        weight=torch_weight, ignore_index=ignore_index, reduction="mean" if reduction_mean else "sum"
    )
    torch_loss = torch.tensor([nll_loss(torch_input, torch_target)])

    (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors_2d(
        torch_input, torch_target, torch_weight, torch_divisor, torch_output, device
    )
    tt_loss = ttl.operations.primary.moreh_nll_loss(
        tt_input, tt_target, tt_weight, tt_divisor, tt_output, ignore_index, reduction_mean
    )
    tt_loss_to_cpu = tt_loss.cpu().to(ttl.tensor.Layout.ROW_MAJOR).unpad_from_tile([1, 1, 1, 1]).to_torch().reshape([1])

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_loss, tt_loss_to_cpu, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing


@pytest.mark.parametrize(
    "shape",
    (
        [1, 2, 32, 32],
        [1, 2, 32, 32],
        [3, 4, 32 * 5, 32 * 6],
    ),
)
@pytest.mark.parametrize("ignore_index", [0, -1])
@pytest.mark.parametrize("reduction_mean", [True, False])
def test_moreh_nll_loss_4d_backward(shape, ignore_index, reduction_mean, device):
    device.enable_program_cache()

    (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors_4d(shape, device)
    nll_loss = torch.nn.NLLLoss(
        weight=torch_weight, ignore_index=ignore_index, reduction="mean" if reduction_mean else "sum"
    )
    torch_loss = nll_loss(torch_input, torch_target)

    (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors_4d(
        torch_input, torch_target, torch_weight, torch_divisor, torch_output, device
    )
    if reduction_mean == False:
        tt_divisor = None
    tt_loss = ttl.operations.primary.moreh_nll_loss(
        tt_input, tt_target, tt_weight, tt_divisor, tt_output, ignore_index, reduction_mean
    )
    tt_loss_to_cpu = tt_loss.cpu().to(ttl.tensor.Layout.ROW_MAJOR).unpad_from_tile([1, 1, 1, 1]).to_torch().reshape([1])

    output_grad = torch.randn_like(torch_loss)
    torch_loss.backward(output_grad)

    tt_output_grad = (
        ttl.tensor.Tensor(output_grad.reshape(1, 1, 1, 1), ttl.tensor.DataType.BFLOAT16)
        .pad_to_tile(float("nan"))
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    tt_input_grad = (
        ttl.tensor.Tensor(torch_input, ttl.tensor.DataType.BFLOAT16)
        .pad_to_tile(float("nan"))
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    tt_input_grad_ = ttl.operations.primary.moreh_nll_loss_backward(
        tt_input, tt_target, tt_weight, tt_divisor, tt_output_grad, tt_input_grad, ignore_index, reduction_mean
    )
    tt_input_grad_to_cpu = tt_input_grad_.cpu().to(ttl.tensor.Layout.ROW_MAJOR).unpad_from_tile(shape).to_torch()

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_input.grad, tt_input_grad_to_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing


@pytest.mark.parametrize("shape", ([1, 2], [3, 4], [12, 6]))
@pytest.mark.parametrize("ignore_index", [0, -1])
@pytest.mark.parametrize("reduction_mean", [True, False])
def test_moreh_nll_loss_2d_backward(shape, ignore_index, reduction_mean, device):
    device.enable_program_cache()

    (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors_2d(shape, device)
    nll_loss = torch.nn.NLLLoss(
        weight=torch_weight, ignore_index=ignore_index, reduction="mean" if reduction_mean else "sum"
    )
    torch_loss = nll_loss(torch_input, torch_target)

    (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors_2d(
        torch_input, torch_target, torch_weight, torch_divisor, torch_output, device
    )
    if reduction_mean == False:
        tt_divisor = None
    tt_loss = ttl.operations.primary.moreh_nll_loss(
        tt_input, tt_target, tt_weight, tt_divisor, tt_output, ignore_index, reduction_mean
    )
    tt_loss_to_cpu = tt_loss.cpu().to(ttl.tensor.Layout.ROW_MAJOR).unpad_from_tile([1, 1, 1, 1]).to_torch().reshape([1])

    output_grad = torch.randn_like(torch_loss)
    torch_loss.backward(output_grad)

    tt_output_grad = (
        ttl.tensor.Tensor(output_grad.reshape(1, 1, 1, 1), ttl.tensor.DataType.BFLOAT16)
        .pad_to_tile(float("nan"))
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    tt_input_grad = (
        ttl.tensor.Tensor(torch_input.unsqueeze(-1).unsqueeze(-1), ttl.tensor.DataType.BFLOAT16)
        .pad_to_tile(float("nan"))
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    ttl.operations.primary.moreh_nll_loss_backward(
        tt_input, tt_target, tt_weight, tt_divisor, tt_output_grad, tt_input_grad, ignore_index, reduction_mean
    )
    tt_input_grad_to_cpu = (
        tt_input_grad.cpu()
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .unpad_from_tile(tt_input_grad.shape_without_padding())
        .to_torch()
        .reshape(shape)
    )

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_input.grad, tt_input_grad_to_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing
