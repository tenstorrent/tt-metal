# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0
from models.utility_functions import comp_pcc

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(shape, device):
    input_shape = [1, shape[0], shape[1], shape[2]]
    mat2_shape = [1, shape[0], shape[2], shape[3]]
    output_shape = [1, shape[0], shape[1], shape[3]]

    torch.manual_seed(2023)
    dtype = ttl.tensor.DataType.BFLOAT16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR

    # create input tensors using torch
    input = torch.randn(input_shape, dtype=torch.bfloat16)
    mat2 = torch.randn(mat2_shape, dtype=torch.bfloat16)
    output_grad = torch.randn(output_shape, dtype=torch.bfloat16)

    # TT matmul
    # set different padded value for tt_a and tt_b.
    tt_input = (ttl.tensor.Tensor(
        input.reshape(-1).tolist(), input_shape, dtype,
        cpu_layout).pad_to_tile(1).to(npu_layout).to(device))

    tt_mat2 = (ttl.tensor.Tensor(
        mat2.reshape(-1).tolist(), mat2_shape, dtype,
        cpu_layout).pad_to_tile(float("nan")).to(npu_layout).to(device))

    tt_output_grad = (ttl.tensor.Tensor(
        output_grad.reshape(-1).tolist(), output_shape, dtype,
        cpu_layout).pad_to_tile(float("nan")).to(npu_layout).to(device))

    torch_input = input.reshape(-1, input_shape[2], input_shape[3])
    torch_mat2 = mat2.reshape(-1, mat2_shape[2], mat2_shape[3])
    torch_output_grad = output_grad.reshape(-1, output_shape[2],
                                            output_shape[3])

    return tt_input, tt_mat2, tt_output_grad, torch_input, torch_mat2, torch_output_grad, input_shape, mat2_shape, output_shape


@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "shape",
    (
        [1, 31, 639, 31],
        [5, 95, 415, 65],
        [10, 191, 447, 159],
        [20, 287, 479, 255],
    ),
)
def test_moreh_bmm(shape, device):
    # get tensors
    tt_input, tt_mat2, _, torch_input, torch_mat2, _, _, _, output_shape = get_tensors(
        shape, device)

    # tt bmm
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_out = ttl.tensor.moreh_bmm(
        tt_input,
        tt_mat2).cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    # torch bmm
    torch_out = torch.bmm(torch_input, torch_mat2)

    ## test for equivalance
    passing_pcc, output_pcc = comp_pcc(torch_out, tt_out, pcc=0.999)
    logger.info(f"Out passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize(
    "shape",
    (
        # input, other, output shape
        [1, 32, 32, 32],
        [3, 31, 31, 31],
        [5, 255, 765, 511],
        [7, 511, 313, 765],
    ))
def test_moreh_bmm_backward(shape, device):
    tt_input, tt_mat2, tt_output_grad, torch_input, torch_mat2, torch_output_grad, input_shape, mat2_shape, _ = get_tensors(
        shape, device)

    # tt matmul
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    ttdev_input_grad, ttdev_mat2_grad = ttl.tensor.moreh_bmm_backward(
        tt_output_grad, tt_input, tt_mat2)

    tt_input_grad = ttdev_input_grad.cpu().to(cpu_layout).unpad_from_tile(
        input_shape).to_torch()

    tt_mat2_grad = ttdev_mat2_grad.cpu().to(cpu_layout).unpad_from_tile(
        mat2_shape).to_torch()

    # torch matmul
    torch_out = torch.bmm(torch_input.requires_grad_(True),
                          torch_mat2.requires_grad_(True))
    torch_out.backward(torch_output_grad)

    # test for equivalance
    passing_pcc, output_pcc = comp_pcc(torch_input.grad,
                                       tt_input_grad,
                                       pcc=0.999)
    logger.info(f"input_grad passing={passing_pcc}")
    logger.info(f"input_grad pcc={output_pcc}")
    assert passing_pcc

    passing_pcc, output_pcc = comp_pcc(torch_mat2.grad,
                                       tt_mat2_grad,
                                       pcc=0.999)
    logger.info(f"mat2_grad passing={passing_pcc}")
    logger.info(f"mat2_grad pcc={output_pcc}")
    assert passing_pcc
