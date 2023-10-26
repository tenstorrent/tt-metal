# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0
from models.utility_functions import comp_pcc


def get_tensors(shape, require_input_grad, require_mat2_grad, device):
    input_shape = [1, shape[0], shape[1], shape[2]]
    mat2_shape = [1, shape[0], shape[2], shape[3]]
    output_shape = [1, shape[0], shape[1], shape[3]]

    torch.manual_seed(2023)
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR

    # create tensors for forward
    input = torch.randn(input_shape, dtype=cpu_dtype)
    mat2 = torch.randn(mat2_shape, dtype=cpu_dtype)

    tt_input = (ttl.tensor.Tensor(
        input.reshape(-1).tolist(), input_shape, npu_dtype,
        cpu_layout).pad_to_tile(1).to(npu_layout).to(device))

    tt_mat2 = (ttl.tensor.Tensor(
        mat2.reshape(-1).tolist(), mat2_shape, npu_dtype,
        cpu_layout).pad_to_tile(float("nan")).to(npu_layout).to(device))

    torch_input = input.reshape(-1, input_shape[2], input_shape[3])
    torch_mat2 = mat2.reshape(-1, mat2_shape[2], mat2_shape[3])

    # tensors for backward
    output_grad = tt_output_grad = torch_output_grad = tt_input_grad = tt_mat2_grad = None
    if require_input_grad or require_mat2_grad:
        output_grad = torch.randn(output_shape, dtype=cpu_dtype)
        tt_output_grad = (ttl.tensor.Tensor(
            output_grad.reshape(-1).tolist(), output_shape, npu_dtype,
            cpu_layout).pad_to_tile(float("nan")).to(npu_layout).to(device))
        torch_output_grad = output_grad.reshape(-1, output_shape[2],
                                                output_shape[3])

        if require_input_grad:
            input_grad = torch.full(input_shape, float('nan'), dtype=cpu_dtype)
            tt_input_grad = ttl.tensor.Tensor(
                input_grad.flatten().tolist(),
                input_shape,
                npu_dtype,
                cpu_layout,
            ).pad_to_tile(float('nan')).to(npu_layout).to(device)

        if require_mat2_grad:
            mat2_grad = torch.full(mat2_shape, float('nan'), dtype=cpu_dtype)
            tt_mat2_grad = ttl.tensor.Tensor(
                mat2_grad.flatten().tolist(),
                mat2_shape,
                npu_dtype,
                cpu_layout,
            ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    return tt_input, tt_mat2, tt_output_grad, tt_input_grad, tt_mat2_grad, torch_input, torch_mat2, torch_output_grad, input_shape, mat2_shape, output_shape


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
    tt_input, tt_mat2, _, _, _, torch_input, torch_mat2, _, _, _, output_shape = get_tensors(
        shape, False, False, device)

    # tt bmm
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_out = ttl.operations.primary.moreh_bmm(
        tt_input,
        tt_mat2).cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    # torch bmm
    torch_out = torch.bmm(torch_input, torch_mat2)

    ## test for equivalance
    passing_pcc, output_pcc = comp_pcc(torch_out, tt_out, pcc=0.999)
    logger.info(f"Out passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize("shape", (
    [1, 32, 32, 32],
    [3, 31, 31, 31],
    [5, 255, 765, 511],
    [7, 511, 313, 765],
))
@pytest.mark.parametrize("requires_grad", (
    (True, False),
    (False, True),
    (True, True),
))
def test_moreh_bmm_backward(shape, requires_grad, device):
    require_input_grad, require_mat2_grad = requires_grad
    tt_input, tt_mat2, tt_output_grad, tt_input_grad, tt_mat2_grad, torch_input, torch_mat2, torch_output_grad, input_shape, mat2_shape, _ = get_tensors(
        shape, require_input_grad, require_mat2_grad, device)

    # tt bmm fwd, bwd
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    ttl.operations.primary.moreh_bmm_backward(tt_output_grad, tt_input,
                                              tt_mat2, tt_input_grad,
                                              tt_mat2_grad)

    # torch bmm fwd, bwd
    torch_out = torch.bmm(torch_input.requires_grad_(require_input_grad),
                          torch_mat2.requires_grad_(require_mat2_grad))
    torch_out.backward(torch_output_grad)

    # test for equivalance
    if require_input_grad:
        ttcpu_input_grad = tt_input_grad.cpu().to(cpu_layout).unpad_from_tile(
            input_shape).to_torch()

        passing_pcc, output_pcc = comp_pcc(torch_input.grad,
                                           ttcpu_input_grad,
                                           pcc=0.999)
        logger.info(f"input_grad passing={passing_pcc}")
        logger.info(f"input_grad pcc={output_pcc}")
        assert passing_pcc

    if require_mat2_grad:
        ttcpu_mat2_grad = tt_mat2_grad.cpu().to(cpu_layout).unpad_from_tile(
            mat2_shape).to_torch()

        passing_pcc, output_pcc = comp_pcc(torch_mat2.grad,
                                           ttcpu_mat2_grad,
                                           pcc=0.999)
        logger.info(f"mat2_grad passing={passing_pcc}")
        logger.info(f"mat2_grad pcc={output_pcc}")
        assert passing_pcc
