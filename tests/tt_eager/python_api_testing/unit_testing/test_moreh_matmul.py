# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0
from models.utility_functions import comp_pcc


def get_tensors(input_shape, other_shape, transpose_input, transpose_other,
                device):
    torch.manual_seed(2023)
    dtype = ttl.tensor.DataType.BFLOAT16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR

    # create input tensors using torch
    input = torch.randn(input_shape, dtype=torch.bfloat16)
    other = torch.randn(other_shape, dtype=torch.bfloat16)

    # TT matmul
    # set different padded value for tt_input and tt_other.
    tt_input = (ttl.tensor.Tensor(
        input.reshape(-1).tolist(), input_shape, dtype,
        cpu_layout).pad_to_tile(1).to(npu_layout).to(device))

    tt_other = (ttl.tensor.Tensor(
        other.reshape(-1).tolist(), other_shape, dtype,
        cpu_layout).pad_to_tile(float("nan")).to(npu_layout).to(device))

    torch_input = torch.transpose(input, 2, 3) if transpose_input else input
    torch_other = torch.transpose(other, 2, 3) if transpose_other else other

    return tt_input, tt_other, torch_input, torch_other


@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 1, 1, 10],  # test not mutiple of 32 case
        [1, 1, 1, 32],  # test single tile
        [1, 1, 1, 640],  # test multiple tiles
        [1, 1, 1, 623],  # test multiple tiles, not a multiple of 32
    ),
)
def test_moreh_matmul_1d(input_shape, device):
    if input_shape[0] != 1 or input_shape[1] != 1 or input_shape[2] != 1:
        pytest.skip(f"dim 0, 1, 2 should be 1")

    # get tensors
    tt_input, tt_other, torch_input, torch_other = get_tensors(
        input_shape, input_shape, False, False, device)

    # tt matmul
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    output_shape = [1, 1, 1, 1]
    tt_out = (ttl.tensor.moreh_matmul(tt_input, tt_other).cpu().to(
        cpu_layout).unpad_from_tile(output_shape).to_torch())

    # torch matmul
    torch_input = torch.reshape(torch_input, (torch_input.shape[-1], ))
    torch_other = torch.reshape(torch_other, (torch_other.shape[-1], ))
    torch_out = torch.matmul(torch_input, torch_other)

    # test for equivalance
    passing_pcc, output_pcc = comp_pcc(torch_out, tt_out, pcc=0.999)
    logger.info(f"Out passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize(
    "params",
    (
        # input, other, output shape
        ([1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32]),
        ([1, 1, 29, 31], [1, 1, 31, 30], [1, 1, 29, 30]),
        ([3, 3, 511, 313], [1, 1, 313, 765], [3, 3, 511, 765]),
        ([1, 1, 511, 313], [3, 3, 313, 765], [3, 3, 511, 765]),
        ([1, 3, 511, 313], [1, 1, 313, 765], [1, 3, 511, 765]),
        ([3, 1, 511, 313], [1, 1, 313, 765], [3, 1, 511, 765]),
        ([1, 1, 511, 313], [1, 3, 313, 765], [1, 3, 511, 765]),
        ([1, 1, 511, 313], [3, 1, 313, 765], [3, 1, 511, 765]),
        ([1, 3, 511, 313], [3, 1, 313, 765], [3, 3, 511, 765]),
        ([3, 1, 511, 313], [1, 3, 313, 765], [3, 3, 511, 765]),
        ([1, 3, 511, 313], [1, 3, 313, 765], [1, 3, 511, 765]),
        ([3, 1, 511, 313], [3, 1, 313, 765], [3, 1, 511, 765]),
        ([3, 3, 511, 313], [3, 3, 313, 765], [3, 3, 511, 765]),
    ))
def test_moreh_matmul(params, device):
    input_shape, other_shape, output_shape = params
    tt_input, tt_other, torch_input, torch_other = get_tensors(
        input_shape, other_shape, False, False, device)

    # tt matmul
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output = (ttl.tensor.moreh_matmul(tt_input, tt_other).cpu().to(
        cpu_layout).unpad_from_tile(output_shape).to_torch())

    # torch matmul
    torch_out = torch.matmul(torch_input, torch_other)

    # test for equivalance
    passing_pcc, output_pcc = comp_pcc(torch_out, tt_output, pcc=0.999)
    logger.info(f"Out passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize(
    "params",
    (
        # input, other, output shape, transpose input, other
        ([1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], False, False),
        ([1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], False, True),
        ([1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], True, False),
        ([1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], True, True),
        ([1, 1, 29, 31], [1, 1, 31, 30], [1, 1, 29, 30], False, False),
        ([1, 1, 29, 31], [1, 1, 30, 31], [1, 1, 29, 30], False, True),
        ([1, 1, 29, 31], [1, 1, 29, 30], [1, 1, 31, 30], True, False),
        ([1, 1, 29, 31], [1, 1, 30, 29], [1, 1, 31, 30], True, True),
        ([1, 3, 511, 313], [1, 1, 765, 313], [1, 3, 511, 765], False, True),
        ([1, 1, 511, 313], [1, 3, 765, 313], [1, 3, 511, 765], False, True),
        ([1, 3, 511, 313], [3, 1, 765, 313], [3, 3, 511, 765], False, True),
        ([3, 3, 511, 313], [3, 3, 765, 313], [3, 3, 511, 765], False, True),
        ([1, 1, 319, 309], [1, 1, 319, 748], [1, 1, 309, 748], True, False),
        ([1, 3, 313, 511], [1, 1, 313, 765], [1, 3, 511, 765], True, False),
        ([1, 1, 313, 511], [1, 3, 313, 765], [1, 3, 511, 765], True, False),
        ([1, 3, 313, 511], [3, 1, 313, 765], [3, 3, 511, 765], True, False),
        ([3, 3, 313, 511], [3, 3, 313, 765], [3, 3, 511, 765], True, False),
        ([3, 3, 313, 511], [3, 3, 765, 313], [3, 3, 511, 765], True, True),
    ))
def test_primary_moreh_matmul(params, device):
    input_shape, other_shape, output_shape, transpose_input, transpose_other = params
    tt_input, tt_other, torch_input, torch_other = get_tensors(
        input_shape, other_shape, transpose_input, transpose_other, device)

    # tt matmul
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output = (ttl.operations.primary.moreh_matmul(
        tt_input,
        tt_other,
        transpose_input=transpose_input,
        transpose_other=transpose_other).cpu().to(cpu_layout).unpad_from_tile(
            output_shape).to_torch())

    # torch matmul
    torch_out = torch.matmul(torch_input, torch_other)

    # test for equivalance
    passing_pcc, output_pcc = comp_pcc(torch_out, tt_output, pcc=0.999)
    logger.info(f"Out passing={passing_pcc}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing_pcc
