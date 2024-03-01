# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc, skip_for_wormhole_b0


def get_tensors(input_shape, other_shape, output_shape, require_input_grad, require_other_grad, is_1d, device):
    torch.manual_seed(2023)
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR

    # create tensors for forward
    input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype)
    other = torch.randint(-2, 3, other_shape, dtype=cpu_dtype)

    tt_input = ttl.tensor.Tensor(input, npu_dtype).pad_to_tile(1).to(npu_layout).to(device)

    tt_other = ttl.tensor.Tensor(other, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    torch_input = input.reshape(-1) if is_1d else input
    torch_other = other.reshape(-1) if is_1d else other

    # tensors for backward
    output_grad = tt_output_grad = torch_output_grad = tt_input_grad = tt_other_grad = None
    if require_input_grad or require_other_grad:
        output_grad = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
        tt_output_grad = ttl.tensor.Tensor(output_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
        torch_output_grad = output_grad[0][0][0][0] if is_1d else output_grad

        if require_input_grad:
            input_grad = torch.full(input_shape, float("nan"), dtype=cpu_dtype)
            tt_input_grad = ttl.tensor.Tensor(input_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

        if require_other_grad:
            other_grad = torch.full(other_shape, float("nan"), dtype=cpu_dtype)
            tt_other_grad = (
                ttl.tensor.Tensor(
                    other_grad,
                    npu_dtype,
                )
                .pad_to_tile(float("nan"))
                .to(npu_layout)
                .to(device)
            )

    return tt_input, tt_other, tt_output_grad, tt_input_grad, tt_other_grad, torch_input, torch_other, torch_output_grad


@skip_for_wormhole_b0()
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
    # get tensors
    output_shape = [1, 1, 1, 1]
    tt_input, tt_other, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, input_shape, output_shape, False, False, True, device
    )

    # tt matmul
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_out = ttl.tensor.moreh_matmul(tt_input, tt_other).cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    # torch matmul
    torch_input = torch.reshape(torch_input, (torch_input.shape[-1],))
    torch_other = torch.reshape(torch_other, (torch_other.shape[-1],))
    torch_out = torch.matmul(torch_input, torch_other)

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_out[0][0][0][0], pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 1, 1, 10],  # test not mutiple of 32 case
        [1, 1, 1, 32],  # test single tile
        [1, 1, 1, 640],  # test multiple tiles
        [1, 1, 1, 623],  # test multiple tiles, not a multiple of 32
    ),
)
@pytest.mark.parametrize(
    "requires_grad",
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
def test_moreh_matmul_1d_backward(input_shape, requires_grad, device):
    require_input_grad, require_other_grad = requires_grad
    output_shape = [1, 1, 1, 1]
    # get tensors
    (
        tt_input,
        tt_other,
        tt_output_grad,
        tt_input_grad,
        tt_other_grad,
        torch_input,
        torch_other,
        torch_output_grad,
    ) = get_tensors(input_shape, input_shape, output_shape, require_input_grad, require_other_grad, True, device)

    # torch matmul
    torch_out = torch.matmul(
        torch_input.requires_grad_(require_input_grad), torch_other.requires_grad_(require_other_grad)
    )
    torch_out.backward(torch_output_grad)

    # tt matmul backward
    ttl.operations.primary.moreh_matmul_backward(tt_output_grad, tt_input, tt_other, tt_input_grad, tt_other_grad)

    # test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    if require_input_grad:
        ttcpu_input_grad = tt_input_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()

        passing, output_pcc = comp_allclose_and_pcc(
            torch_input.grad, ttcpu_input_grad.reshape(-1), pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"input_grad passing={passing}")
        logger.debug(f"input_grad pcc={output_pcc}")
        assert passing

    if require_other_grad:
        ttcpu_other_grad = tt_other_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()

        passing, output_pcc = comp_allclose_and_pcc(
            torch_other.grad, ttcpu_other_grad.reshape(-1), pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"other_grad passing={passing}")
        logger.debug(f"other_grad pcc={output_pcc}")
        assert passing


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "params",
    (
        # input, other, output shape
        ([1, 1, 511, 255], [1, 1, 255, 765], [1, 1, 511, 765]),
        ([1, 1, 325, 127], [1, 1, 127, 126], [1, 1, 325, 126]),
    ),
)
@pytest.mark.parametrize("input_b1", (1, 2))
@pytest.mark.parametrize("input_b2", (1, 3))
@pytest.mark.parametrize("other_b1", (1, 2))
@pytest.mark.parametrize("other_b2", (1, 3))
@pytest.mark.parametrize(
    "requires_grad",
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
def test_moreh_matmul_backward(params, input_b1, input_b2, other_b1, other_b2, requires_grad, device):
    input_shape, other_shape, output_shape = params
    input_shape[0] = input_b1
    input_shape[1] = input_b2
    other_shape[0] = other_b1
    other_shape[1] = other_b2
    output_shape[0] = max(input_b1, other_b1)
    output_shape[1] = max(input_b2, other_b2)

    require_input_grad, require_other_grad = requires_grad

    # get tensors
    (
        tt_input,
        tt_other,
        tt_output_grad,
        tt_input_grad,
        tt_other_grad,
        torch_input,
        torch_other,
        torch_output_grad,
    ) = get_tensors(input_shape, other_shape, output_shape, require_input_grad, require_other_grad, False, device)

    # torch matmul
    torch_out = torch.matmul(
        torch_input.requires_grad_(require_input_grad), torch_other.requires_grad_(require_other_grad)
    )
    torch_out.backward(torch_output_grad)

    # tt matmul backward
    ttl.operations.primary.moreh_matmul_backward(tt_output_grad, tt_input, tt_other, tt_input_grad, tt_other_grad)
    # test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    if require_input_grad:
        ttcpu_input_grad = tt_input_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()

        # TODO(dongjin.na): Check this case.
        if input_b1 == 1 and input_b2 == 1 and other_b1 == 2 and other_b2 == 3 and input_shape[2] == 511:
            atol = 1

        passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, ttcpu_input_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"input_grad passing={passing}")
        logger.debug(f"input_grad pcc={output_pcc}")
        assert passing

    if require_other_grad:
        ttcpu_other_grad = tt_other_grad.cpu().to(cpu_layout).unpad_from_tile(other_shape).to_torch()

        passing, output_pcc = comp_allclose_and_pcc(torch_other.grad, ttcpu_other_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"other_grad passing={passing}")
        logger.debug(f"other_grad pcc={output_pcc}")
        assert passing


@skip_for_wormhole_b0()
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
    ),
)
def test_moreh_matmul(params, device):
    input_shape, other_shape, output_shape = params
    tt_input, tt_other, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, other_shape, output_shape, False, False, False, device
    )

    # tt matmul
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output = (
        ttl.tensor.moreh_matmul(tt_input, tt_other).cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()
    )

    # torch matmul
    torch_out = torch.matmul(torch_input, torch_other)

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_output, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing


@skip_for_wormhole_b0()
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
    ),
)
def test_primary_moreh_matmul(params, device):
    input_shape, other_shape, output_shape, transpose_input, transpose_other = params
    tt_input, tt_other, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, other_shape, output_shape, False, False, False, device
    )

    torch_input = torch_input.transpose(3, 2) if transpose_input else torch_input
    torch_other = torch_other.transpose(3, 2) if transpose_other else torch_other

    # tt matmul
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output = (
        ttl.operations.primary.moreh_matmul(
            tt_input, tt_other, transpose_input_a=transpose_input, transpose_input_b=transpose_other
        )
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    # torch matmul
    rtol = atol = 0.1
    torch_out = torch.matmul(torch_input, torch_other)

    # test for equivalance
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_output, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing
