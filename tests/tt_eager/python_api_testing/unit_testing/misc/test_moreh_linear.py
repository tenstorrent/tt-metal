# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc, skip_for_wormhole_b0
from tests.tt_eager.python_api_testing.unit_testing.misc.test_moreh_matmul import get_tensors
from loguru import logger


# TODO: add this feature in get_tensors method
def get_bias_tensors(bias_shape, require_bias_grad, device):
    torch.manual_seed(2023)
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR

    bias = torch.randint(-2, 3, bias_shape, dtype=cpu_dtype)
    tt_bias = ttl.tensor.Tensor(bias, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    tt_bias_grad = None
    if require_bias_grad:
        bias_grad = torch.full(bias_shape, float("nan"), dtype=cpu_dtype)
        tt_bias_grad = ttl.tensor.Tensor(bias_grad, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_bias, bias, tt_bias_grad


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        ([1, 1, 1, 31], [1, 1, 30, 31], [1, 1, 1, 30], [1, 1, 1, 30]),
        ([1, 1, 1, 31], [1, 1, 30, 31], [1, 1, 1, 1], [1, 1, 1, 30]),
        ([1, 1, 31, 31], [1, 1, 30, 31], [1, 1, 1, 30], [1, 1, 31, 30]),
        ([1, 1, 31, 31], [1, 1, 30, 31], [1, 1, 1, 1], [1, 1, 31, 30]),
        ([4, 4, 2, 31], [1, 1, 30, 31], [1, 1, 1, 30], [4, 4, 2, 30]),
        ([4, 4, 2, 31], [1, 1, 30, 31], [1, 1, 1, 1], [4, 4, 2, 30]),
        ([1, 1, 2, 2047], [1, 1, 1023, 2047], [1, 1, 1, 1023], [1, 1, 2, 1023]),
        ([1, 1, 2, 2047], [1, 1, 1023, 2047], [1, 1, 1, 1], [1, 1, 2, 1023]),
        ([1, 1, 32, 64], [1, 1, 1024, 64], [1, 1, 1, 1024], [1, 1, 32, 1024]),
        ([1, 1, 32, 64], [1, 1, 1024, 64], [1, 1, 1, 1], [1, 1, 32, 1024]),
        ([1, 1, 32, 1023], [1, 1, 2047, 1023], [1, 1, 1, 2047], [1, 1, 32, 2047]),
        ([1, 1, 32, 1023], [1, 1, 2047, 1023], [1, 1, 1, 1], [1, 1, 32, 2047]),
        ([2, 4, 4, 1024], [1, 1, 2047, 1024], [1, 1, 1, 2047], [2, 4, 4, 2047]),
        ([2, 4, 4, 1024], [1, 1, 2047, 1024], [1, 1, 1, 1], [2, 4, 4, 2047]),
    ),
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_moreh_linear(shapes, has_bias, device):
    input_shape, weight_shape, bias_shape, output_shape = shapes
    tt_input, tt_weight, _, _, _, torch_input, torch_weight, _ = get_tensors(
        input_shape, weight_shape, output_shape, False, False, False, device
    )

    if has_bias:
        tt_bias, torch_bias, _ = get_bias_tensors(bias_shape, False, device)
        tt_output = ttl.operations.primary.moreh_linear(tt_input, tt_weight, tt_bias)
    else:
        torch_bias = None
        tt_output = ttl.operations.primary.moreh_linear(tt_input, tt_weight)

    ## reference
    torch_output = torch.nn.functional.linear(torch_input, torch_weight[0][0], torch_bias)

    ## test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    ttcpu_output = tt_output.cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()
    passing, output_pcc = comp_allclose_and_pcc(torch_output, ttcpu_output, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Passing = {passing}")
    logger.debug(f"Output PCC = {output_pcc}")

    assert passing


@pytest.mark.parametrize(
    "shapes",
    (
        # input, weight, bias(1d or scalar), output
        ([1, 1, 1, 31], [1, 1, 30, 31], [1, 1, 1, 30], [1, 1, 1, 30]),
        ([1, 1, 1, 31], [1, 1, 30, 31], [1, 1, 1, 1], [1, 1, 1, 30]),
        ([1, 1, 31, 31], [1, 1, 30, 31], [1, 1, 1, 30], [1, 1, 31, 30]),
        ([1, 1, 31, 31], [1, 1, 30, 31], [1, 1, 1, 1], [1, 1, 31, 30]),
        ([4, 4, 2, 31], [1, 1, 30, 31], [1, 1, 1, 30], [4, 4, 2, 30]),
        ([4, 4, 2, 31], [1, 1, 30, 31], [1, 1, 1, 1], [4, 4, 2, 30]),
        ([1, 1, 2, 2047], [1, 1, 1023, 2047], [1, 1, 1, 1023], [1, 1, 2, 1023]),
        ([1, 1, 2, 2047], [1, 1, 1023, 2047], [1, 1, 1, 1], [1, 1, 2, 1023]),
        ([1, 1, 32, 64], [1, 1, 1024, 64], [1, 1, 1, 1024], [1, 1, 32, 1024]),
        ([1, 1, 32, 64], [1, 1, 1024, 64], [1, 1, 1, 1], [1, 1, 32, 1024]),
        ([1, 1, 32, 1023], [1, 1, 1536, 1023], [1, 1, 1, 1536], [1, 1, 32, 1536]),
        ([1, 1, 32, 1023], [1, 1, 1536, 1023], [1, 1, 1, 1], [1, 1, 32, 1536]),
        ([2, 4, 4, 1024], [1, 1, 1536, 1024], [1, 1, 1, 1536], [2, 4, 4, 1536]),
        # TODO: Check this case with 1300 -> 1536
        ([2, 4, 4, 1024], [1, 1, 1300, 1024], [1, 1, 1, 1], [2, 4, 4, 1300]),
    ),
)
@skip_for_wormhole_b0("disabled due to watcher error, see issue #5868")
@pytest.mark.parametrize(
    "requires_grads",
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
@pytest.mark.parametrize("requires_bias_grad", [True, False])
def test_moreh_linear_backward(shapes, requires_grads, requires_bias_grad, device):
    input_shape, weight_shape, bias_shape, output_shape = shapes
    requires_input_grad, requires_weight_grad = requires_grads
    if not requires_input_grad and not requires_weight_grad and not requires_bias_grad:
        pytest.skip("At least one grad is requires")

    (
        tt_input,
        tt_weight,
        tt_output_grad,
        tt_input_grad,
        tt_weight_grad,
        torch_input,
        torch_weight,
        torch_output_grad,
    ) = get_tensors(input_shape, weight_shape, output_shape, requires_input_grad, requires_weight_grad, False, device)

    _, torch_bias, tt_bias_grad = get_bias_tensors(bias_shape, requires_bias_grad, device)

    ## tt linear backward
    ttl.operations.primary.moreh_linear_backward(
        tt_output_grad, tt_input, tt_weight, tt_input_grad, tt_weight_grad, tt_bias_grad
    )
    ## reference
    torch_weight = torch_weight.reshape(-1, torch_weight.shape[3])
    torch_output = torch.nn.functional.linear(
        torch_input.requires_grad_(requires_input_grad),
        torch_weight.requires_grad_(requires_weight_grad),
        torch_bias.requires_grad_(requires_bias_grad),
    )
    torch_output.backward(torch_output_grad)

    ## test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    if requires_input_grad:
        ttcpu_input_grad = tt_input_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, ttcpu_input_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"input_grad passing={passing} pcc={output_pcc}")
        assert passing

    if requires_weight_grad:
        ttcpu_weight_grad = tt_weight_grad.cpu().to(cpu_layout).unpad_from_tile(weight_shape).to_torch()[0][0]
        passing, output_pcc = comp_allclose_and_pcc(
            torch_weight.grad, ttcpu_weight_grad, pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"weight_grad passing={passing} pcc={output_pcc}")
        assert passing

    if requires_bias_grad:
        ttcpu_bias_grad = tt_bias_grad.cpu().to(cpu_layout).unpad_from_tile(bias_shape).to_torch()

        passing, output_pcc = comp_allclose_and_pcc(torch_bias.grad, ttcpu_bias_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"bias_grad passing={passing} pcc={output_pcc}")
        assert passing
