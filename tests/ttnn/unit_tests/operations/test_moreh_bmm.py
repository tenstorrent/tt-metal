# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.unit_testing.misc.test_moreh_matmul import get_tensors
from models.utility_functions import comp_allclose_and_pcc

from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)


# @pytest.mark.parametrize(
#     "shape",
#     (
#         # batch, m, k, n
#         [1, 31, 639, 31],
#         [5, 95, 415, 65],
#         [10, 191, 447, 159],
#     ),
# )
# @pytest.mark.parametrize("has_output", [False, True])
# @pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
# def test_moreh_bmm(shape, has_output, compute_kernel_options, device):
#     input_shape = [shape[0], shape[1], shape[2]]
#     mat2_shape = [shape[0], shape[2], shape[3]]
#     output_shape = [shape[0], shape[1], shape[3]]

#     # get tensors
#     tt_input, tt_mat2, tt_output, _, _, _, input, mat2, _ = get_tensors(
#         input_shape, mat2_shape, output_shape, False, False, False, device
#     )

#     compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

#     # tt bmm
#     cpu_layout = ttnn.ROW_MAJOR_LAYOUT
#     tt_output = (
#         ttnn.operations.moreh.bmm(
#             tt_input, tt_mat2, output=tt_output if has_output else None, compute_kernel_config=compute_kernel_config
#         )
#         .cpu()
#         .to(cpu_layout)
#         .unpad_from_tile(output_shape)
#         .to_torch()
#     )

#     # torch bmm
#     output = torch.bmm(input, mat2)

#     ## test for equivalance
#     passing, output_pcc = comp_allclose_and_pcc(output, tt_output, pcc=0.999)
#     logger.debug(f"Out passing={passing}")
#     logger.debug(f"Output pcc={output_pcc}")

#     assert passing


@pytest.mark.parametrize(
    "shape",
    (
        # batch, m, k, n
        [1, 32, 32, 32],
        [3, 31, 31, 31],
        [7, 511, 313, 765],
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
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_bmm_backward(shape, requires_grad, compute_kernel_options, device):
    require_input_grad, require_mat2_grad = requires_grad
    input_shape = [shape[0], shape[1], shape[2]]
    mat2_shape = [shape[0], shape[2], shape[3]]
    output_shape = [shape[0], shape[1], shape[3]]

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    logger.debug("Came 1")
    # get tensors
    (
        tt_input,
        tt_mat2,
        _,
        tt_output_grad,
        tt_input_grad,
        tt_mat2_grad,
        input,
        mat2,
        output_grad,
    ) = get_tensors(input_shape, mat2_shape, output_shape, require_input_grad, require_mat2_grad, False, device)

    # tt bmm fwd, bwd
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    logger.debug("Came 2")
    logger.debug(f"tt_output_grad={tt_output_grad}")
    logger.debug(f"tt_input={tt_input}")
    logger.debug(f"tt_mat2={tt_mat2}")
    logger.debug(f"require_input_grad={require_input_grad}")
    logger.debug(f"require_mat2_grad={require_mat2_grad}")
    logger.debug(f"input_grad={tt_input_grad if require_input_grad else None}")
    logger.debug(f"mat2_grad={tt_mat2_grad if require_mat2_grad else None}")
    logger.debug(f"compute_kernel_config={compute_kernel_config}")

    ttnn.operations.moreh.bmm_backward(
        tt_output_grad,
        tt_input,
        tt_mat2,
        are_required_outputs=(require_input_grad, require_mat2_grad),
        input_grad=tt_input_grad if require_input_grad else None,
        mat2_grad=tt_mat2_grad if require_mat2_grad else None,
        compute_kernel_config=compute_kernel_config,
    )
    logger.debug("Came 3")

    # torch bmm fwd, bwd
    output = torch.bmm(input.requires_grad_(require_input_grad), mat2.requires_grad_(require_mat2_grad))
    output.backward(output_grad)
    logger.debug("Came 4")

    # test for equivalance
    rtol = atol = 0.1
    if require_input_grad:
        ttcpu_input_grad = tt_input_grad.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(input.grad, ttcpu_input_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"input_grad passing={passing}")
        logger.debug(f"input_grad pcc={output_pcc}")
        assert passing

    if require_mat2_grad:
        ttcpu_mat2_grad = tt_mat2_grad.cpu().to(cpu_layout).unpad_from_tile(mat2_shape).to_torch()
        passing, output_pcc = comp_allclose_and_pcc(mat2.grad, ttcpu_mat2_grad, pcc=0.999, rtol=rtol, atol=atol)
        logger.debug(f"mat2_grad passing={passing}")
        logger.debug(f"mat2_grad pcc={output_pcc}")
        assert passing
