# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
)

pytestmark = pytest.mark.use_module_device

# Same shapes and norm_type as the factory test, so these cases reuse its ~7 compiled programs.
INPUT_SHAPES = [[1, 1, TILE_HEIGHT, TILE_WIDTH]] * 3


def run_moreh_clip_grad_norm_test(input_shapes, max_norm, norm_type, device, error_if_nonfinite=False):
    torch_params = []
    tt_inputs = []
    for shape in input_shapes:
        param = torch.nn.Parameter(torch.empty(shape))
        param.grad = torch.empty(shape).uniform_(0, 2.5)
        torch_params.append(param)
        tt_inputs.append(create_ttnn_tilized_tensor(param.grad.bfloat16(), device, ttnn.bfloat16))

    torch_total_norm = torch.nn.utils.clip_grad_norm_(
        torch_params, max_norm, norm_type, error_if_nonfinite=error_if_nonfinite
    )
    tt_total_norm = ttnn.moreh_clip_grad_norm(
        tt_inputs,
        max_norm,
        norm_type,
        error_if_nonfinite=error_if_nonfinite,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    actual_total_norm = ttnn.to_torch(tt_total_norm).reshape(1)
    passing, output_pcc = comp_allclose_and_pcc(torch_total_norm.reshape(1), actual_total_norm, rtol=0.1, atol=0.1)
    assert passing, output_pcc

    for param, tt_input in zip(torch_params, tt_inputs):
        passing, output_pcc = comp_allclose_and_pcc(param.grad, ttnn.to_torch(tt_input), rtol=0.1, atol=0.1)
        assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_clip_grad_norm(device):
    torch.manual_seed(0)
    # The total norm of these gradients is ~80, so max_norm=2.0 makes step 3 actually scale them down.
    run_moreh_clip_grad_norm_test([[1, 1, TILE_HEIGHT, TILE_WIDTH]] * 3, max_norm=2.0, norm_type=2.0, device=device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "max_norm, error_if_nonfinite",
    [
        # Above the ~80 total norm: the clip coefficient clamps to 1 and the gradients stay unchanged.
        (1000.0, False),
        # A finite norm passes the check, which reads the total norm back to the host before step 3.
        (2.0, True),
    ],
    ids=["no_clipping", "error_if_nonfinite_finite_norm"],
)
def test_moreh_clip_grad_norm_corner_cases(max_norm, error_if_nonfinite, device):
    torch.manual_seed(0)
    run_moreh_clip_grad_norm_test(
        INPUT_SHAPES, max_norm=max_norm, norm_type=2.0, device=device, error_if_nonfinite=error_if_nonfinite
    )


@pytest.mark.merge_gate
def test_moreh_clip_grad_norm_nonfinite_norm_type(device, expect_error):
    torch.manual_seed(0)
    # A NaN norm_type is rejected on the host before any program runs.
    tt_input = create_ttnn_tilized_tensor(torch.rand(INPUT_SHAPES[0]).bfloat16(), device, ttnn.bfloat16)
    with expect_error(RuntimeError, "The total norm of order"):
        ttnn.moreh_clip_grad_norm([tt_input], 1.0, float("nan"), error_if_nonfinite=True)
