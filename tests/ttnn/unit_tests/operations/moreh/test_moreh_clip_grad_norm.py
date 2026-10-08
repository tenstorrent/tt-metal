# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options

pytestmark = pytest.mark.use_module_device

# Nightly has no clip_grad_norm helper (its logic is inline in test_* functions), so this file has its own.
# step1 runs one core per input (Sum[|x|^p]), step2 one core (Sum^(1/p)), step3 one core per input (x *= clip_coef).
# Nothing is a compile-time define: p and 1/p are split into integer part, fraction and sign at runtime.


def run_moreh_clip_grad_norm_test(
    input_shapes, norm_type, device, clip_ratio=0.5, max_norm=None, error_if_nonfinite=False, provide_total_norm=False
):
    torch_params = []
    tt_inputs = []
    for shape in input_shapes:
        param = torch.nn.Parameter(torch.empty(shape))
        param.grad = torch.empty(shape).uniform_(0, 2.5)
        torch_params.append(param)
        tt_inputs.append(create_ttnn_tilized_tensor(param.grad.bfloat16(), device, ttnn.bfloat16))

    # By default max_norm is relative to the actual norm, so every norm_type clips by the same factor and the scaled
    # gradients stay near 1, where a wrong clip coefficient shows. clip_ratio >= 1 leaves the gradients unchanged.
    if max_norm is None:
        norm = torch.linalg.vector_norm(torch.cat([param.grad.flatten() for param in torch_params]), ord=norm_type)
        max_norm = clip_ratio * norm.item()
    torch_total_norm = torch.nn.utils.clip_grad_norm_(
        torch_params, max_norm, norm_type, error_if_nonfinite=error_if_nonfinite
    )

    # NaN, so a total norm the op never writes fails.
    tt_total_norm = None
    if provide_total_norm:
        tt_total_norm = create_ttnn_tilized_tensor(torch.full([1, 1], float("nan")), device, ttnn.bfloat16)
    result = ttnn.moreh_clip_grad_norm(
        tt_inputs,
        max_norm,
        norm_type,
        error_if_nonfinite=error_if_nonfinite,
        total_norm=tt_total_norm,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    # Relative only: a negative norm_type gives a total norm around 1e-5, which an absolute tolerance of 0.1 would
    # pass whatever the op returned.
    actual_total_norm = ttnn.to_torch(tt_total_norm if provide_total_norm else result).reshape(1).float()
    assert torch.allclose(actual_total_norm, torch_total_norm.reshape(1), rtol=0.1, atol=0), (
        actual_total_norm,
        torch_total_norm,
    )

    # step3 scales the inputs in place.
    for param, tt_input in zip(torch_params, tt_inputs):
        passing, output_pcc = comp_allclose_and_pcc(param.grad, ttnn.to_torch(tt_input), rtol=0.1, atol=0.1)
        assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "norm_type",
    [
        # Integer p in step1; step2 raises to 1/p = 0.5, a fraction.
        2.0,
        # 1/p = 1: step2's fraction is 0.
        1.0,
        # Fractional p in step1.
        2.2,
        # Negative p and 1/p: both steps take their reciprocal paths.
        -0.8,
    ],
    ids=["p2", "p1", "p2_2", "p_negative"],
)
def test_moreh_clip_grad_norm(norm_type, device):
    torch.manual_seed(0)
    run_moreh_clip_grad_norm_test([[1, 1, 32, 32]] * 3, norm_type, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shapes, clip_ratio, error_if_nonfinite, provide_total_norm",
    [
        # Partly filled tiles and several tiles per input: step1 masks the padding and loops over tiles.
        ([[1, 1, 30, 50], [2, 3, 40, 20], [1, 2, 64, 70]], 0.5, False, False),
        # More inputs than any device has cores: step1 and step3 run in several rounds.
        ([[1, 1, 32, 32]] * 149, 0.5, False, False),
        # max_norm twice the norm: the clip coefficient clamps to 1 and the gradients stay unchanged.
        ([[1, 1, 32, 32]] * 3, 2.0, False, False),
        # A finite norm passes the check, which reads the total norm back to the host before step3.
        ([[1, 1, 32, 32]] * 3, 0.5, True, False),
        ([[1, 1, 32, 32]] * 3, 0.5, False, True),
    ],
    ids=["unaligned", "multi_round", "no_clipping", "error_if_nonfinite_finite_norm", "provided_total_norm"],
)
def test_moreh_clip_grad_norm_corner_cases(input_shapes, clip_ratio, error_if_nonfinite, provide_total_norm, device):
    torch.manual_seed(0)
    run_moreh_clip_grad_norm_test(
        input_shapes,
        2.0,
        device,
        clip_ratio=clip_ratio,
        error_if_nonfinite=error_if_nonfinite,
        provide_total_norm=provide_total_norm,
    )
