# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
)

pytestmark = pytest.mark.use_module_device

INPUT_SHAPE = [2, 3, TILE_HEIGHT, TILE_WIDTH]
# Spans two tiles in H and W without filling them: exercises the tile loops and the padding masks.
UNALIGNED_SHAPE = [2, 3, TILE_HEIGHT * 2 - 1, TILE_WIDTH * 2 - 1]

# On a rank-4 input: dim 3 -> W factory, dim 2 -> H factory, dim 1 -> NC factory.
FACTORY_DIMS = [3, 2, 1]
FACTORY_IDS = ["w", "h", "nc"]

INF = float("inf")


def run_moreh_norm_test(input_shape, p, dim, device, keepdim=True, provide_output=False):
    torch_input = torch.empty(input_shape).uniform_(-1, 1)
    torch_output = torch.norm(torch_input, p=p, dim=dim, keepdim=keepdim)

    tt_input = create_ttnn_tilized_tensor(torch_input, device, ttnn.bfloat16)
    tt_output = (
        create_ttnn_tilized_tensor(torch.zeros_like(torch_output), device, ttnn.bfloat16) if provide_output else None
    )
    tt_output = ttnn.moreh_norm(
        tt_input,
        p,
        dim=dim,
        keepdim=keepdim,
        output=tt_output,
        compute_kernel_config=get_compute_kernel_options(False),
    )
    actual = ttnn.to_torch(tt_output).reshape(torch_output.shape)

    passing, out = comp_allclose(torch_output, actual, rtol=0.06, atol=0.06)
    assert passing, out


def run_moreh_norm_backward_test(input_shape, p, dim, device, keepdim=True):
    torch_input = torch.empty(input_shape).uniform_(-1, 1).requires_grad_()
    torch_output = torch.norm(torch_input, p=p, dim=dim, keepdim=keepdim)
    torch_output_grad = torch.empty(torch_output.shape).uniform_(-1, 1)
    torch_output.backward(torch_output_grad)

    tt_input = create_ttnn_tilized_tensor(torch_input.detach(), device, ttnn.bfloat16)
    tt_output = create_ttnn_tilized_tensor(torch_output.detach(), device, ttnn.bfloat16)
    tt_output_grad = create_ttnn_tilized_tensor(torch_output_grad, device, ttnn.bfloat16)
    tt_input_grad = ttnn.to_torch(
        ttnn.moreh_norm_backward(
            tt_input,
            tt_output,
            tt_output_grad,
            p,
            dim=dim,
            keepdim=keepdim,
            compute_kernel_config=get_compute_kernel_options(False),
        )
    )

    passing, out = comp_allclose(torch_input.grad, tt_input_grad, rtol=0.06, atol=0.06)
    assert passing, out


@pytest.mark.merge_gate
@pytest.mark.parametrize("dim", FACTORY_DIMS, ids=FACTORY_IDS)
def test_moreh_norm(dim, device):
    torch.manual_seed(0)
    # p is inf, so this runs the moreh_norm device op. Other p values call moreh_sum, which is left uncovered.
    run_moreh_norm_test(INPUT_SHAPE, float("inf"), dim, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, p, dim, keepdim",
    [
        (UNALIGNED_SHAPE, INF, 3, True),
        (UNALIGNED_SHAPE, INF, 2, True),
        (UNALIGNED_SHAPE, INF, 1, True),
        # p=0 counts nonzeros (sum reduce); p=-inf masks padding to +inf and reduces min(|x|).
        (UNALIGNED_SHAPE, 0.0, 3, True),
        (UNALIGNED_SHAPE, 0.0, 2, True),
        (UNALIGNED_SHAPE, 0.0, 1, True),
        (UNALIGNED_SHAPE, -INF, 3, True),
        (UNALIGNED_SHAPE, -INF, 2, True),
        (UNALIGNED_SHAPE, -INF, 1, True),
        (UNALIGNED_SHAPE, INF, [2, 3], True),
        (INPUT_SHAPE, INF, None, True),
        (INPUT_SHAPE, INF, [0, 1], False),
    ],
    ids=[
        "w_unaligned",
        "h_unaligned",
        "nc_unaligned",
        "p0_w",
        "p0_h",
        "p0_nc",
        "minus_inf_w",
        "minus_inf_h",
        "minus_inf_nc",
        "hw",
        "all_dims",
        "nc_keepdim_false",
    ],
)
def test_moreh_norm_corner_cases(input_shape, p, dim, keepdim, device):
    torch.manual_seed(0)
    run_moreh_norm_test(input_shape, p, dim, device, keepdim=keepdim)


@pytest.mark.merge_gate
def test_moreh_norm_provided_output(device):
    torch.manual_seed(0)
    run_moreh_norm_test(INPUT_SHAPE, INF, 3, device, provide_output=True)


@pytest.mark.merge_gate
def test_moreh_norm_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_norm_test(INPUT_SHAPE, INF, 3, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = create_ttnn_tilized_tensor(torch.zeros(INPUT_SHAPE), device, ttnn.bfloat16)
    run_moreh_norm_test(INPUT_SHAPE, INF, 3, device)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
def test_moreh_norm_backward(device):
    torch.manual_seed(0)
    run_moreh_norm_backward_test(INPUT_SHAPE, 2.0, 1, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, p, dim, keepdim",
    [
        (UNALIGNED_SHAPE, 2.0, [2, 3], True),
        (INPUT_SHAPE, 2.0, None, True),
        (INPUT_SHAPE, 2.0, [0, 1], False),
        # A fractional p takes the decimal-exponent path of the power ladder.
        (INPUT_SHAPE, 2.5, 3, True),
    ],
    ids=["hw_unaligned", "all_dims", "nc_keepdim_false", "p2_5"],
)
def test_moreh_norm_backward_corner_cases(input_shape, p, dim, keepdim, device):
    torch.manual_seed(0)
    run_moreh_norm_backward_test(input_shape, p, dim, device, keepdim=keepdim)
