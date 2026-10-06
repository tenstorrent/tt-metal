# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import get_compute_kernel_options, to_torch, to_ttnn

pytestmark = pytest.mark.use_module_device

INPUT_SHAPE = [5, 10]
IGNORE_INDEX = 1
# (N, C, W) with W past one tile and one face: the 3d readers' multi-face path (regression #51278).
RANK_3_SHAPE = [2, 10, 33]
RANK_4_SHAPE = [2, 3, 5, 4]


def get_torch_tensors(input_shape, ignored_target=None):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_target = torch.randint(0, input_shape[1], input_shape[:1] + input_shape[2:])
    if ignored_target is not None:
        # Every other target is ignored, so both paths run and a "mean" divisor stays non-zero.
        torch_target.view(-1)[::2] = ignored_target
    torch_weight = torch.rand(input_shape[1])
    return torch_input, torch_target, torch_weight


def run_moreh_nll_loss_test(
    input_shape, device, reduction="mean", none_weight=False, ignore_index=IGNORE_INDEX, has_ignored=False
):
    torch_input, torch_target, torch_weight = get_torch_tensors(input_shape, ignore_index if has_ignored else None)
    if none_weight:
        torch_weight = None
    nll_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=ignore_index, reduction=reduction)
    torch_loss = nll_loss(torch_input, torch_target).detach()
    if reduction != "none":
        torch_loss = torch_loss.reshape(1)

    # "mean" is the only reduction that runs both the step1 (divisor) and step2 (loss) device ops.
    tt_loss = ttnn.moreh_nll_loss(
        to_ttnn(torch_input.detach(), device=device),
        to_ttnn(torch_target, device=device, dtype=ttnn.int32),
        reduction,
        weight_tensor=to_ttnn(torch_weight, device=device),
        divisor_tensor=to_ttnn(torch.zeros(1), device=device) if reduction == "mean" else None,
        output_tensor=to_ttnn(torch.zeros(torch_loss.shape), device=device),
        ignore_index=ignore_index,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    passing, output_pcc = comp_allclose_and_pcc(
        torch_loss, to_torch(tt_loss, shape=torch_loss.shape), pcc=0.999, rtol=0.05, atol=0.05
    )
    assert passing, output_pcc


def run_moreh_nll_loss_backward_test(input_shape, device, reduction_mean=True, none_weight=False, has_ignored=False):
    torch_input, torch_target, torch_weight = get_torch_tensors(input_shape, IGNORE_INDEX if has_ignored else None)
    if none_weight:
        torch_weight = None
    nll_loss = torch.nn.NLLLoss(
        weight=torch_weight, ignore_index=IGNORE_INDEX, reduction="mean" if reduction_mean else "sum"
    )
    torch_loss = nll_loss(torch_input, torch_target)
    torch_output_grad = torch.randn_like(torch_loss)
    torch_loss.backward(torch_output_grad)

    tt_target = to_ttnn(torch_target, device=device, dtype=ttnn.int32)
    tt_weight = to_ttnn(torch_weight, device=device)
    tt_divisor = None
    if reduction_mean:
        tt_divisor = to_ttnn(torch.zeros(1), device=device)
        # The forward fills tt_divisor, which the mean-reduction backward reads.
        ttnn.moreh_nll_loss(
            to_ttnn(torch_input.detach(), device=device),
            tt_target,
            "mean",
            weight_tensor=tt_weight,
            divisor_tensor=tt_divisor,
            output_tensor=to_ttnn(torch.zeros(1), device=device),
            ignore_index=IGNORE_INDEX,
            compute_kernel_config=get_compute_kernel_options(False),
        )
    tt_input_grad = ttnn.moreh_nll_loss_backward(
        tt_target,
        to_ttnn(torch_output_grad, device=device),
        reduction_mean=reduction_mean,
        weight_tensor=tt_weight,
        input_grad_tensor=to_ttnn(torch_input.detach(), device=device),
        divisor_tensor=tt_divisor,
        ignore_index=IGNORE_INDEX,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    passing, output_pcc = comp_allclose_and_pcc(
        torch_input.grad, to_torch(tt_input_grad, shape=input_shape), pcc=0.999, rtol=0.05, atol=0.05
    )
    assert passing, output_pcc


def run_moreh_nll_loss_unreduced_backward_test(input_shape, device, none_weight=False):
    torch_input, torch_target, torch_weight = get_torch_tensors(input_shape)
    if none_weight:
        torch_weight = None
    nll_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=IGNORE_INDEX, reduction="none")
    torch_loss = nll_loss(torch_input, torch_target)
    torch_output_grad = torch.randn_like(torch_loss)
    torch_loss.backward(torch_output_grad)

    tt_input_grad = ttnn.moreh_nll_loss_unreduced_backward(
        to_ttnn(torch_target, device=device, dtype=ttnn.int32),
        to_ttnn(torch_output_grad, device=device),
        weight_tensor=to_ttnn(torch_weight, device=device),
        input_grad_tensor=to_ttnn(torch_input.detach(), device=device),
        ignore_index=IGNORE_INDEX,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    passing, output_pcc = comp_allclose_and_pcc(
        torch_input.grad, to_torch(tt_input_grad, shape=input_shape), pcc=0.999, rtol=0.05, atol=0.05
    )
    assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_nll_loss(device):
    torch.manual_seed(0)
    run_moreh_nll_loss_test(INPUT_SHAPE, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, reduction, none_weight, ignore_index, has_ignored",
    [
        (INPUT_SHAPE, "sum", False, IGNORE_INDEX, False),
        (INPUT_SHAPE, "none", False, IGNORE_INDEX, False),
        (INPUT_SHAPE, "mean", True, IGNORE_INDEX, False),
        (INPUT_SHAPE, "mean", False, IGNORE_INDEX, True),
        (INPUT_SHAPE, "mean", False, -100, True),
        (RANK_3_SHAPE, "mean", False, IGNORE_INDEX, False),
        (RANK_4_SHAPE, "mean", False, IGNORE_INDEX, False),
    ],
    ids=["sum", "none", "no_weight", "ignored_targets", "negative_ignore_index", "rank_3", "rank_4"],
)
def test_moreh_nll_loss_corner_cases(input_shape, reduction, none_weight, ignore_index, has_ignored, device):
    torch.manual_seed(0)
    run_moreh_nll_loss_test(
        input_shape,
        device,
        reduction=reduction,
        none_weight=none_weight,
        ignore_index=ignore_index,
        has_ignored=has_ignored,
    )


@pytest.mark.merge_gate
def test_moreh_nll_loss_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_nll_loss_test(INPUT_SHAPE, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = to_ttnn(torch.zeros(INPUT_SHAPE), device=device)
    run_moreh_nll_loss_test(INPUT_SHAPE, device)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
def test_moreh_nll_loss_backward(device):
    torch.manual_seed(0)
    run_moreh_nll_loss_backward_test(INPUT_SHAPE, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, reduction_mean, none_weight, has_ignored",
    [
        (INPUT_SHAPE, False, False, False),
        (INPUT_SHAPE, True, True, False),
        (INPUT_SHAPE, True, False, True),
        (RANK_3_SHAPE, True, False, False),
    ],
    ids=["sum", "no_weight", "ignored_targets", "rank_3"],
)
def test_moreh_nll_loss_backward_corner_cases(input_shape, reduction_mean, none_weight, has_ignored, device):
    torch.manual_seed(0)
    run_moreh_nll_loss_backward_test(
        input_shape, device, reduction_mean=reduction_mean, none_weight=none_weight, has_ignored=has_ignored
    )


@pytest.mark.merge_gate
def test_moreh_nll_loss_unreduced_backward(device):
    torch.manual_seed(0)
    run_moreh_nll_loss_unreduced_backward_test(INPUT_SHAPE, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, none_weight",
    [
        (INPUT_SHAPE, True),
        (RANK_3_SHAPE, False),
    ],
    ids=["no_weight", "rank_3"],
)
def test_moreh_nll_loss_unreduced_backward_corner_cases(input_shape, none_weight, device):
    torch.manual_seed(0)
    run_moreh_nll_loss_unreduced_backward_test(input_shape, device, none_weight=none_weight)
