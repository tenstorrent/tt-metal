# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_nll_loss import (
    run_moreh_nll_loss_backward,
    run_moreh_nll_loss_regression,
)
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_nll_loss_unreduced import (
    get_torch_tensors as get_unreduced_torch_tensors,
    get_tt_backward_tensors,
    run_moreh_nll_loss_unreduced,
)
from tests.ttnn.unit_tests.operations.test_utils import get_compute_kernel_options, to_torch

pytestmark = pytest.mark.use_module_device

# "mean" runs step1 (per-target weights for the divisor), then step2 (the loss); "sum" runs step2 only. Both then
# reduce with moreh_sum. step2 has a reader per input rank (2d, 3d, 4d); [2, 10, 33] has W past one tile and one
# face, the 3d reader's multi-face path (#51278).
SHAPES = [[5, 10], [2, 10, 33], [2, 3, 5, 4]]
SHAPE_IDS = ["rank_2", "rank_3", "rank_4"]


def run_moreh_nll_loss_unreduced_backward_test(shape, device, none_weight=False, fp32_dest_acc_en=False):
    # Not nightly run_moreh_nll_loss_unreduced_backward: it pre-fills input_grad with the expected gradient, so a
    # kernel that writes nothing still passes. Same steps, but input_grad starts as the input values.
    torch_input, torch_target, torch_weight, _ = get_unreduced_torch_tensors(shape, torch.float32)
    if none_weight:
        torch_weight = None
    torch_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=1, reduction="none")(torch_input, torch_target)
    torch_output_grad = torch.randn_like(torch_loss)
    torch_loss.backward(torch_output_grad)

    tt_target, tt_weight, tt_output_grad, tt_input_grad = get_tt_backward_tensors(
        torch_target, torch_weight, torch_output_grad, torch_input.detach(), device, ttnn.bfloat16
    )
    # Check the input_grad buffer passed in, not the return value: the op must write into it.
    ttnn.moreh_nll_loss_unreduced_backward(
        tt_target,
        tt_output_grad,
        weight_tensor=tt_weight,
        input_grad_tensor=tt_input_grad,
        ignore_index=1,
        compute_kernel_config=get_compute_kernel_options(fp32_dest_acc_en),
    )

    passing, output_pcc = comp_allclose_and_pcc(
        torch_input.grad, to_torch(tt_input_grad, shape=shape), pcc=0.999, rtol=0.05, atol=0.05
    )
    assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_moreh_nll_loss(shape, reduction, device):
    torch.manual_seed(0)
    run_moreh_nll_loss_regression(shape, -100, reduction, False, device, compute_kernel_options=False)


@pytest.mark.merge_gate
def test_moreh_nll_loss_unreduced(device):
    torch.manual_seed(0)
    # "none" runs step2 alone and returns a loss per target.
    run_moreh_nll_loss_unreduced(
        [2, 3, 5, 4], 1, False, device, torch_dtype=torch.float32, compute_kernel_options=False
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shape, reduction, none_weight, ignore_index, has_ignored, fp32_dest_acc_en",
    [
        ([5, 10], "mean", True, -100, False, False),
        # Every other target is ignored, so the ignored and the valid path run in the same launch.
        ([5, 10], "mean", False, 1, True, False),
        ([5, 10], "mean", False, -100, True, False),
        # W = 2048 spans many faces and pages: the 3d reader used to write past the CB page here (#51278).
        ([2, 10, 2048], "sum", False, -100, False, False),
        ([5, 10], "mean", False, -100, False, True),
        # 32768 classes: the weight vector (1024 tiles) doesn't fit in L1, so step1 uses its large reader.
        ([4, 32768], "mean", False, -100, False, False),
    ],
    ids=["no_weight", "ignored_targets", "negative_ignore_index", "rank_3_wide", "fp32_dest_acc", "step1_large"],
)
def test_moreh_nll_loss_corner_cases(
    shape, reduction, none_weight, ignore_index, has_ignored, fp32_dest_acc_en, device
):
    torch.manual_seed(0)
    run_moreh_nll_loss_regression(
        shape,
        ignore_index,
        reduction,
        none_weight,
        device,
        has_ignored=has_ignored,
        compute_kernel_options=fp32_dest_acc_en,
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize("reduction_mean", [True, False], ids=["mean", "sum"])
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_moreh_nll_loss_backward(shape, reduction_mean, device):
    torch.manual_seed(0)
    run_moreh_nll_loss_backward(shape, -100, reduction_mean, False, device, compute_kernel_options=False)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shape, ignore_index, none_weight, fp32_dest_acc_en",
    [
        ([5, 10], -100, True, False),
        # 66 targets over 10 classes: some equal ignore_index (all 66 missing class 1 has ~0.1% odds, and the
        # seed fixes the draw).
        ([2, 10, 33], 1, False, False),
        ([5, 10], -100, False, True),
    ],
    ids=["no_weight", "ignored_targets", "fp32_dest_acc"],
)
def test_moreh_nll_loss_backward_corner_cases(shape, ignore_index, none_weight, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    run_moreh_nll_loss_backward(shape, ignore_index, True, none_weight, device, compute_kernel_options=fp32_dest_acc_en)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shape, none_weight, fp32_dest_acc_en",
    [
        (SHAPES[0], False, False),
        (SHAPES[1], False, False),
        (SHAPES[2], False, False),
        (SHAPES[0], True, False),
        (SHAPES[0], False, True),
    ],
    ids=SHAPE_IDS + ["no_weight", "fp32_dest_acc"],
)
def test_moreh_nll_loss_unreduced_backward(shape, none_weight, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    run_moreh_nll_loss_unreduced_backward_test(
        shape, device, none_weight=none_weight, fp32_dest_acc_en=fp32_dest_acc_en
    )
