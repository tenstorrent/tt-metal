# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.common.utility_functions import comp_allclose_and_pcc
from loguru import logger

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_torch,
    to_ttnn,
)

# Module-scoped device: opens once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device


def get_torch_tensors(shape, torch_dtype):
    C = shape[1]
    target_shape = shape[:1] + shape[2:]

    cpu_index_dtype = torch.long

    torch_input = torch.rand(shape, dtype=torch_dtype).requires_grad_()
    torch_target = torch.randint(0, C, target_shape, dtype=cpu_index_dtype)
    torch_weight = torch.rand(C, dtype=torch_dtype)
    torch_divisor = torch.tensor([0], dtype=torch_dtype)
    torch_output = torch.tensor([0], dtype=torch_dtype)

    return torch_input, torch_target, torch_weight, torch_divisor, torch_output


def get_tt_tensors(torch_input, torch_target, torch_weight, torch_divisor, torch_output, device, ttnn_dtype):
    npu_index_dtype = ttnn.int32

    tt_input = to_ttnn(torch_input, dtype=ttnn_dtype, device=device)
    tt_target = to_ttnn(torch_target, dtype=npu_index_dtype, device=device)
    tt_weight = to_ttnn(torch_weight, dtype=ttnn_dtype, device=device)
    tt_divisor = to_ttnn(torch_divisor, dtype=ttnn_dtype, device=device)
    tt_output = to_ttnn(torch_output, dtype=ttnn_dtype, device=device)

    return tt_input, tt_target, tt_weight, tt_divisor, tt_output


def run_moreh_nll_loss(
    shape,
    ignore_index,
    reduction,
    none_weight,
    device,
    *,
    torch_dtype=torch.float32,
    ttnn_dtype=ttnn.bfloat16,
    compute_kernel_options=None,
):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors(shape, torch_dtype)

    if none_weight:
        torch_weight = None

    nll_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=ignore_index, reduction=reduction)
    torch_loss = torch.tensor([nll_loss(torch_input, torch_target)])

    (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors(
        torch_input, torch_target, torch_weight, torch_divisor, torch_output, device, ttnn_dtype
    )

    assert reduction in ["sum", "mean"]

    tt_loss = ttnn.operations.moreh.nll_loss(
        tt_input,
        tt_target,
        reduction,  # reduction_mean,
        weight_tensor=tt_weight,
        divisor_tensor=tt_divisor,
        output_tensor=tt_output,
        ignore_index=ignore_index,
        compute_kernel_config=compute_kernel_config,
    )

    tt_loss = to_torch(tt_loss, shape=[1])
    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_loss, tt_loss, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")
    assert passing


def run_moreh_nll_loss_backward(
    shape,
    ignore_index,
    reduction_mean,
    none_weight,
    device,
    *,
    torch_dtype=torch.float32,
    ttnn_dtype=ttnn.bfloat16,
    compute_kernel_options=None,
):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors(shape, torch_dtype)
    if none_weight:
        torch_weight = None

    nll_loss = torch.nn.NLLLoss(
        weight=torch_weight, ignore_index=ignore_index, reduction="mean" if reduction_mean else "sum"
    )
    torch_loss = nll_loss(torch_input, torch_target)

    (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors(
        torch_input, torch_target, torch_weight, torch_divisor, torch_output, device, ttnn_dtype
    )
    reduction = "mean"
    if reduction_mean == False:
        tt_divisor = None
        reduction = "sum"
    tt_loss = ttnn.operations.moreh.nll_loss(
        tt_input,
        tt_target,
        reduction,
        weight_tensor=tt_weight,
        divisor_tensor=tt_divisor,
        output_tensor=tt_output,
        ignore_index=ignore_index,
        compute_kernel_config=compute_kernel_config,
    )

    # run backward
    output_grad = torch.randn_like(torch_loss)
    torch_loss.backward(output_grad)

    tt_output_grad = to_ttnn(output_grad, device=device)
    tt_input_grad = to_ttnn(torch_input, device=device)

    tt_input_grad = ttnn.operations.moreh.nll_loss_backward(
        target_tensor=tt_target,
        weight_tensor=tt_weight,
        divisor_tensor=tt_divisor,
        output_grad_tensor=tt_output_grad,
        input_grad_tensor=tt_input_grad,
        ignore_index=ignore_index,
        reduction_mean=reduction_mean,
        compute_kernel_config=compute_kernel_config,
    )
    tt_input_grad = to_torch(tt_input_grad, shape=shape)

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_input.grad, tt_input_grad, pcc=0.999, rtol=rtol, atol=atol)

    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing


@pytest.mark.parametrize(
    "shape",
    [
        [5, 10],
        [5, 50, 2, 7, 50, 70],
    ],
)
@pytest.mark.parametrize("ignore_index", [1])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("none_weight", [True, False])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_moreh_nll_loss(shape, ignore_index, reduction, none_weight, device, ttnn_dtype):
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip("Support for bfloat8_b is currently unavailable.")

    torch.manual_seed(0)
    run_moreh_nll_loss(shape, ignore_index, reduction, none_weight, device)


@pytest.mark.parametrize(
    "shape",
    [
        [5, 10],
        [5, 6, 7],
        [5, 6, 8, 9],
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_moreh_nll_loss_callback(shape, reduction, device):
    torch.manual_seed(0)
    ignore_index = 0

    # Start from an empty cache: the module-scoped device carries entries over from earlier tests in this file.
    device.clear_program_cache()
    num_program_cache_entries_list = []
    for i in range(4):
        if i < 2:
            none_weight = True
        else:
            none_weight = False

        run_moreh_nll_loss(shape, ignore_index, reduction, none_weight, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)

        num_program_cache_entries_list.append(device.num_program_cache_entries())

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    # Guard that the op registers cached programs at all; the equality checks alone
    # would still pass even if it never does.
    assert num_program_cache_entries_list[0] > 0
    assert (
        num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
        and num_program_cache_entries_list[2] == num_program_cache_entries_list[3]
    )


@pytest.mark.parametrize(
    "shape",
    [
        [5, 10],
        [10, 20, 30],
        [10, 20, 30, 40],
    ],
)
@pytest.mark.parametrize("ignore_index", [1])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("none_weight", [True, False])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_moreh_nll_loss_compute_kernel_options(
    shape, ignore_index, reduction, none_weight, compute_kernel_options, device, ttnn_dtype
):
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip("Support for bfloat8_b is currently unavailable.")

    torch.manual_seed(0)
    run_moreh_nll_loss(
        shape, ignore_index, reduction, none_weight, device, compute_kernel_options=compute_kernel_options
    )


@pytest.mark.parametrize(
    "shape",
    [
        [400, 300],
        [20, 300, 320],
        [5, 2, 5, 40, 70],
    ],
)
@pytest.mark.parametrize("ignore_index", [1])
@pytest.mark.parametrize("reduction_mean", [True, False])
@pytest.mark.parametrize("none_weight", [True, False])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_moreh_nll_loss_backward(shape, ignore_index, reduction_mean, none_weight, device, ttnn_dtype):
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip("Support for bfloat8_b is currently unavailable.")

    torch.manual_seed(0)
    run_moreh_nll_loss_backward(shape, ignore_index, reduction_mean, none_weight, device)


@pytest.mark.parametrize(
    "shape",
    [
        [2, 3],
        [2, 3, 4],
        [2, 3, 5, 4],
    ],
)
@pytest.mark.parametrize("reduction_mean", [True, False])
def test_moreh_nll_loss_backward_test_callback(shape, reduction_mean, device):
    torch.manual_seed(0)

    ignore_index = 0

    # Start from an empty cache: the module-scoped device carries entries over from earlier tests in this file.
    device.clear_program_cache()
    num_program_cache_entries_list = []
    for i in range(4):
        if i < 2:
            none_weight = True
        else:
            none_weight = False

        run_moreh_nll_loss_backward(shape, ignore_index, reduction_mean, none_weight, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)

        num_program_cache_entries_list.append(device.num_program_cache_entries())

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    # Guard that the op registers cached programs at all; the equality checks alone
    # would still pass even if it never does.
    assert num_program_cache_entries_list[0] > 0
    assert (
        num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
        and num_program_cache_entries_list[2] == num_program_cache_entries_list[3]
    )


@pytest.mark.parametrize(
    "shape",
    [
        [5, 10],
        [10, 20, 30, 40],
    ],
)
@pytest.mark.parametrize("reduction_mean", [True, False])
@pytest.mark.parametrize("none_weight", [True, False])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_moreh_nll_loss_backward_compute_kernel_options(
    shape, reduction_mean, none_weight, compute_kernel_options, ttnn_dtype, device
):
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip("Support for bfloat8_b is currently unavailable.")

    torch.manual_seed(0)
    ignore_index = 0

    run_moreh_nll_loss_backward(
        shape, ignore_index, reduction_mean, none_weight, device, compute_kernel_options=compute_kernel_options
    )


# ---------------------------------------------------------------------------
# Regression tests for the step2 reader fixes (items 6 and 7 of #51278). The
# helpers above never generate a target that is actually equal to ignore_index
# or otherwise out of range, and the 3d coverage stops at W = 7 (< FACE_WIDTH),
# so the ignored-target path of the 2d reader and the multi-face path of the
# 3d reader were unreachable through the existing tests.
# ---------------------------------------------------------------------------


def run_moreh_nll_loss_regression(
    shape,
    ignore_index,
    reduction,
    none_weight,
    device,
    *,
    has_ignored=False,
    torch_dtype=torch.float32,
    ttnn_dtype=ttnn.bfloat16,
    compute_kernel_options=None,
):
    """Like run_moreh_nll_loss, but with control over whether targets equal to
    ignore_index are actually present in the target tensor."""
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    (torch_input, torch_target, torch_weight, torch_divisor, torch_output) = get_torch_tensors(shape, torch_dtype)

    if has_ignored:
        # Deterministically ignore every other target so that the ignored and
        # the valid path run in the same kernel launch, while at least one
        # valid target always remains (an all-ignored "mean" would divide by
        # a zero divisor).
        torch_target.view(-1)[::2] = ignore_index

    if none_weight:
        torch_weight = None

    nll_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=ignore_index, reduction=reduction)
    torch_loss = torch.tensor([nll_loss(torch_input, torch_target)])

    (tt_input, tt_target, tt_weight, tt_divisor, tt_output) = get_tt_tensors(
        torch_input, torch_target, torch_weight, torch_divisor, torch_output, device, ttnn_dtype
    )

    assert reduction in ["sum", "mean"]

    tt_loss = ttnn.operations.moreh.nll_loss(
        tt_input,
        tt_target,
        reduction,
        weight_tensor=tt_weight,
        divisor_tensor=tt_divisor,
        output_tensor=tt_output,
        ignore_index=ignore_index,
        compute_kernel_config=compute_kernel_config,
    )

    tt_loss = to_torch(tt_loss, shape=[1])
    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(torch_loss, tt_loss, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")
    assert passing


@pytest.mark.parametrize("has_ignored", [True, False])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_moreh_nll_loss_negative_ignore_index_2d(reduction, has_ignored, device):
    """Weighted 2d loss with ignore_index = -100 (item 6 of #51278).

    The stock reader_moreh_nll_loss_step2_2d.cpp looked the weight up outside
    the target-validity branch, so for a target equal to ignore_index it still
    computed noc_id = (uint32)(-100) / TILE_WIDTH -- a wild page id far past
    the weight tensor -- and stored whatever the NOC read returned into
    tmp_weight, leaving the ignored row's contribution 0 x garbage instead of
    a deterministic 0 x 0. With the fix the weight lookup is gated on target
    validity, matching the 4d reader. has_ignored=False is the all-valid
    control: the weight path must be unchanged for valid targets.
    """
    torch.manual_seed(0)
    run_moreh_nll_loss_regression(
        [5, 10],
        -100,
        reduction,
        none_weight=False,
        device=device,
        has_ignored=has_ignored,
    )


@pytest.mark.parametrize(
    "shape, reduction, none_weight",
    [
        # Over-iteration zone, no page overrun: the stock bound is already correct here.
        [[2, 10, 33], "sum", True],
        [[2, 10, 33], "sum", False],
        [[2, 10, 33], "mean", True],
        [[2, 10, 33], "mean", False],
        # CB-page overrun zone: the stock bound makes the trailing faces write tmp_input (and, when
        # weights are used, tmp_weight) past the 1024-element CB page into the neighbouring CBs.
        # Weighted sum is the case where that clobbered data is actually consumed.
        [[2, 10, 2048], "sum", True],
        [[2, 10, 2048], "sum", False],
        [[2, 10, 2048], "mean", True],
        [[2, 10, 2048], "mean", False],
        [[2, 10, 4096], "mean", True],
        [[2, 10, 4096], "mean", False],
    ],
)
def test_moreh_nll_loss_step2_3d_multi_face(shape, reduction, none_weight, device):
    """3d loss with W > FACE_WIDTH (item 7 of #51278).

    The stock reader_moreh_nll_loss_step2_3d.cpp bounded its per-face target
    loop with idx_max = min(w + FACE_WIDTH, W), mixing an absolute W coordinate
    into a face-local count: for w > 0 the loop ran past the 16 target int32s
    actually loaded into the CB page, and for W > 1024 the trailing faces
    wrote tmp_input (and tmp_weight) past the 1024-element CB page into the
    neighbouring CBs. W = 33 exercises the over-iteration zone (past the
    loaded face, still within the page); W = 2048 and W = 4096 exercise the
    page-overrun zone. With the fix the bound is face-local:
    min(FACE_WIDTH, W - w).

    Measured on ttsim with this file, stock kernel vs the kernel of this PR
    (same test binary, cache cleared between runs): the W = 2048 weighted-sum,
    W = 4096 mean and W = 4096 weighted-mean cases FAIL without the fix
    (|got-ref| = 269.3 / 0.248 / 461.5 against a tolerance of
    atol + rtol * |got|) and PASS with it (|got-ref| = 22.7 / 0.0018 / 0.020).
    The remaining cases are the all-valid / no-overrun controls: they pass
    either way, which is what the pre-existing coverage covered.
    """
    torch.manual_seed(0)
    run_moreh_nll_loss_regression(
        shape,
        -100,
        reduction,
        none_weight=none_weight,
        device=device,
        has_ignored=False,
    )
