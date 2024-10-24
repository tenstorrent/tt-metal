# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.utility_functions import comp_allclose
from loguru import logger


from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    TILE_HEIGHT,
    TILE_WIDTH,
    to_ttnn,
)


def torch_group_norm(input, num_groups, gamma=None, beta=None, eps=1e-05, compute_mean_rstd=True):
    N = input.shape[0]

    output = F.group_norm(input, num_groups, gamma, beta, eps)

    mean = rstd = None
    if compute_mean_rstd:
        x_view = input.view(N, num_groups, -1)
        mean = x_view.mean(dim=-1, keepdim=True)
        var = ((x_view - mean) ** 2).mean(dim=-1, keepdim=False)
        rstd = (var + eps).rsqrt()
        mean = mean.view(N, num_groups)
        rstd = rstd.view(N, num_groups)

    return output, mean, rstd


def torch_group_norm_backward(
    input,
    output_grad,
    num_groups,
    input_requires_grad,
    gamma_requires_grad,
    beta_requires_grad,
    gamma=None,
    beta=None,
    eps=1e-05,
):
    if not input_requires_grad and not gamma_requires_grad and not beta_requires_grad:
        return None, None, None

    input.requires_grad_(input_requires_grad)
    if gamma is not None:
        gamma.requires_grad_(gamma_requires_grad)
    if beta is not None:
        beta.requires_grad_(beta_requires_grad)

    output = F.group_norm(input, num_groups, gamma, beta, eps)
    output.backward(output_grad)

    gamma_grad = beta_grad = None
    if gamma is not None:
        gamma_grad = gamma.grad
    if beta is not None:
        beta_grad = beta.grad

    return input.grad, gamma_grad, beta_grad


def tt_group_norm(
    input, num_groups, gamma=None, beta=None, eps=1e-05, compute_mean_rstd=True, device=None, compute_kernel_config=None
):
    N = input.shape[0]

    mean_rstd_shape = [N, num_groups]

    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT

    npu_input = ttnn.from_torch(input, dtype=dtype, layout=layout, device=device)
    if gamma is not None:
        npu_gamma = ttnn.from_torch(gamma, dtype=dtype, layout=layout, device=device)
    else:
        npu_gamma = None

    if beta is not None:
        npu_beta = ttnn.from_torch(beta, dtype=dtype, layout=layout, device=device)
    else:
        npu_beta = None

    npu_mean = npu_rstd = None
    if compute_mean_rstd:
        npu_mean = torch.empty(mean_rstd_shape, dtype=torch.bfloat16)
        npu_mean = ttnn.from_torch(npu_mean, dtype=dtype, layout=layout, device=device)
        npu_rstd = torch.empty(mean_rstd_shape, dtype=torch.bfloat16)
        npu_rstd = ttnn.from_torch(npu_rstd, dtype=dtype, layout=layout, device=device)

    # Forward
    npu_output, npu_mean, npu_rstd = ttnn.moreh_group_norm(
        npu_input,
        num_groups,
        eps,
        npu_gamma,
        npu_beta,
        are_required_outputs=(True, compute_mean_rstd, compute_mean_rstd),
        mean=npu_mean,
        rstd=npu_rstd,
        compute_kernel_config=compute_kernel_config,
    )

    tt_output = ttnn.to_torch(npu_output)

    tt_mean = tt_rstd = None
    if compute_mean_rstd:
        tt_mean = ttnn.to_torch(npu_mean)
        tt_rstd = ttnn.to_torch(npu_rstd)

    return tt_output, tt_mean, tt_rstd


def tt_group_norm_backward(
    input,
    output_grad,
    num_groups,
    input_requires_grad,
    gamma_requires_grad,
    beta_requires_grad,
    gamma=None,
    eps=1e-05,
    device=None,
):
    N, C, _, _ = input.shape

    gamma_beta_shape = [1, 1, 1, C]
    mean_rstd_shape = [1, 1, N, num_groups]

    x_view = input.view(N, num_groups, -1)
    mean = x_view.mean(dim=-1, keepdim=True)
    var = ((x_view - mean) ** 2).mean(dim=-1, keepdim=False)
    rstd = (var + eps).rsqrt()

    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT

    npu_output_grad = ttnn.from_torch(output_grad, dtype=dtype, layout=layout, device=device)
    npu_input = ttnn.from_torch(input, dtype=dtype, layout=layout, device=device)
    # below codes will be modified to ttnn.from_torch after refactoring backward.
    npu_mean = to_ttnn(mean, device=device, shape=mean_rstd_shape)
    npu_rstd = to_ttnn(rstd, device=device, shape=mean_rstd_shape)
    npu_gamma = to_ttnn(gamma, device=device, shape=gamma_beta_shape)

    npu_dx = None
    if input_requires_grad:
        npu_dx = torch.empty(input.shape, dtype=torch.bfloat16)
        npu_dx = ttnn.from_torch(npu_dx, dtype=dtype, layout=layout, device=device)

    npu_dg = None
    if gamma_requires_grad:
        npu_dg = torch.empty(gamma_beta_shape, dtype=torch.bfloat16)
        npu_dg = ttnn.from_torch(npu_dg, dtype=dtype, layout=layout, device=device)

    npu_db = None
    if beta_requires_grad:
        npu_db = torch.empty(gamma_beta_shape, dtype=torch.bfloat16)
        npu_db = ttnn.from_torch(npu_db, dtype=dtype, layout=layout, device=device)

    # Backward
    npu_dx, npu_dg, npu_db = ttnn.operations.moreh.group_norm_backward(
        npu_output_grad,
        npu_input,
        npu_mean,
        npu_rstd,
        num_groups,
        are_required_outputs=(input_requires_grad, gamma_requires_grad, beta_requires_grad),
        gamma=npu_gamma,
        input_grad=npu_dx,
        gamma_grad=npu_dg,
        beta_grad=npu_db,
    )

    tt_input_grad = ttnn.to_torch(npu_dx) if npu_dx else None

    tt_gamma_grad = tt_beta_grad = None
    if npu_dg is not None:
        tt_gamma_grad = ttnn.to_torch(npu_dg).view(C)
    if npu_db is not None:
        tt_beta_grad = ttnn.to_torch(npu_db).view(C)

    return tt_input_grad, tt_gamma_grad, tt_beta_grad


def make_input_tensors(input_shape, affine, do_backward=False):
    C = input_shape[1]

    # output_grad_shape
    output_grad_shape = input_shape

    # gamma_beta_shape
    gamma_beta_shape = [C]

    # cpu_dtype
    cpu_dtype = torch.float32

    # input
    input = torch.empty(input_shape, dtype=cpu_dtype).uniform_(-2, 2)

    # gamma
    gamma = None
    if affine:
        gamma = torch.empty(gamma_beta_shape, dtype=cpu_dtype).uniform_(-2, 2)

    # beta
    beta = None
    if affine:
        beta = torch.empty(gamma_beta_shape, dtype=cpu_dtype).uniform_(-2, 2)

    # output_grad
    output_grad = None
    if do_backward:
        output_grad = torch.empty(output_grad_shape, dtype=cpu_dtype).uniform_(-2, 2)

    return input, gamma, beta, output_grad


def run_test_moreh_group_norm(
    input_shape, num_groups, eps, device, affine=False, compute_mean_rstd=False, compute_kernel_options=None
):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    cpu_input, cpu_beta, cpu_gamma, _ = make_input_tensors(input_shape, affine)

    # expected
    expected_output, expected_mean, expected_rstd = torch_group_norm(
        cpu_input, num_groups, cpu_gamma, cpu_beta, eps, compute_mean_rstd
    )
    # actual
    actual_output, actual_mean, actual_rstd = tt_group_norm(
        cpu_input,
        num_groups,
        cpu_gamma,
        cpu_beta,
        eps,
        compute_mean_rstd=compute_mean_rstd,
        device=device,
        compute_kernel_config=compute_kernel_config,
    )

    # Set rtol and atol
    rtol = atol = 0.08

    # Check output
    pass_output, out_output = comp_allclose(expected_output, actual_output, rtol=rtol, atol=atol)
    logger.debug(f"output's {out_output}")
    assert pass_output

    # Check mean
    if compute_mean_rstd:
        pass_mean, out_mean = comp_allclose(expected_mean, actual_mean, rtol=rtol, atol=atol)
        logger.debug(f"mean's {out_mean}")
        assert pass_mean
    else:
        assert actual_mean is None

    # Check rstd
    if compute_mean_rstd:
        pass_rstd, out_rstd = comp_allclose(expected_rstd, actual_rstd, rtol=rtol, atol=atol)
        logger.debug(f"rstd's {out_rstd}")
        assert pass_rstd
    else:
        assert actual_rstd is None


@pytest.mark.parametrize(
    "input_shape, num_groups",
    [
        # 2d
        [[1, 32], 1],  # simple case
        [[1, 64], 2],  # num_groups is 2
        [[1, 62], 2],  # not multiple of 32
        [[20, 400], 100],  # complex case
        # 3d
        [[1, 5, 6], 1],  # simple case
        [[1, 64, 6], 2],  # num_groups is 2
        [[1, 62, 32], 2],  # not multiple of 32
        [[20, 60, 100], 5],  # complex case
        # 4d
        [[1, 1, 32, 32], 1],
        [[1, 1, 32, 80], 1],
        [[1, 10, 32, 80], 2],
        # 6d
        [[5, 6, 4, 3, 32, 80], 3],
    ],
)
@pytest.mark.parametrize(
    "affine",
    [
        True,
        False,
    ],
)
def test_moreh_group_norm(input_shape, num_groups, affine, device):
    """
    This is a test for various types of n-dimensional input tensors.
    In this test, `mean` and `rstd` are always included in the output.
    Testing both the cases where they are computed and where they are not would result in too many tests.
    """
    torch.manual_seed(2024)

    eps = 1e-05
    compute_mean_rstd = True

    run_test_moreh_group_norm(
        input_shape, num_groups, eps, device=device, affine=affine, compute_mean_rstd=compute_mean_rstd
    )


@pytest.mark.parametrize(
    "input_shape, num_groups",
    [
        # 2d
        [[20, 400], 100],
        # 3d
        [[20, 60, 100], 5],
        # 6d
        [[5, 6, 4, 3, 32, 80], 3],
    ],
)
def test_moreh_group_norm_not_compute_mean_rstd(input_shape, num_groups, device):
    """
    This is a test for the case where `mean` and `rstd` are not output.
    """
    torch.manual_seed(2024)

    eps = 1e-05
    affine = True
    compute_mean_rstd = False

    run_test_moreh_group_norm(
        input_shape, num_groups, eps, device=device, affine=affine, compute_mean_rstd=compute_mean_rstd
    )


@pytest.mark.parametrize(
    "input_shape, num_groups",
    [
        # 2d
        [[20, 400], 100],
        # 3d
        [[20, 60, 100], 5],
        # 4d
        [[20, 10, 32, 80], 5],
    ],
)
@pytest.mark.parametrize(
    "affine",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_group_norm_compute_kernel_options(input_shape, num_groups, affine, compute_kernel_options, device):
    torch.manual_seed(2024)

    eps = 1e-05
    compute_mean_rstd = True

    run_test_moreh_group_norm(
        input_shape,
        num_groups,
        eps,
        device=device,
        affine=affine,
        compute_mean_rstd=compute_mean_rstd,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "input_shape, num_groups",
    [
        # 2d
        [[20, 400], 100],
        # 3d
        [[20, 60, 100], 5],
        # 4d
        [[20, 10, 32, 80], 2],
    ],
)
def test_moreh_group_norm_callback(input_shape, num_groups, device, use_program_cache):
    torch.manual_seed(2024)

    eps = 1e-05
    affine = True
    compute_mean_rstd = True

    num_program_cache_entries_list = []
    for _ in range(2):
        run_test_moreh_group_norm(
            input_shape, num_groups, eps, device=device, affine=affine, compute_mean_rstd=compute_mean_rstd
        )
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


def run_test_moreh_group_norm_backward(
    N, C_num_groups, HW, eps, affine, input_requires_grad, gamma_requires_grad, beta_requires_grad, device
):
    H, W = HW
    if not affine and (gamma_requires_grad or beta_requires_grad):
        pytest.skip("gamma_requires_grad and beta_requires_grad are only valid when affine is True.")

    C, num_groups = C_num_groups
    input_shape = (N, C, H, W)
    cpu_input, cpu_gamma, cpu_beta, cpu_output_grad = make_input_tensors(input_shape, affine, do_backward=True)

    # expected
    expected_input_grad, expected_gamma_grad, expected_beta_grad = torch_group_norm_backward(
        cpu_input,
        cpu_output_grad,
        num_groups,
        input_requires_grad,
        gamma_requires_grad,
        beta_requires_grad,
        cpu_gamma,
        cpu_beta,
        eps,
    )
    # actual
    actual_input_grad, actual_gamma_grad, actual_beta_grad = tt_group_norm_backward(
        cpu_input,
        cpu_output_grad,
        num_groups,
        input_requires_grad,
        gamma_requires_grad,
        beta_requires_grad,
        cpu_gamma,
        eps,
        device,
    )

    # Set rtol and atol
    rtol = atol = 0.1

    # Check input_grad
    if expected_input_grad is not None:
        pass_input_grad, out_input_grad = comp_allclose(expected_input_grad, actual_input_grad, rtol=rtol, atol=atol)
        logger.debug(f"input_grad's {out_input_grad}")
        assert pass_input_grad
    else:
        assert actual_input_grad is None

    # I divide gamma_grad and beta_grad by (N * C * Ht * Wt), because the error of bf16 sum increases.
    Ht = (H + TILE_HEIGHT - 1) // TILE_HEIGHT
    Wt = (W + TILE_WIDTH - 1) // TILE_WIDTH
    divisor = N * C * Ht * Wt

    # Check gamma_grad
    if expected_gamma_grad is not None:
        pass_gamma_grad, out_gamma_grad = comp_allclose(
            expected_gamma_grad / divisor, actual_gamma_grad / divisor, rtol=rtol, atol=atol
        )
        logger.debug(f"gamma_grad's {out_gamma_grad}")
        assert pass_gamma_grad
    else:
        assert actual_gamma_grad is None

    # Check beta_grad
    if expected_beta_grad is not None:
        pass_beta_grad, out_beta_grad = comp_allclose(
            expected_beta_grad / divisor, actual_beta_grad / divisor, rtol=rtol, atol=atol
        )
        logger.debug(f"beta_grad's {out_beta_grad}")
        assert pass_beta_grad
    else:
        assert actual_beta_grad is None


@pytest.mark.parametrize(
    "N",
    [
        2,
    ],
)
@pytest.mark.parametrize(
    "C_num_groups",
    [
        [4, 1],
        [4, 2],
        [4, 4],
    ],
)
@pytest.mark.parametrize(
    "HW",
    [[23, 23], [512, 512]],
)
@pytest.mark.parametrize(
    "eps",
    [
        1e-05,
    ],
)
@pytest.mark.parametrize(
    "affine",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "input_requires_grad",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "gamma_requires_grad",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "beta_requires_grad",
    [
        True,
        False,
    ],
)
def test_moreh_group_norm_backward(
    N, C_num_groups, HW, eps, affine, input_requires_grad, gamma_requires_grad, beta_requires_grad, device
):
    torch.manual_seed(2024)

    run_test_moreh_group_norm_backward(
        N, C_num_groups, HW, eps, affine, input_requires_grad, gamma_requires_grad, beta_requires_grad, device
    )


@pytest.mark.parametrize(
    "N",
    [
        2,
    ],
)
@pytest.mark.parametrize(
    "C_num_groups",
    [
        [4, 1],
    ],
)
@pytest.mark.parametrize(
    "HW",
    [
        [23, 23],
    ],
)
@pytest.mark.parametrize(
    "eps",
    [
        1e-05,
    ],
)
@pytest.mark.parametrize(
    "affine",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "input_requires_grad",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "gamma_requires_grad",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "beta_requires_grad",
    [
        True,
    ],
)
def test_moreh_group_norm_backward_callback(
    N,
    C_num_groups,
    HW,
    eps,
    affine,
    input_requires_grad,
    gamma_requires_grad,
    beta_requires_grad,
    device,
    use_program_cache,
):
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        run_test_moreh_group_norm_backward(
            N, C_num_groups, HW, eps, affine, input_requires_grad, gamma_requires_grad, beta_requires_grad, device
        )
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
