# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, skip_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_layer_norm import (
    make_input_tensors,
    run_moreh_layer_norm,
    run_moreh_layer_norm_backward,
    run_moreh_layer_norm_backward_with_gamma_or_beta,
    torch_layer_norm,
)
from tests.ttnn.unit_tests.operations.test_utils import get_compute_kernel_options, to_torch, to_ttnn

pytestmark = pytest.mark.use_module_device

# With normalized_dims=1 the op writes mean/rstd wrong whenever they hold more than one value (only every 16th
# value is correct), so mean/rstd are only requested there for single-row inputs.


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape_normalized_dims, elementwise_affine",
    [
        # normalized_dims=1 reduces each row (REDUCE_ROW); 20 is unaligned, so the reader masks W.
        (([1, 20], 1), True),
        (([1, 32], 1), False),
        # normalized_dims=2 reduces whole tiles (REDUCE_SCALAR); 77 x 109 makes the reader mask both H and W.
        (([2, 77, 109], 2), True),
        (([2, 64, 64], 2), False),
    ],
    ids=["lastdim_affine", "lastdim_no_affine_aligned", "hw_affine", "hw_no_affine_aligned"],
)
def test_moreh_layer_norm(input_shape_normalized_dims, elementwise_affine, device):
    torch.manual_seed(0)
    run_moreh_layer_norm(
        input_shape_normalized_dims, elementwise_affine, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape_normalized_dims, fp32_dest_acc_en",
    [
        (([2, 3, 45, 45], 3), False),
        (([1, 20], 1), True),
        # 2 x 500 x 1000 = 512 tiles per normalized group: too big for L1, so the large (streaming) kernels run.
        (([1, 2, 500, 1000], 2), False),
    ],
    ids=["normalized_dims_3", "fp32_dest_acc", "large_algorithm"],
)
def test_moreh_layer_norm_corner_cases(input_shape_normalized_dims, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    run_moreh_layer_norm(
        input_shape_normalized_dims, True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=fp32_dest_acc_en
    )


@pytest.mark.merge_gate
def test_moreh_layer_norm_allocated_output_no_mean_rstd(device):
    torch.manual_seed(0)
    # Not run_moreh_layer_norm: it always passes `output`, and with create_mean_rstd=False it doesn't check that
    # mean/rstd come back as None. 109 spans four tiles in W.
    input_shape = [3, 77, 109]
    cpu_input, cpu_gamma, cpu_beta, _ = make_input_tensors(input_shape, 1, True)
    expected_output, _, _ = torch_layer_norm(cpu_input, normalized_dims=1, eps=1e-5, gamma=cpu_gamma, beta=cpu_beta)
    tt_output, tt_mean, tt_rstd = ttnn.moreh_layer_norm(
        to_ttnn(cpu_input, device=device),
        1,
        1e-5,
        to_ttnn(cpu_gamma, device=device),
        to_ttnn(cpu_beta, device=device),
        compute_kernel_config=get_compute_kernel_options(False),
    )

    assert tt_mean is None and tt_rstd is None
    passing, out = comp_allclose(expected_output, to_torch(tt_output, shape=input_shape), rtol=0.1, atol=0.1)
    assert passing, out


# On Blackhole gamma_grad is wrong when it reduces over both a batch dim and H, so every backward shape reduces
# gamma_grad/beta_grad over H only or over the batch only.
@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape_normalized_dims, elementwise_affine",
    [
        # Both grads requested, so the input_grad and gamma_beta_grad factories both run.
        (([20, 30], 1), True),
        (([20, 30], 2), True),
        (([6, 64, 64], 2), True),
        # No gamma/beta: input_grad without gamma, and the gamma_beta_grad factory is skipped.
        (([20, 30], 1), False),
    ],
    ids=["lastdim_unaligned", "hw_unaligned", "hw_aligned_batch", "no_affine"],
)
def test_moreh_layer_norm_backward(input_shape_normalized_dims, elementwise_affine, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward(
        input_shape_normalized_dims, elementwise_affine, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize("gamma_or_beta", [True, False], ids=["gamma_only", "beta_only"])
def test_moreh_layer_norm_backward_gamma_or_beta(gamma_or_beta, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward_with_gamma_or_beta(
        ([20, 30], 1), gamma_or_beta, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False
    )


@pytest.mark.merge_gate
def test_moreh_layer_norm_backward_fp32_dest_acc(device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward(([20, 30], 1), True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=True)


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.merge_gate
def test_moreh_layer_norm_backward_large_algorithm(device):
    torch.manual_seed(0)
    # 512 tiles per normalized group: too big for L1, so the input_grad factory runs its large (streaming) kernels.
    run_moreh_layer_norm_backward(
        ([1, 2, 500, 1000], 2), True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False
    )


@pytest.mark.merge_gate
def test_moreh_layer_norm_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_layer_norm(([1, 20], 1), True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False)
    num_program_cache_entries = device.num_program_cache_entries()
    # Without this, the equality below would also pass for an op that never caches a program.
    assert num_program_cache_entries > 0
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = to_ttnn(torch.zeros([1, 20]), device=device)
    run_moreh_layer_norm(([1, 20], 1), True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
def test_moreh_layer_norm_backward_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_layer_norm_backward(([20, 30], 1), True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False)
    num_program_cache_entries = device.num_program_cache_entries()
    # Without this, the equality below would also pass for an op that never caches a program.
    assert num_program_cache_entries > 0
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = to_ttnn(torch.zeros([20, 30]), device=device)
    run_moreh_layer_norm_backward(([20, 30], 1), True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False)
    assert device.num_program_cache_entries() == num_program_cache_entries
