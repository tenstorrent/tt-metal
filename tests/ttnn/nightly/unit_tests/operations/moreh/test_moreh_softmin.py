# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest
from models.common.utility_functions import comp_allclose_and_pcc
from loguru import logger
import torch.nn.functional as F
from models.common.utility_functions import is_wormhole_b0

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)

# Module-scoped device: opens once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device


def get_torch_dtype(dtype):
    if dtype == ttnn.int32:
        return torch.int32
    elif dtype == ttnn.float32:
        return torch.float32
    else:
        return torch.bfloat16


def run_moreh_softmin_test(
    shape,
    dim,
    ttnn_dtype,
    layout,
    device,
    rtol,
    atol,
    use_randint,
    use_optional_output_tensor=False,
    strategy=None,
    compute_kernel_options=None,
):
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip(f"bfloat8_b is not supported")
    torch_dtype = get_torch_dtype(ttnn_dtype)
    if use_randint == True:
        torch_input = torch.randint(low=0, high=4, size=shape).to(torch_dtype) + 100
    else:
        torch_input = torch.rand(size=shape, dtype=torch_dtype) + 100
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=layout, device=device)

    torch_output = F.softmin(torch_input, dim)

    if use_optional_output_tensor == True:
        optional_output = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=layout, device=device)
        ttnn_output = ttnn.operations.moreh.softmin(ttnn_input, dim, output_tensor=optional_output)
    elif compute_kernel_options is not None:
        compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
        if strategy is None:
            ttnn_output = ttnn.operations.moreh.softmin(ttnn_input, dim, compute_kernel_config=compute_kernel_config)
        else:
            ttnn_output = ttnn.operations.moreh.softmin(
                ttnn_input, dim, compute_kernel_config=compute_kernel_config, strategy=strategy
            )
    else:
        ttnn_output = ttnn.operations.moreh.softmin(ttnn_input, dim, strategy=strategy)

    ttnn_output = ttnn.to_torch(ttnn_output).to(torch_dtype)

    assert list(ttnn_output.shape) == list(torch_output.shape)

    passing, out = comp_allclose_and_pcc(torch_output, ttnn_output, rtol=rtol, atol=atol)
    logger.debug(out)
    assert passing


def run_moreh_softmin_backward_test(
    shape,
    dim,
    ttnn_dtype,
    layout,
    device,
    rtol,
    atol,
    use_randint,
    use_optional_output_tensor=False,
    strategy=None,
    compute_kernel_options=None,
):
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip(f"bfloat8_b is not supported")
    torch_dtype = get_torch_dtype(ttnn_dtype)
    if use_randint == True:
        torch_x = torch.randint(low=0, high=4, size=shape).to(torch_dtype).requires_grad_(True)
        torch_dy = torch.randint(low=0, high=4, size=shape).to(torch_dtype)
    else:
        torch_x = torch.rand(size=shape, dtype=torch_dtype).requires_grad_(True)
        torch_dy = torch.rand(size=shape, dtype=torch_dtype)

    torch_y = F.softmin(torch_x, dim)

    ttnn_y = ttnn.from_torch(torch_y, dtype=ttnn_dtype, layout=layout, device=device)
    ttnn_dy = ttnn.from_torch(torch_dy, dtype=ttnn_dtype, layout=layout, device=device)

    torch_y.backward(torch_dy)

    if use_optional_output_tensor == True:
        optional_output = ttnn.from_torch(torch_dy, dtype=ttnn_dtype, layout=layout, device=device)
        ttnn_output = ttnn.operations.moreh.softmin_backward(ttnn_y, ttnn_dy, dim, input_grad_tensor=optional_output)
    elif compute_kernel_options is not None:
        compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
        if strategy is None:
            ttnn_output = ttnn.operations.moreh.softmin_backward(
                ttnn_y, ttnn_dy, dim, compute_kernel_config=compute_kernel_config
            )
        else:
            ttnn_output = ttnn.operations.moreh.softmin_backward(
                ttnn_y, ttnn_dy, dim, compute_kernel_config=compute_kernel_config, strategy=strategy
            )
    else:
        ttnn_output = ttnn.operations.moreh.softmin_backward(ttnn_y, ttnn_dy, dim, strategy=strategy)

    ttnn_output = ttnn.to_torch(ttnn_output).to(torch_dtype)

    assert list(ttnn_output.shape) == list(torch_x.grad.shape)

    passing, out = comp_allclose_and_pcc(torch_x.grad, ttnn_output, rtol=rtol, atol=atol)
    logger.debug(out)
    assert passing


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[32, 32], 1],  # single tile
        [[3, 32, 32 * 5], 2],  # mutiple tile with dim W
        [[5, 6, 32, 32], 3],  # multiple cores
        [[10, 20, 32 * 3, 32 * 5], 3],  # multiple tiles per core
        [[32, 32], 0],  # single tile
        [[3, 32 * 5, 32], 1],  # mutiple tile with dim H
        [[5, 6, 32, 32], 2],  # multiple cores
        [[10, 20, 32 * 3, 32 * 5], 2],  # multiple tiles per core
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_softmin_for_dim_hw(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.05
    run_moreh_softmin_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[2, 3, 32 * 4, 32 * 5], 3],
        [[2, 3, 32 * 4, 32 * 5], 2],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_softmin_large_algorithm_for_dim_hw(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.05
    run_moreh_softmin_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape, dim, strategy",
    [
        ([1, 1, 32, 32], 3, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_W),
        ([1, 1, 32, 32], 2, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_H),
        ([1, 1, 32, 64], 3, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_W),
        ([1, 1, 64, 32], 2, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_H),
    ],
)
def test_softmin_large_last_tile_subtracts_max(shape, dim, strategy, device):
    # Force LARGE_*; these shapes would otherwise pick SMALL_*.
    torch_input = torch.empty(shape, dtype=torch.bfloat16)
    if shape[dim] == 32:
        torch_input.fill_(100)
    elif dim == 3:
        torch_input[..., :32] = 103
        torch_input[..., 32:] = 100
    else:
        torch_input[..., :32, :] = 103
        torch_input[..., 32:, :] = 100

    torch_output = F.softmin(torch_input, dim)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.to_torch(ttnn.operations.moreh.softmin(ttnn_input, dim, strategy=strategy)).to(torch.bfloat16)

    # atol=0.05 would accept an all-zero output on the constant (1/32) case.
    passing, out = comp_allclose_and_pcc(torch_output, ttnn_output, rtol=0.02, atol=1e-3)
    logger.debug(out)
    assert passing, out


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[1, 1, 10, 15], 3],  # single tile
        [[1, 1, 10, 32 * 2 + 10], 3],  # mutiple tile with dim
        [[1, 1, 15, 10], 2],  # single tile
        [[1, 1, 32 * 2 + 10, 32], 2],  # mutiple tile with dim
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_softmin_not_multiple_of_32_for_dim_hw(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.05

    run_moreh_softmin_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[1, 15, 32, 32], 1],  # single tile c
        [[1, 15, 32 * 7, 32 * 5], 1],  # mutiple cores
        [[109, 15, 32, 32], 1],  # mutiple tiles per cores
        [[15, 1, 32, 32], 0],  # single tile n
        [[15, 1, 32 * 7, 32 * 5], 0],  # mutiple cores
        [[15, 109, 32 * 2, 32 * 2], 0],  # mutiple tiles per cores
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_softmin_for_dim_nc(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.05

    run_moreh_softmin_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[32, 32], 1],  # single tile
        [[3, 32, 32 * 5], 2],  # mutiple tile with dim W
        [[5, 6, 32, 32], 3],  # multiple cores
        [[10, 20, 32 * 3, 32 * 5], 3],  # multiple tiles per core
        [[32, 32], 0],  # single tile
        [[3, 32 * 5, 32], 1],  # mutiple tile with dim H
        [[5, 6, 32, 32], 2],  # multiple cores
        [[10, 20, 32 * 3, 32 * 5], 2],  # multiple tiles per core
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_softmin_backward_for_dim_hw(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.05

    run_moreh_softmin_backward_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[2, 3, 32 * 4, 32 * 5], 3],
        [[2, 3, 32 * 4, 32 * 5], 2],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_softmin_backward_large_algorithmfor_dim_hw(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)

    rtol = atol = 0.05
    run_moreh_softmin_backward_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[1, 1, 10, 15], 3],  # single tile
        [[1, 1, 10, 32 * 2 + 10], 3],  # mutiple tile with dim
        [[1, 1, 15, 10], 2],  # single tile
        [[1, 1, 32 * 2 + 10, 32], 2],  # mutiple tile with dim
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_softmin_backward_not_multiple_of_32_for_dim_hw(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.05

    run_moreh_softmin_backward_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[1, 15, 32, 32], 1],  # single tile c
        [[1, 15, 32 * 7, 32 * 5], 1],  # mutiple cores
        [[109, 15, 32, 32], 1],  # mutiple tiles per cores
        [[15, 1, 32, 32], 0],  # single tile n
        [[15, 1, 32 * 7, 32 * 5], 0],  # mutiple cores
        [[15, 109, 32 * 2, 32 * 2], 0],  # mutiple tiles per cores
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_softmin_backward_for_dim_nc(shape_dim, dtype, compute_kernel_options, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.05

    run_moreh_softmin_backward_test(
        shape,
        dim,
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        rtol,
        atol,
        True,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[32, 32], 1],
    ],  # single tile
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_softmin_optional_output_tensor(shape_dim, dtype, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.05

    run_moreh_softmin_test(
        shape, dim, dtype, ttnn.TILE_LAYOUT, device, rtol, atol, True, use_optional_output_tensor=True
    )


@pytest.mark.parametrize(
    "shape_dim",
    [
        [[32, 32], 1],
    ],  # single tile
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_softmin_backward_optional_output_tensor(shape_dim, dtype, device):
    shape, dim = shape_dim
    torch.manual_seed(0)
    rtol = atol = 0.05

    run_moreh_softmin_backward_test(
        shape, dim, dtype, ttnn.TILE_LAYOUT, device, rtol, atol, True, use_optional_output_tensor=True
    )


@pytest.mark.parametrize(
    "shape_dim_strategy",
    [
        [[32, 32], 1, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_W],
        [[32, 32], 0, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_H],
        [[2, 3, 32 * 4, 32 * 5], 3, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_W],
        [[2, 3, 32 * 4, 32 * 5], 2, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_H],
        [[1, 15, 32, 32], 1, ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_C],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_softmin_callback(shape_dim_strategy, dtype, device):
    shape, dim, strategy = shape_dim_strategy
    torch.manual_seed(0)
    rtol = atol = 0.05
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests in this file.
    device.clear_program_cache()
    for i in range(2):
        run_moreh_softmin_test(shape, dim, dtype, ttnn.TILE_LAYOUT, device, rtol, atol, True, strategy=strategy)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
        torch_dummy = torch.randn([32, 32])
        tt_dummy = ttnn.from_torch(torch_dummy, device=device)


@pytest.mark.parametrize(
    "shape_dim_strategy",
    [
        [[32, 32], 1, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.SMALL_W],
        [[32, 32], 0, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.SMALL_H],
        [[2, 3, 32 * 4, 32 * 5], 3, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.LARGE_W],
        [[2, 3, 32 * 4, 32 * 5], 2, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.LARGE_H],
        [[1, 15, 32, 32], 1, ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy.LARGE_C],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_softmin_backward_callback(shape_dim_strategy, dtype, device):
    shape, dim, strategy = shape_dim_strategy
    torch.manual_seed(0)
    rtol = atol = 0.05
    num_program_cache_entries = None
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests in this file.
    device.clear_program_cache()
    for i in range(2):
        run_moreh_softmin_backward_test(
            shape, dim, dtype, ttnn.TILE_LAYOUT, device, rtol, atol, True, strategy=strategy
        )
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
        torch_dummy = torch.randn([32, 32])
        tt_dummy = ttnn.from_torch(torch_dummy, device=device)


@pytest.mark.parametrize(
    "shape, dim, strategy",
    [
        pytest.param(
            [32, 32],
            1,
            ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_W,
            id="small-w-single",
        ),
        pytest.param(
            [32, 128],
            1,
            ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_W,
            id="large-w-4tiles",
        ),
        pytest.param(
            [128, 32],
            0,
            ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_H,
            id="large-h-4tiles",
        ),
        pytest.param(
            [1, 6, 32, 32],
            1,
            ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.LARGE_C,
            id="large-c-dim1",
        ),
        pytest.param(
            [1, 1, 32, 33],
            3,
            ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_W,
            id="small-w-unaligned-2tiles",
        ),
        pytest.param(
            [1, 1, 33, 32],
            2,
            ttnn.operations.moreh.SoftmaxOpParallelizationStrategy.SMALL_H,
            id="small-h-unaligned-2tiles",
        ),
    ],
)
@pytest.mark.parametrize("plant_pos", ["first", "last"], ids=["plant-first", "plant-last"])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize(
    "special_value",
    [float("inf"), float("-inf")],
    ids=["plus-inf", "minus-inf"],
)
def test_softmin_special_value_in_row(shape, dim, strategy, plant_pos, compute_kernel_options, special_value, device):
    """The one planted special value must land where the documented device contract says.

    Each case plants exactly one special element, so exactly one reduction group is affected and every other
    group must still match torch. `plant_pos` picks the lane along the reduced dim: `first` is the second
    lane, `last` is the final one -- the masked tail tile for every multi-tile entry here -- so the padding
    and the per-tile staging/folding are exercised with a special value in them, not only with finite data.

    +inf: torch returns a valid distribution with 0 at the +inf position, and the kernel matches it.

    -inf: torch returns NaNs (inf - inf) in the affected reduction group; this kernel instead returns a
    finite distribution concentrated on the (unique) -inf minimum. That divergence is the behaviour
    recorded in #56371 -- a measured kernel result, not an IEEE-equivalent one -- and it is the single
    place where the MIN statistic changes which operands the subtraction sees (the minimum is -inf
    where main's maximum was finite, and the subtraction itself lowers to the FPU, not the SFPU).
    The affected group is therefore graded against the stated contract rather than against torch's NaN.

    See issue #56371.
    """
    torch.manual_seed(0)

    torch_input = torch.rand(size=shape, dtype=torch.bfloat16) + 100
    plant_idx = [0] * len(shape)
    plant_idx[dim] = 1 if plant_pos == "first" else shape[dim] - 1
    torch_input[tuple(plant_idx)] = special_value

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.operations.moreh.softmin(
        ttnn_input,
        dim,
        strategy=strategy,
        compute_kernel_config=get_compute_kernel_options(compute_kernel_options),
    )
    ttnn_output = ttnn.to_torch(ttnn_output).to(torch.bfloat16)

    actual = ttnn_output.float()
    expected = F.softmin(torch_input.float(), dim).to(torch.bfloat16).float()
    assert list(actual.shape) == list(expected.shape)

    if special_value == float("-inf"):
        # Documented device contract: the affected group is one-hot on the planted -inf minimum.
        # torch instead yields NaNs there, so the group is replaced rather than compared.
        group_idx = [0] * len(shape)
        group_idx[dim] = slice(None)
        expected[tuple(group_idx)] = 0.0
        expected[tuple(plant_idx)] = 1.0

    assert torch.isfinite(actual).all()
    assert (actual >= 0).all()

    # The unaffected groups are still torch's rows, so a uniform or mislocated distribution fails here.
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=1e-3)
    # Normalisation over the reduced dim, independent of where the mass sits in the affected group.
    torch.testing.assert_close(
        actual.sum(dim=dim),
        torch.ones_like(actual.sum(dim=dim)),
        rtol=0,
        atol=0.02,
    )

    if special_value == float("inf"):
        torch.testing.assert_close(
            actual[tuple(plant_idx)],
            torch.zeros_like(actual[tuple(plant_idx)]),
            rtol=0,
            atol=1e-3,
        )
    else:
        # "Concentrated on the planted minimum" is the contract, and the elementwise tolerances above
        # would also accept only 0.98 there with the remaining 0.02 spread over the group, so check the
        # concentration itself.
        assert actual[tuple(plant_idx)] >= 0.9
