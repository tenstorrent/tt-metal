# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_lib_dtype,
)
from loguru import logger


@pytest.mark.parametrize("shape", [[1, 512, 2, 256]])
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("out_dtype", ["bfloat16", "float32"])
def test_bernoulli_p_zero(shape, in_dtype, out_dtype, device):
    """
    Unit test for bernoulli with p=0. Output must be all zeros.
    """
    cpu_input = torch.zeros(shape, dtype=get_lib_dtype(torch, in_dtype))
    npu_input = ttnn.from_torch(cpu_input, device=device, dtype=get_lib_dtype(ttnn, in_dtype), layout=ttnn.TILE_LAYOUT)

    npu_output = ttnn.bernoulli(npu_input, dtype=get_lib_dtype(ttnn, out_dtype))
    tt_output = ttnn.to_torch(npu_output).reshape(shape)

    assert torch.all(tt_output == 0), "Output must be all zeros when p=0"


@pytest.mark.parametrize("shape", [[1, 512, 2, 256]])
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("out_dtype", ["bfloat16", "float32"])
def test_bernoulli_p_one(shape, in_dtype, out_dtype, device):
    """
    Unit test for bernoulli with p=1. Output must be all ones.
    """
    cpu_input = torch.ones(shape, dtype=get_lib_dtype(torch, in_dtype))
    npu_input = ttnn.from_torch(cpu_input, device=device, dtype=get_lib_dtype(ttnn, in_dtype), layout=ttnn.TILE_LAYOUT)

    npu_output = ttnn.bernoulli(npu_input, dtype=get_lib_dtype(ttnn, out_dtype))
    tt_output = ttnn.to_torch(npu_output).reshape(shape)

    assert torch.all(tt_output == 1), "Output must be all ones when p=1"


@pytest.mark.parametrize(
    "shape",
    [
        [2003],
        [500, 500],
        [1, 512, 2, 256],
    ],
)
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("out_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("is_out_alloc", [True, False])
def test_bernoulli_api_contract(shape, in_dtype, out_dtype, device, is_out_alloc):
    """
    Unit test for bernoulli API contract.
    Checks shape, dtype, and that output contains only 0s or 1s.
    """
    p_value = 0.5
    seed = 0  # Use a non-deterministic value for a general case
    cpu_input = torch.full(shape, p_value, dtype=get_lib_dtype(torch, in_dtype))
    npu_input = ttnn.from_torch(cpu_input, device=device, dtype=get_lib_dtype(ttnn, in_dtype), layout=ttnn.TILE_LAYOUT)

    npu_output = None
    if is_out_alloc:
        # Create an output tensor to be written into
        npu_output = ttnn.empty(shape, device=device, dtype=get_lib_dtype(ttnn, out_dtype), layout=ttnn.TILE_LAYOUT)
        ttnn.bernoulli(npu_input, seed, output=npu_output, dtype=get_lib_dtype(ttnn, out_dtype))
    else:
        npu_output = ttnn.bernoulli(npu_input, seed, dtype=get_lib_dtype(ttnn, out_dtype))

    # Check output properties
    assert list(npu_output.shape) == shape
    assert npu_output.layout == ttnn.TILE_LAYOUT
    assert npu_output.dtype == get_lib_dtype(ttnn, out_dtype)

    # Check output content
    tt_output = ttnn.to_torch(npu_output).reshape(shape)
    # Verify that all elements are either 0 or 1
    assert torch.all((tt_output == 0) | (tt_output == 1)), "Output tensor must only contain 0s and 1s"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 21, 123, 24],
    ],
)
@pytest.mark.parametrize("seed", [1408])
@pytest.mark.parametrize("in_dtype", ["float32"])
@pytest.mark.parametrize("out_dtype", ["float32"])
def test_bernoulli_callback(shape, seed, in_dtype, out_dtype, device):
    """
    Tests program cache hits for the bernoulli operation on identical (same-seed) calls.

    NOTE: seed is now part of compute_program_hash (descriptor framework can't re-apply
    non-Buffer RTAs on cache hit), so calls with different seeds each produce a new
    cache entry. Same-seed calls still hit cache, which is what this test verifies.
    """
    num_program_cache_entries_list = []
    for i in range(2):
        cpu_input = torch.full(shape, 0.5, dtype=get_lib_dtype(torch, in_dtype))
        npu_input = ttnn.from_torch(
            cpu_input, device=device, dtype=get_lib_dtype(ttnn, in_dtype), layout=ttnn.TILE_LAYOUT
        )
        # Same seed each iteration → must cache-hit on the second call.
        ttnn.bernoulli(npu_input, seed, dtype=get_lib_dtype(ttnn, out_dtype))
        # Add dummy tensor to make sure that created tensor in 2nd iteration doesn't share the same addr
        ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@pytest.mark.parametrize(
    "shape",
    [[512, 512], [5, 8, 70, 40]],
)
@pytest.mark.parametrize("in_dtype", ["float32"])
@pytest.mark.parametrize("out_dtype", ["float32"])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_bernoulli_with_compute_kernel_options(shape, in_dtype, out_dtype, device, compute_kernel_options):
    """
    Tests that bernoulli runs correctly with different compute kernel options.
    """
    p_value = 0.5
    seed = 0
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    if compute_kernel_config is None:
        pytest.skip("Kernel option is not available")

    cpu_input = torch.full(shape, p_value, dtype=get_lib_dtype(torch, in_dtype))
    npu_input = ttnn.from_torch(cpu_input, device=device, dtype=get_lib_dtype(ttnn, in_dtype), layout=ttnn.TILE_LAYOUT)

    # Just ensure the operation runs without error and returns a valid tensor.
    npu_output = ttnn.bernoulli(
        npu_input, seed, dtype=get_lib_dtype(ttnn, out_dtype), compute_kernel_config=compute_kernel_config
    )
    assert list(npu_output.shape) == shape


def test_bernoulli_seed_distinguishes_cache_entries(device):
    """Regression: descriptor framework only re-patches Buffer* slots on cache hit.
    `seed` is a non-Buffer RTA scalar; if it's omitted from compute_program_hash,
    two same-shape calls with different seeds collide on the cached program and
    the second uses the first call's seed. Asserts both the observable outcome
    (different seed -> different output) and the underlying mechanism
    (different seed -> distinct cache entry)."""
    import torch

    shape = [1, 32, 64]
    cpu_input = torch.full(shape, 0.5, dtype=torch.float32)
    npu_input = ttnn.from_torch(cpu_input, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    device.disable_and_clear_program_cache()
    device.enable_program_cache()

    out_a = ttnn.to_torch(ttnn.bernoulli(npu_input, seed=1234, dtype=ttnn.float32))
    cache_after_a = device.num_program_cache_entries()

    ttnn.bernoulli(npu_input, seed=1234, dtype=ttnn.float32)
    cache_after_b = device.num_program_cache_entries()

    out_c = ttnn.to_torch(ttnn.bernoulli(npu_input, seed=5678, dtype=ttnn.float32))
    cache_after_c = device.num_program_cache_entries()

    assert cache_after_b == cache_after_a, "same seed must reuse the cached program"
    assert cache_after_c > cache_after_b, (
        "different seed must produce a distinct cache entry — otherwise the second call "
        "silently reuses the first call's seed (cache-collision bug)"
    )
    assert not torch.equal(out_a, out_c), "different seeds must produce different outputs"
