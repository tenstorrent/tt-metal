# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_lib_dtype,
)
from models.utility_functions import skip_for_grayskull, skip_for_blackhole
from collections import Counter
from loguru import logger


# Due to the issue with tensix instruction to generated pseudo-random numbers: #13904, the seed is temporarily fixed to make the test result consistent.
def run_bernoulli(shape, in_dtype, out_dtype, device, is_out_alloc=False, compute_kernel_options=None, p_value=0.5):
    # Bernoulli operation is a comparison operation between input and random generated number.
    # RNG is expected to be from a uniform distribution but due to the issue with tensix instruction to generated pseudo-random numbers: #13904
    # it is not perfectly uniform.
    k = 10  # number of bernoulli operations to run on k tensors to get a normal distribution approximation
    seed = 0  # randomize number generation sequence for every bernoulli operation
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    # set input to fixed p value to test reliability of bernoulli sampler
    cpu_input = torch.empty(shape, dtype=get_lib_dtype(torch, in_dtype))
    cpu_input.fill_(p_value)

    npu_input = ttnn.from_torch(cpu_input, device=device, dtype=get_lib_dtype(ttnn, in_dtype), layout=ttnn.TILE_LAYOUT)

    npu_output = None
    if is_out_alloc:
        cpu_output = torch.rand(shape, dtype=get_lib_dtype(torch, out_dtype))
        npu_output = ttnn.from_torch(
            cpu_output, device=device, dtype=get_lib_dtype(ttnn, out_dtype), layout=ttnn.TILE_LAYOUT
        )
    one_probs = []
    for i in range(k):
        if is_out_alloc:
            ttnn.bernoulli(
                npu_input,
                seed,
                output=npu_output,
                dtype=get_lib_dtype(ttnn, out_dtype),
                compute_kernel_config=compute_kernel_config,
            )
        else:
            npu_output = ttnn.bernoulli(
                npu_input,
                seed,
                dtype=get_lib_dtype(ttnn, out_dtype),
                compute_kernel_config=compute_kernel_config,
            )
        tt_output = ttnn.to_torch(npu_output).reshape(shape)
        tt_output_list = tt_output.flatten().tolist()
        c = Counter(tt_output_list)
        # one_probs is proportion of 1 in tt_output_list
        one_probs.append(c[1] / len(tt_output_list))

    estimated_one_prob = np.mean(one_probs)
    expected_one_prob = p_value
    logger.info(f"estimated_one_prob={estimated_one_prob}, expected_one_prob={expected_one_prob}")
    standard_error = np.sqrt(estimated_one_prob * (1 - estimated_one_prob) / (len(tt_output_list)))
    confidence_interval = 2.57 * standard_error  # 99% confidence interval
    logger.info(f"confidence_interval={confidence_interval}")
    assert expected_one_prob - confidence_interval < estimated_one_prob < expected_one_prob + confidence_interval


@skip_for_grayskull("Requires wormhole_b0 to run")
# @skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("p_value", [0.5])
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
def test_bernoulli(shape, in_dtype, out_dtype, device, is_out_alloc, p_value):
    # torch.manual_seed(seed)
    run_bernoulli(shape, in_dtype, out_dtype, device, is_out_alloc=is_out_alloc, p_value=p_value)


@skip_for_grayskull("Requires wormhole_b0 to run")
# @skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "shape",
    [
        [1, 21, 123, 24],
    ],
)

# TODO: Re-enable this test when we return to fixed seed testing
# @pytest.mark.parametrize("seed", [1408])
# @pytest.mark.parametrize("in_dtype", ["float32"])
# @pytest.mark.parametrize("out_dtype", ["float32"])
# @pytest.mark.parametrize("is_out_alloc", [True, False])
# def test_bernoulli_callback(shape, seed, in_dtype, out_dtype, device, is_out_alloc):
#     num_program_cache_entries_list = []
#     for _ in range(2):
#         run_bernoulli(shape, in_dtype, out_dtype, device, is_out_alloc=is_out_alloc)
#         # Add dummy tensor to make sure that created tensor in 2 iteration don't share the same addr
#         tt_dummy_tensor = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
#         num_program_cache_entries_list.append(device.num_program_cache_entries())
#         # Cache must hit when we change seed and seed runtime arg is overrode
#         seed = seed + 1

#     logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
#     assert num_program_cache_entries_list[0] > 0
#     assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@skip_for_grayskull("Requires wormhole_b0 to run")
# @skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "shape",
    [[512, 512], [5, 8, 70, 40]],
)
@pytest.mark.parametrize("in_dtype", ["float32"])
@pytest.mark.parametrize("out_dtype", ["float32"])
@pytest.mark.parametrize("seed", [1408])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_bernoulli_with_compute_kernel_options(shape, seed, in_dtype, out_dtype, device, compute_kernel_options):
    run_bernoulli(shape, in_dtype, out_dtype, device, compute_kernel_options=compute_kernel_options)
