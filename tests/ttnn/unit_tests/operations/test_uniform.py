# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import numpy as np
import ttnn
from loguru import logger
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_cpu,
    to_npu,
)
from models.utility_functions import skip_for_grayskull


def test_torch_uniform_bfloat16():
    input = torch.zeros(10, 10, dtype=torch.bfloat16).uniform_(2.1, 2.11)
    print(input)


def get_lib_dtype(lib, dtype):
    """Maps dtype to corresponding library dtype."""
    dtype_map = {
        "bfloat16": lib.bfloat16,
        "float32": lib.float32,
    }
    return dtype_map.get(dtype, None)


def run_uniform(shape, rand_range, dtype, device, compute_kernel_options=None):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    rand_from, rand_to = rand_range[0], rand_range[1]
    cpu_input = torch.ones(shape, dtype=get_lib_dtype(torch, dtype))
    npu_input = to_npu(cpu_input, device, npu_dtype=get_lib_dtype(ttnn, dtype))

    # Exec uniform operation on ttnn using NPU
    ttnn.uniform(npu_input, rand_from, rand_to, compute_kernel_config=compute_kernel_config)
    tt_input = to_cpu(npu_input, shape)

    hash_mp = {}
    for element in tt_input.flatten():
        hash_mp[element.item()] = hash_mp.get(element.item(), 0) + 1

    expected_mean, expected_var = (rand_from + rand_to) / 2, pow(rand_to - rand_from, 2) / 12
    npu_mean, npu_var = torch.mean(tt_input).item(), torch.var(tt_input).item()
    min_val, max_val = torch.min(tt_input).item(), torch.max(tt_input).item()

    logger.info(f"Distinct elements: {len(list(hash_mp.keys()))}")
    logger.info(f"Expected mean: {expected_mean}, NPU mean: {npu_mean}")
    logger.info(f"Expected var: {expected_var}, NPU var: {npu_var}")

    bfloat16_ep = 0.015

    """
    Random bfloat16 is converted from random float. As 16 bits are truncated, the generated number might be smaller/bigger than from/to.
    (even torch can't handle that case, check test_torch_uniform_bfloat16() function). I use bfloat16_ep is used to avoid asserting fail.
    """
    if dtype == "bfloat16":
        assert rand_from - bfloat16_ep <= min_val and max_val < rand_to + bfloat16_ep
    else:
        assert rand_from <= min_val and max_val < rand_to
    assert np.allclose(npu_mean, expected_mean, rtol=0.5)
    assert np.allclose(npu_var, expected_var, rtol=0.5)


# fmt: off
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("shape",
    [
        [100, 100],
        [1, 512, 2, 256],
        [400, 400],
    ],
)
@pytest.mark.parametrize("rand_range",
    [
        [2.1, 9],
        [-5.1, 1.2] # negative float range
    ]
)
@pytest.mark.parametrize("dtype",
    [
        "bfloat16",
        "float32"
    ]
)
# fmt: on
def test_uniform(shape, rand_range, dtype, device):
    torch.manual_seed(0)
    run_uniform(shape, rand_range, dtype, device)


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "shape",
    [[2, 32, 32, 16]],
)
@pytest.mark.parametrize("rand_range", [[-3, 4]])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_uniform_callback(shape, rand_range, dtype, device, use_program_cache):
    torch.manual_seed(0)
    for i in range(2):
        run_uniform(shape, rand_range, dtype, device)
        # Add dummy tensor to make sure that created tensor in 2 iteration don't share the same addr
        tt_dummy_tensor = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "shape",
    [[512, 512], [5, 2, 4, 70, 40]],
)
@pytest.mark.parametrize("rand_range", [[0, 1]])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_uniform_with_compute_kernel_options(shape, rand_range, dtype, device, compute_kernel_options):
    torch.manual_seed(0)
    run_uniform(shape, rand_range, dtype, device, compute_kernel_options)
