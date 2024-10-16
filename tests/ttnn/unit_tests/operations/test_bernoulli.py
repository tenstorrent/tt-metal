# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np
from loguru import logger
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)
from models.utility_functions import skip_for_grayskull
from collections import Counter


def get_lib_dtype(lib, dtype):
    """Maps dtype to corresponding library dtype."""
    dtype_map = {
        "bfloat16": lib.bfloat16,
        "float32": lib.float32,
    }
    return dtype_map.get(dtype, None)


def run_bernoulli(shape, in_dtype, out_dtype, device, compute_kernel_options=None):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    cpu_input = torch.rand(shape, dtype=get_lib_dtype(torch, in_dtype))
    npu_input = ttnn.from_torch(cpu_input, device=device, dtype=get_lib_dtype(ttnn, in_dtype), layout=ttnn.TILE_LAYOUT)

    one_probs = []
    for i in range(10):
        npu_output = ttnn.bernoulli(
            npu_input, out_dtype=get_lib_dtype(ttnn, out_dtype), compute_kernel_config=compute_kernel_config
        )
        tt_output = ttnn.to_torch(ttnn.to_layout(npu_output, ttnn.ROW_MAJOR_LAYOUT)).reshape(shape)
        tt_output_list = tt_output.flatten().tolist()

        c = Counter(tt_output_list)
        print(c)
        one_probs.append(c[1] / len(tt_output_list))
    print(one_probs)
    expected_one_prob = 0.5

    assert np.allclose(expected_one_prob, np.mean(one_probs), rtol=0.1)


# fmt: off
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("shape",
    [
        [3, 4],
        # [1892],
        # [1, 512, 2, 256],
        # [500, 500]
    ],
)
@pytest.mark.parametrize("in_dtype",
    [
        "bfloat16",
        # "float32"
    ]
)
@pytest.mark.parametrize("out_dtype",
    [
        "bfloat16",
        "float32"
    ]
)
# fmt: on
def test_bernoulli(shape, in_dtype, out_dtype, device):
    torch.manual_seed(0)
    run_bernoulli(shape, in_dtype, out_dtype, device)


# @skip_for_grayskull("Requires wormhole_b0 to run")
# @pytest.mark.parametrize(
#     "shape",
#     [
#         [1, 21, 123, 23],
#     ],
# )
# @pytest.mark.parametrize("in_dtype", ["float32"])
# @pytest.mark.parametrize("out_dtype", ["float32"])
# def test_bernoulli_callback(shape, in_dtype, out_dtype, device, use_program_cache):
#     torch.manual_seed(0)
#     for i in range(2):
#         run_bernoulli(shape, in_dtype, out_dtype, device)
#         # Add dummy tensor to make sure that created tensor in 2 iteration don't share the same addr
#         tt_dummy_tensor = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
#         if i == 0:
#             num_program_cache_entries = device.num_program_cache_entries()
#             assert num_program_cache_entries > 0
#         else:
#             assert device.num_program_cache_entries() == num_program_cache_entries


# @skip_for_grayskull("Requires wormhole_b0 to run")
# @pytest.mark.parametrize(
#     "shape",
#     [[512, 512], [5, 2, 4, 70, 40]],
# )
# @pytest.mark.parametrize("in_dtype", ["float32"])
# @pytest.mark.parametrize("out_dtype", ["float32"])
# @pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
# def test_uniform_with_compute_kernel_options(shape, in_dtype, out_dtype, device, compute_kernel_options):
#     torch.manual_seed(0)
#     run_bernoulli(shape, in_dtype, out_dtype, device, compute_kernel_options)
