# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import pytest
import numpy as np
import ttnn
from scipy.stats import kstest
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_cpu,
    to_npu,
)


def get_lib_dtype(lib, dtype):
    """Maps dtype to corresponding library dtype."""
    dtype_map = {
        "bfloat16": lib.bfloat16,
        "float32": lib.float32,
    }
    return dtype_map.get(dtype, None)


def run_uniform(shape, rand_range, dtype, device):
    rand_from, rand_to = rand_range[0], rand_range[1]
    cpu_input = torch.ones(shape, dtype=get_lib_dtype(torch, dtype))
    npu_input = to_npu(cpu_input, device, npu_dtype=get_lib_dtype(ttnn, dtype))

    # Exec uniform operation on torch using CPU
    cpu_input.uniform_(rand_from, rand_to)

    # Exec uniform operation on ttnn using NPU
    ttnn.uniform(npu_input, rand_from, rand_to)
    tt_input = to_cpu(npu_input, shape)

    print("Dtype: ", dtype)
    print("CPU: ", cpu_input)
    print("NPU: ", tt_input)

    hash_mp = {}
    for element in tt_input.flatten():
        hash_mp[element.item()] = hash_mp.get(element.item(), 0) + 1
    print(len(list(hash_mp.keys())))

    # KS Test to compare distributions
    ks_statistic, p_value = kstest(cpu_input.flatten(), tt_input.flatten())
    expected_mean, expected_var = (rand_from + rand_to) / 2, pow(rand_to - rand_from, 2) / 12
    # torch.mean(cpu_input), torch.var(cpu_input)
    actual_mean, actual_var = torch.mean(tt_input).item(), torch.var(tt_input).item()
    # print(, )
    print(f"Expected mean: {expected_mean}, Actual mean: {actual_mean}")
    print(f"Expected var: {expected_var}, Actual var: {actual_var}")
    # assert abs(expected_mean - actual_mean) < 0.5
    # assert abs(expected_var - actual_var) < 0.5
    assert np.allclose(actual_mean, expected_mean, rtol=0.05)
    assert np.allclose(actual_var, expected_var, rtol=0.05)
    # print(f"KS Statistic: {ks_statistic}, P-Value: {p_value}")


@pytest.mark.parametrize(
    "shape",
    [
        # [3, 5],  # abcccccccccccccccccccccc
        # [64, 32],
        [100, 64],
        # [128, 64],  # abcccccccccccccccccccccc
        # [128, 128],
        [1, 512, 2, 256],
        [1000, 120],  # abcccccccccccccccccccccc
    ],
)
@pytest.mark.parametrize("rand_range", [[3, 5], [-5, 1]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_uniform(shape, rand_range, dtype, device):
    torch.manual_seed(0)
    run_uniform(shape, rand_range, dtype, device)
