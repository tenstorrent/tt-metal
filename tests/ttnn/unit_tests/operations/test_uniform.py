# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_cpu,
    to_npu,
)


def run_uniform(shape, rand_from, rand_to, device):
    cpu_input = torch.ones(shape, dtype=torch.float)
    npu_input = to_npu(cpu_input, device, npu_dtype=ttnn.float32)

    # Exec uniform operation on torch using CPU
    cpu_input.uniform_(rand_from, rand_to)

    # Exec uniform operation on ttnn using NPU
    ttnn.uniform(npu_input, rand_from, rand_to)
    tt_input = to_cpu(npu_input, shape)

    print("CPU: ", cpu_input)
    print("NPU: ", tt_input)

    hash_mp = {}
    for element in tt_input.flatten():
        hash_mp[element.item()] = hash_mp.get(element.item(), 0) + 1
    print(len(list(hash_mp.keys())))
    print(list(hash_mp.values()))


@pytest.mark.parametrize("shape", [[10, 15]])
@pytest.mark.parametrize("rand_from", [3])
@pytest.mark.parametrize("rand_to", [5])
def test_uniform(shape, rand_from, rand_to, device):
    torch.manual_seed(0)
    run_uniform(shape, rand_from, rand_to, device)
