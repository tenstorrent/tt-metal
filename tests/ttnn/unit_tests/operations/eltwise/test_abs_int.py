# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import pytest
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

random.seed(0)


@pytest.mark.parametrize(
    "input_shape",
    ((torch.Size([15, 1])),),
)
def test_abs(input_shape, device):
    torch.manual_seed(0)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), ttnn.int32
    )(input_shape)

    golden_function = ttnn.get_golden_function(ttnn.abs)
    torch_output_tensor = golden_function(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    start_time = start_measuring_time()
    result = ttnn.abs(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    print(torch_input_tensor_a)
    print(torch_output_tensor)
    print(output_tensor)

    print(e2e_perf)

    status, value = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    print("PCC::", value)
    assert status
