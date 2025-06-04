# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
import pytest
import torch
import random
import ttnn
from tests.sweep_framework.framework.permutations import *
from tests.sweep_framework.sweep_utils.utils import gen_shapes, sanitize_shape
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_topk_simmilarity
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 32, 64], [2, 6, 128, 128], [1, 1, 32, 64], 32)
        + gen_shapes([1, 32, 64], [12, 256, 1024], [1, 32, 64], 8)
        + gen_shapes([32, 64], [256, 1024], [32, 64], 8),
        "dim": [0, 1, 2, 3, -1, -2, -3, -4, None],
        "largest": [True],
        "k": [32],  # only k = 32 is supported for now
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "xfail": {
        "input_shape": gen_shapes([1, 1, 32, 64], [6, 12, 256, 1024], [1, 1, 32, 64], 32)
        + gen_shapes([1, 32, 64], [12, 256, 1024], [1, 32, 64], 8)
        + gen_shapes([32, 64], [256, 1024], [32, 64], 8),
        "dim": [0, 1, 2, 3, -1, -2, -3, -4, None],
        "largest": [False],
        "k": [32],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if len(test_vector["input_shape"]) != 4:
        return True, "Input shape must be 4D"
    if test_vector["dim"] != -1:
        return True, "Only the last dim is supported right now"
    if test_vector["dim"] is not None:
        if (
            test_vector["dim"] * (-1) > len(test_vector["input_shape"])
            and test_vector["dim"] < 0
            or test_vector["dim"] > (len(test_vector["input_shape"]) - 1)
            and test_vector["dim"] >= 0
        ):
            return True, "Absolute value of dim must be less or equal than the rank of input tensor"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Unary operation requires tensor to be in Tile layout when working with non-sharded input tensor"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_a_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b is only supported on tiled layout"
    return False, None


def run_topk(
    input_shape, dim, largest, k, input_a_dtype, input_layout, input_a_memory_config, output_memory_config, device
):
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    input_shape = sanitize_shape(input_shape, "topk")

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    output_indices_shape = input_shape.copy()
    output_indices_shape[dim] = k

    torch_optional_output = [
        gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
            output_indices_shape
        ),
        torch.randint(-1, 1, output_indices_shape, dtype=torch.int16),
    ]
    torch_output_values, torch_output_indices = torch.topk(
        torch_input_tensor_a, k, dim=dim, largest=largest, sorted=True
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_a_memory_config,
    )
    op_output_values = ttnn.from_torch(
        torch_optional_output[0],
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=output_memory_config,
    )
    op_output_indices = ttnn.from_torch(
        torch_optional_output[1],
        dtype=ttnn.uint16,
        layout=input_layout,
        device=device,
        memory_config=output_memory_config,
    )

    start_time = start_measuring_time()
    ttnn.topk(input_tensor_a, k=k, dim=dim, largest=largest, sorted=True, out=(op_output_values, op_output_indices))
    output_values = ttnn.to_torch(op_output_values)
    output_indices = ttnn.to_torch(op_output_indices).to(torch.int64)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.999
    tensors = [input_tensor_a, op_output_values, op_output_indices]
    return get_run_return(torch_output_values, output_values, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize("params", list(permutations(parameters["nightly"])))
def test_nightly(device, params):
    invalidated, output_str = invalidate_vector(params)

    if invalidated:
        pytest.skip(output_str)

    res, _ = run_topk(**params, device=device)

    assert res[0], res[1]


@pytest.mark.xfail
@pytest.mark.parametrize("params", list(permutations(parameters["xfail"])))
def test_nightly(device, params):
    invalidated, output_str = invalidate_vector(params)

    if invalidated:
        pytest.skip(output_str)

    res, _ = run_topk(**params, device=device)

    assert res[0], res[1]


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    dim,
    largest,
    k,
    input_a_dtype,
    input_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    return run_topk(
        input_shape,
        dim,
        largest,
        k,
        input_a_dtype,
        input_layout,
        input_a_memory_config,
        output_memory_config,
        device,
    )
