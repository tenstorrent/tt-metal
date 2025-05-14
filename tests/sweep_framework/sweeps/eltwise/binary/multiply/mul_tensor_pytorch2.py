# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt, gen_constant

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Ref: https://github.com/tenstorrent/pytorch2.0_ttnn/blob/main/docs/operations/aten.mul.Tensor.md

# Test to check for cases with inputs that give inf in pytorch and not in TTNN because 'inf' threshold is different for Torch and TTNN. Hence, this is tested separately.
# pytorch gives 'inf' for values beyond ±3.4e38 but in TTNN, we get inf when the value exceeds ±3.41e38
parameters = {
    "check_inf_cases": {
        "input_shape": [
            {"self": [1, 1, 1, 10], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 12], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 14], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 15], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 17], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 1], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 201], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 2048], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 256], "other": -3.3895313892515355e38},
            {"self": [1, 1, 1, 25], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 2], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 5], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 6], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 7], "other": -3.3895313892515355e38},
            {"self": [1, 1, 1, 8], "other": -3.3895313892515355e38},
            {"self": [1, 1, 1, 9], "other": -3.4028234663852886e38},
        ],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    torch_input_tensor_a = gen_constant(input_shape["self"], 1.0)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.mul(input_tensor_a, input_shape["other"])
    e2e_perf = stop_measuring_time(start_time)
    expected_result = ttnn.full(
        input_shape["self"], fill_value=input_shape["other"], dtype=input_a_dtype, layout=input_a_layout, device=device
    )

    check = ttnn.eq(expected_result, result)
    check_one_tensor = ttnn.to_torch(check)
    check_one_result = torch.all(check_one_tensor == 1.0).item()

    torch_input_tensor_a = gen_constant(input_shape["self"], -1.0)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.mul(input_tensor_a, input_shape["other"])
    e2e_perf = stop_measuring_time(start_time)
    expected_result = ttnn.full(
        input_shape["self"],
        fill_value=-1 * input_shape["other"],
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
    )
    check = ttnn.eq(expected_result, result)
    check_m_one_tensor = ttnn.to_torch(check)
    check_m_one_result = torch.all(check_m_one_tensor == 1.0).item()

    status = check_one_result and check_m_one_result
    return [(status, "1.0" if status else "0.0"), e2e_perf]


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": [
            {"self": [0], "other": 0.5},
            {"self": [1, 1, 1, 10], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 12], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 14], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 15], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 17], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 1], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 201], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 2048], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 256], "other": -3.3895313892515355e38},
            {"self": [1, 1, 1, 25], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 2], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 42], "other": -0.75},
            {"self": [1, 1, 1, 42], "other": 1.25},
            {"self": [1, 1, 1, 42], "other": 1.9761904761904763},
            {"self": [1, 1, 1, 5], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 6], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 7], "other": -3.3895313892515355e38},
            {"self": [1, 1, 1, 8], "other": -3.3895313892515355e38},
            {"self": [1, 1, 1, 9], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1024], "other": 0.03125},
            {"self": [1, 1, 1024], "other": 0.044715},
            {"self": [1, 1, 1024], "other": 0.125},
            {"self": [1, 1, 1024], "other": 0.5},
            {"self": [1, 1, 1024], "other": 0.7978845608028654},
            {"self": [1, 1, 224, 224], "other": 0.448},
            {"self": [1, 1, 224, 224], "other": 0.45},
            {"self": [1, 1, 224, 224], "other": 0.458},
            {"self": [1, 1, 256], "other": 1.0},
            {"self": [1, 1, 3072], "other": 0.044715},
            {"self": [1, 1, 3072], "other": 0.5},
            {"self": [1, 1, 3072], "other": 0.7978845608028654},
            {"self": [1, 1, 32, 1], "other": -0.75},
            {"self": [1, 1, 32, 1], "other": 1.25},
            {"self": [1, 1, 32, 1], "other": 1.5625},
            {"self": [1, 1, 4096], "other": 0.044715},
            {"self": [1, 1, 4096], "other": 0.5},
            {"self": [1, 1, 4096], "other": 0.7978845608028654},
            {"self": [1, 1, 480, 640], "other": 10.0},
            {"self": [1, 1, 512], "other": 0.04419417382415922},
            {"self": [1, 1, 768], "other": 0.03608439182435161},
            {"self": [1, 1, 768], "other": 0.125},
            {"self": [1, 12, 3072], "other": 0.044715},
            {"self": [1, 12, 3072], "other": 0.5},
            {"self": [1, 12, 3072], "other": 0.7978845608028654},
            {"self": [1, 12, 64, 64], "other": 16.0},
            {"self": [1, 14, 3072], "other": 0.044715},
            {"self": [1, 14, 3072], "other": 0.5},
            {"self": [1, 14, 3072], "other": 0.7978845608028654},
            {"self": [1, 15, 1024], "other": 0.044715},
            {"self": [1, 15, 1024], "other": 0.5},
            {"self": [1, 15, 1024], "other": 0.7978845608028654},
            {"self": [1, 16, 64, 64], "other": 16.0},
            {"self": [1, 160], "other": 1.0},
            {"self": [1, 19, 1024], "other": 0.125},
            {"self": [1, 19, 1024], "other": 32.0},
            {"self": [1, 1], "other": 0.0},
            {"self": [1, 1], "other": 16.0},
            {"self": [1, 1], "other": 50258.0},
            {"self": [1, 1], "other": 50259.0},
            {"self": [1, 1], "other": 50359.0},
            {"self": [1, 1], "other": 50363.0},
            {"self": [1, 23, 40], "other": 6.283185307179586},
            {"self": [1, 24, 49, 32], "other": 0.1767766952966369},
            {"self": [1, 24, 64, 64], "other": 16.0},
            {"self": [1, 24, 768], "other": 0.125},
            {"self": [1, 3, 16, 16, 2], "other": 2.0},
            {"self": [1, 3, 32, 32, 2], "other": 2.0},
            {"self": [1, 3, 64, 64, 2], "other": 2.0},
            {"self": [1, 3, 64, 64], "other": 16.0},
            {"self": [1, 32, 49, 32], "other": 0.1767766952966369},
            {"self": [1, 32, 6144], "other": 0.044715},
            {"self": [1, 32, 6144], "other": 0.5},
            {"self": [1, 32, 6144], "other": 0.79788456},
            {"self": [1, 32, 64, 64], "other": 16.0},
            {"self": [1, 4, 64, 64], "other": 16.0},
            {"self": [1, 45, 3072], "other": 0.044715},
            {"self": [1, 45, 3072], "other": 0.5},
            {"self": [1, 45, 3072], "other": 0.7978845608028654},
            {"self": [1, 5, 4096], "other": 0.044715},
            {"self": [1, 5, 4096], "other": 0.5},
            {"self": [1, 5, 4096], "other": 0.7978845608028654},
            {"self": [1, 50, 3072], "other": 1.702},
            {"self": [1, 50, 768], "other": 0.125},
            {"self": [1, 59, 1024], "other": 0.125},
            {"self": [1, 6, 64, 64], "other": 16.0},
            {"self": [1, 7, 3072], "other": 0.044715},
            {"self": [1, 7, 3072], "other": 0.5},
            {"self": [1, 7, 3072], "other": 0.7978845608028654},
            {"self": [1, 8, 64, 64], "other": 16.0},
            {"self": [1, 9, 128], "other": 0.044715},
            {"self": [1, 9, 128], "other": 0.5},
            {"self": [1, 9, 128], "other": 0.7978845608028654},
            {"self": [1, 9, 16384], "other": 0.044715},
            {"self": [1, 9, 16384], "other": 0.5},
            {"self": [1, 9, 16384], "other": 0.7978845608028654},
            {"self": [1, 9, 3072], "other": 0.044715},
            {"self": [1, 9, 3072], "other": 0.5},
            {"self": [1, 9, 3072], "other": 0.7978845608028654},
            {"self": [1, 9, 4096], "other": 0.044715},
            {"self": [1, 9, 4096], "other": 0.5},
            {"self": [1, 9, 4096], "other": 0.7978845608028654},
            {"self": [1, 9, 8192], "other": 0.044715},
            {"self": [1, 9, 8192], "other": 0.5},
            {"self": [1, 9, 8192], "other": 0.7978845608028654},
            {"self": [10, 10], "other": 16.0},
            {"self": [10, 10], "other": 8.0},
            {"self": [100], "other": 0.5},
            {"self": [1066], "other": 0.600375234521576},
            {"self": [120], "other": 0.5},
            {"self": [128], "other": 0.125},
            {"self": [128], "other": 0.25},
            {"self": [128], "other": 0.5},
            {"self": [128], "other": 1.0},
            {"self": [128], "other": 2.0},
            {"self": [12], "other": 32.0},
            {"self": [136], "other": 0.5},
            {"self": [14], "other": 0.5},
            {"self": [15, 15], "other": 16.0},
            {"self": [15, 15], "other": 8.0},
            {"self": [16, 6, 49, 32], "other": 0.1767766952966369},
            {"self": [16, 8, 49, 32], "other": 0.1767766952966369},
            {"self": [160], "other": -9.210340371976184},
            {"self": [160], "other": 0.5},
            {"self": [16], "other": 0.5},
            {"self": [16], "other": 32.0},
            {"self": [17, 17], "other": 16.0},
            {"self": [2, 2], "other": 16.0},
            {"self": [2, 7, 2048], "other": 1.702},
            {"self": [2, 7, 512], "other": 0.125},
            {"self": [23], "other": 31.304347826086957},
            {"self": [240], "other": 0.5},
            {"self": [28], "other": 0.25},
            {"self": [28], "other": 0.5},
            {"self": [300], "other": 1.6},
            {"self": [300], "other": 2.1333333333333333},
            {"self": [30], "other": 0.5},
            {"self": [320], "other": 0.5},
            {"self": [320], "other": 1.0},
            {"self": [320], "other": 1.5},
            {"self": [320], "other": 2.0},
            {"self": [3234, 2], "other": 0.5},
            {"self": [3234], "other": 0.5},
            {"self": [32], "other": 0.5},
            {"self": [4, 12, 49, 32], "other": 0.1767766952966369},
            {"self": [4, 16, 49, 32], "other": 0.1767766952966369},
            {"self": [40], "other": 0.5},
            {"self": [40], "other": 32.0},
            {"self": [480], "other": 0.5},
            {"self": [50], "other": 0.5},
            {"self": [56], "other": 0.125},
            {"self": [56], "other": 0.25},
            {"self": [56], "other": 0.5},
            {"self": [60], "other": 0.5},
            {"self": [64, 3, 49, 32], "other": 0.1767766952966369},
            {"self": [64, 4, 49, 32], "other": 0.1767766952966369},
            {"self": [640], "other": 0.5},
            {"self": [64], "other": 0.5},
            {"self": [68], "other": 0.5},
            {"self": [7], "other": 0.42857142857142855},
            {"self": [800], "other": 0.6},
            {"self": [80], "other": 0.5},
            {"self": [8732, 2], "other": 0.5},
            {"self": [8732], "other": 0.5},
            # vec other
            {"self": [0, 1], "other": [0, 1]},
            {"self": [0], "other": []},
            {"self": [1, 1, 1, 17], "other": [1, 1, 1, 17]},
            {"self": [1, 1, 1, 1], "other": [1, 1, 1, 1]},
            {"self": [1, 1, 1, 2], "other": [1, 1, 1, 2]},
            {"self": [1, 1, 1, 42], "other": [1, 1, 1, 42]},
            {"self": [1, 1, 1024], "other": [1, 1, 1024]},
            {"self": [1, 1, 1024], "other": [1, 1, 1]},
            {"self": [1, 1, 16, 32], "other": [1, 1, 1, 32]},
            {"self": [1, 1, 3072], "other": [1, 1, 3072]},
            {"self": [1, 1, 32, 1], "other": [1, 1, 32, 1]},
            {"self": [1, 1, 4096], "other": [1, 1, 4096]},
            {"self": [1, 1, 512], "other": [1, 1, 1]},
            {"self": [1, 1, 7, 64], "other": [1, 1, 7, 64]},
            {"self": [1, 1, 768], "other": [1, 1, 1]},
            {"self": [1, 10, 1024], "other": [1, 10, 1]},
            {"self": [1, 10, 512], "other": [1, 10, 1]},
            {"self": [1, 10, 768], "other": [1, 10, 1]},
            {"self": [1, 1024, 1, 1], "other": [1, 1024, 1, 1]},
            {"self": [1, 1024, 2560], "other": [1, 1024, 2560]},
            {"self": [1, 1024, 45, 80], "other": [1, 1024, 1, 1]},
            {"self": [1, 1024, 50, 68], "other": [1, 1024, 1, 1]},
            {"self": [1, 1024, 7, 7], "other": [1, 1024, 1, 1]},
            {"self": [1, 104, 1, 1], "other": [1, 104, 28, 28]},
            {"self": [1, 1056, 1, 1], "other": [1, 1056, 48, 48]},
            {"self": [1, 10], "other": [1, 10]},
            {"self": [1, 12, 3072], "other": [1, 12, 3072]},
            {"self": [1, 120, 1, 1], "other": [1, 120, 14, 14]},
            {"self": [1, 120, 1, 1], "other": [1, 120, 28, 28]},
            {"self": [1, 120, 1, 1], "other": [1, 120, 40, 40]},
            {"self": [1, 120, 28, 28], "other": [1, 120, 1, 1]},
            {"self": [1, 120, 28, 28], "other": [1, 120, 28, 28]},
            {"self": [1, 1232, 1, 1], "other": [1, 1232, 14, 14]},
            {"self": [1, 128, 1, 1], "other": [1, 128, 1, 1]},
            {"self": [1, 128, 100, 136], "other": [1, 128, 1, 1]},
            {"self": [1, 128, 180, 320], "other": [1, 128, 1, 1]},
            {"self": [1, 128, 200, 272], "other": [1, 128, 1, 1]},
            {"self": [1, 128, 90, 160], "other": [1, 128, 1, 1]},
            {"self": [1, 1392, 1, 1], "other": [1, 1392, 14, 14]},
            {"self": [1, 14, 3072], "other": [1, 14, 3072]},
            {"self": [1, 144, 1, 1], "other": [1, 144, 14, 14]},
            {"self": [1, 144, 1, 1], "other": [1, 144, 28, 28]},
            {"self": [1, 15, 1024], "other": [1, 15, 1024]},
            {"self": [1, 15, 512], "other": [1, 15, 1]},
            {"self": [1, 1512, 1, 1], "other": [1, 1512, 7, 7]},
            {"self": [1, 16, 1, 1], "other": [1, 16, 56, 56]},
            {"self": [1, 184, 14, 14], "other": [1, 184, 14, 14]},
            {"self": [1, 192, 32, 42], "other": [1, 1, 1, 42]},
            {"self": [1, 192, 32, 42], "other": [1, 1, 32, 1]},
            {"self": [1, 1], "other": [1, 160]},
            {"self": [1, 200, 14, 14], "other": [1, 200, 14, 14]},
            {"self": [1, 2016, 1, 1], "other": [1, 2016, 7, 7]},
            {"self": [1, 2048, 1, 1], "other": [1, 2048, 1, 1]},
            {"self": [1, 2048, 23, 40], "other": [1, 2048, 1, 1]},
            {"self": [1, 2048, 25, 34], "other": [1, 2048, 1, 1]},
            {"self": [1, 208, 1, 1], "other": [1, 208, 14, 14]},
            {"self": [1, 216, 1, 1], "other": [1, 216, 28, 28]},
            {"self": [1, 224, 1, 1], "other": [1, 224, 56, 56]},
            {"self": [1, 232, 1, 1], "other": [1, 232, 56, 56]},
            {"self": [1, 24, 64, 64], "other": [24, 1, 1]},
            {"self": [1, 240, 1, 1], "other": [1, 240, 14, 14]},
            {"self": [1, 240, 28, 28], "other": [1, 240, 28, 28]},
            {"self": [1, 256, 1, 1], "other": [1, 256, 1, 1]},
            {"self": [1, 256, 100, 136], "other": [1, 256, 1, 1]},
            {"self": [1, 256, 128, 128], "other": [128, 1]},
            {"self": [1, 256, 128, 128], "other": [128]},
            {"self": [1, 256, 180, 320], "other": [1, 256, 1, 1]},
            {"self": [1, 256, 200, 272], "other": [1, 256, 1, 1]},
            {"self": [1, 256, 45, 80], "other": [1, 256, 1, 1]},
            {"self": [1, 256, 50, 68], "other": [1, 256, 1, 1]},
            {"self": [1, 256, 5120], "other": [1, 256, 5120]},
            {"self": [1, 256, 56, 56], "other": [1, 256, 1, 1]},
            {"self": [1, 256, 90, 160], "other": [1, 256, 1, 1]},
            {"self": [1, 288, 1, 1], "other": [1, 288, 7, 7]},
            {"self": [1, 2904, 1, 1], "other": [1, 2904, 24, 24]},
            {"self": [1, 3, 16, 16, 2], "other": [1, 3, 16, 16, 2]},
            {"self": [1, 3, 16, 16, 2], "other": []},
            {"self": [1, 3, 300, 300], "other": [300, 1]},
            {"self": [1, 3, 300, 300], "other": [300]},
            {"self": [1, 3, 32, 32, 2], "other": [1, 3, 32, 32, 2]},
            {"self": [1, 3, 32, 32, 2], "other": []},
            {"self": [1, 3, 320, 320], "other": [320, 1]},
            {"self": [1, 3, 320, 320], "other": [320]},
            {"self": [1, 3, 64, 64, 2], "other": [1, 3, 64, 64, 2]},
            {"self": [1, 3, 64, 64, 2], "other": []},
            {"self": [1, 3, 800, 1066], "other": [1066]},
            {"self": [1, 3, 800, 1066], "other": [800, 1]},
            {"self": [1, 3024, 1, 1], "other": [1, 3024, 7, 7]},
            {"self": [1, 32, 6144], "other": [1, 32, 6144]},
            {"self": [1, 32, 64, 64], "other": [32, 1, 1]},
            {"self": [1, 320, 1, 1], "other": [1, 320, 14, 14]},
            {"self": [1, 32], "other": [1, 32]},
            {"self": [1, 336, 1, 1], "other": [1, 336, 14, 14]},
            {"self": [1, 3712, 1, 1], "other": [1, 3712, 7, 7]},
            {"self": [1, 4096, 1280], "other": [1, 4096, 1280]},
            {"self": [1, 440, 1, 1], "other": [1, 440, 7, 7]},
            {"self": [1, 448, 1, 1], "other": [1, 448, 28, 28]},
            {"self": [1, 45, 3072], "other": [1, 45, 3072]},
            {"self": [1, 48, 1, 1], "other": [1, 48, 56, 56]},
            {"self": [1, 480, 1, 1], "other": [1, 480, 10, 10]},
            {"self": [1, 480, 1, 1], "other": [1, 480, 14, 14]},
            {"self": [1, 480, 1, 1], "other": [1, 480, 20, 20]},
            {"self": [1, 480, 14, 14], "other": [1, 480, 1, 1]},
            {"self": [1, 480, 14, 14], "other": [1, 480, 14, 14]},
            {"self": [1, 5, 16, 32], "other": [1, 5, 1, 32]},
            {"self": [1, 5, 4096], "other": [1, 5, 4096]},
            {"self": [1, 50, 3072], "other": [1, 50, 3072]},
            {"self": [1, 512, 1, 1], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 1, 1], "other": [1, 512, 38, 38]},
            {"self": [1, 512, 100, 136], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 23, 40], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 25, 34], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 28, 28], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 45, 80], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 50, 68], "other": [1, 512, 1, 1]},
            {"self": [1, 512, 90, 160], "other": [1, 512, 1, 1]},
            {"self": [1, 528, 1, 1], "other": [1, 528, 96, 96]},
            {"self": [1, 576, 1, 1], "other": [1, 576, 14, 14]},
            {"self": [1, 576, 1, 1], "other": [1, 576, 7, 7]},
            {"self": [1, 59], "other": [1, 59]},
            {"self": [1, 60], "other": [1, 60]},
            {"self": [1, 64, 1, 1], "other": [1, 64, 1, 1]},
            {"self": [1, 64, 1, 1], "other": [1, 64, 56, 56]},
            {"self": [1, 64, 120, 160], "other": [1, 1, 120, 160]},
            {"self": [1, 64, 120, 160], "other": [120, 1]},
            {"self": [1, 64, 120, 160], "other": [160]},
            {"self": [1, 64, 180, 320], "other": [1, 64, 1, 1]},
            {"self": [1, 64, 200, 272], "other": [1, 64, 1, 1]},
            {"self": [1, 64, 240, 320], "other": [240, 1]},
            {"self": [1, 64, 240, 320], "other": [320]},
            {"self": [1, 64, 30, 40], "other": [1, 1, 30, 40]},
            {"self": [1, 64, 30, 40], "other": [30, 1]},
            {"self": [1, 64, 30, 40], "other": [40]},
            {"self": [1, 64, 360, 640], "other": [1, 64, 1, 1]},
            {"self": [1, 64, 400, 544], "other": [1, 64, 1, 1]},
            {"self": [1, 64, 480, 640], "other": [480, 1]},
            {"self": [1, 64, 480, 640], "other": [640]},
            {"self": [1, 64, 5120], "other": [1, 64, 5120]},
            {"self": [1, 64, 60, 80], "other": [1, 1, 60, 80]},
            {"self": [1, 64, 60, 80], "other": [60, 1]},
            {"self": [1, 64, 60, 80], "other": [80]},
            {"self": [1, 672, 1, 1], "other": [1, 672, 10, 10]},
            {"self": [1, 672, 1, 1], "other": [1, 672, 14, 14]},
            {"self": [1, 672, 1, 1], "other": [1, 672, 20, 20]},
            {"self": [1, 672, 1, 1], "other": [1, 672, 7, 7]},
            {"self": [1, 672, 14, 14], "other": [1, 672, 1, 1]},
            {"self": [1, 672, 14, 14], "other": [1, 672, 14, 14]},
            {"self": [1, 672, 7, 7], "other": [1, 672, 1, 1]},
            {"self": [1, 696, 1, 1], "other": [1, 696, 28, 28]},
            {"self": [1, 7, 3072], "other": [1, 7, 3072]},
            {"self": [1, 71, 7, 64], "other": [1, 1, 7, 64]},
            {"self": [1, 72, 1, 1], "other": [1, 72, 28, 28]},
            {"self": [1, 72, 1, 1], "other": [1, 72, 40, 40]},
            {"self": [1, 72, 1, 1], "other": [1, 72, 56, 56]},
            {"self": [1, 72, 28, 28], "other": [1, 72, 1, 1]},
            {"self": [1, 72, 56, 56], "other": [1, 72, 56, 56]},
            {"self": [1, 7392, 1, 1], "other": [1, 7392, 12, 12]},
            {"self": [1, 768, 14, 14], "other": [1, 768, 1, 1]},
            {"self": [1, 784, 1, 1], "other": [1, 784, 7, 7]},
            {"self": [1, 888, 1, 1], "other": [1, 888, 7, 7]},
            {"self": [1, 896, 1, 1], "other": [1, 896, 14, 14]},
            {"self": [1, 9, 128], "other": [1, 9, 128]},
            {"self": [1, 9, 16384], "other": [1, 9, 16384]},
            {"self": [1, 9, 3072], "other": [1, 9, 3072]},
            {"self": [1, 9, 4096], "other": [1, 9, 4096]},
            {"self": [1, 9, 8192], "other": [1, 9, 8192]},
            {"self": [1, 96, 1, 1], "other": [1, 96, 14, 14]},
            {"self": [1, 960, 1, 1], "other": [1, 960, 7, 7]},
            {"self": [1, 960, 7, 7], "other": [1, 960, 1, 1]},
            {"self": [1, 960, 7, 7], "other": [1, 960, 7, 7]},
            {"self": [100], "other": []},
            {"self": [1024], "other": [1, 1, 1024]},
            {"self": [1024], "other": [1, 10, 1024]},
            {"self": [1024], "other": [1, 197, 1024]},
            {"self": [12], "other": []},
            {"self": [136], "other": []},
            {"self": [13], "other": []},
            {"self": [16, 1], "other": [1, 1, 32]},
            {"self": [16, 6, 64, 64], "other": [6, 1, 1]},
            {"self": [16, 8, 64, 64], "other": [8, 1, 1]},
            {"self": [17], "other": []},
            {"self": [1], "other": [1]},
            {"self": [2, 1], "other": []},
            {"self": [2, 7, 2048], "other": [2, 7, 2048]},
            {"self": [25], "other": []},
            {"self": [300], "other": []},
            {"self": [3234, 1], "other": [3234, 1]},
            {"self": [3234, 2], "other": [2]},
            {"self": [34], "other": []},
            {"self": [4, 12, 64, 64], "other": [12, 1, 1]},
            {"self": [4, 16, 64, 64], "other": [16, 1, 1]},
            {"self": [50], "other": []},
            {"self": [512], "other": [1, 1, 512]},
            {"self": [512], "other": [1, 10, 512]},
            {"self": [512], "other": [1, 15, 512]},
            {"self": [64, 3, 64, 64], "other": [3, 1, 1]},
            {"self": [64, 4, 64, 64], "other": [4, 1, 1]},
            {"self": [68], "other": []},
            {"self": [768], "other": [1, 1, 768]},
            {"self": [768], "other": [1, 10, 768]},
            {"self": [768], "other": [1, 197, 768]},
            {"self": [7], "other": []},
            {"self": [8732, 1], "other": [8732, 1]},
            {"self": [8732, 2], "other": [2]},
            {"self": [9], "other": []},
            {"self": [], "other": [0, 1]},
            {"self": [], "other": [1, 1, 768]},
            {"self": [], "other": [1, 24, 768]},
            {"self": [], "other": [3234, 1]},
            {"self": [], "other": [8732, 1]},
        ],
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape["self"])

    if isinstance(input_shape["other"], list):
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(input_shape["other"])
    else:
        torch_input_tensor_b = torch.tensor(input_shape["other"], dtype=torch.float32)
        # torch_input_tensor_b = input_shape["other"]

    golden_function = ttnn.get_golden_function(ttnn.mul)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    # if isinstance(input_shape["other"], list):
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )
    # else:
    #     input_tensor_b = input_shape["other"]

    start_time = start_measuring_time()
    result = ttnn.mul(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, pcc=0.99), e2e_perf]
