# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import numpy as np
from scipy.optimize import minimize_scalar, bisect
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.utility_functions import torch_random
from functools import partial

device_id = 0
device = ttnn.open_device(device_id=device_id)


def compute_max_error(eltwise_op, R):
    if R <= 0:
        return np.inf

    # Define the error function for a given x within [-R, R]
    def error_function(x):
        input_a_dtype = ttnn.bfloat16
        torch_input_tensor = ttnn.from_torch(
            torch.tensor([[x]], dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=None,
        )
        torch_input_tensor = ttnn.to_torch(torch_input_tensor)

        golden_function = ttnn.get_golden_function(eltwise_op)
        golden_func = golden_function(torch_input_tensor)

        input_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=input_a_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn_cos = eltwise_op(input_tensor)
        ttnn_cos_torch = ttnn.to_torch(ttnn_cos)
        error = torch.abs(ttnn_cos_torch - golden_func).item()
        return error

    result = minimize_scalar(lambda x: -error_function(x), bounds=(-R, R), method="bounded")

    return error_function(result.x)


def find_max_R(eltwise_op, tolerance, R_min=0.1, R_max=100, precision=1e-3):
    def error_minus_tolerance(R):
        return compute_max_error(eltwise_op, R) - tolerance

    err_min = error_minus_tolerance(R_min)
    err_max = error_minus_tolerance(R_max)

    if err_min > 0:
        print("Error at R_min exceeds tolerance. Decrease R_min or increase tolerance.")
        return None
    if err_max < 0:
        print("Error at R_max is within tolerance. Increase R_max to find a larger range.")
        return R_max

    max_R = bisect(error_minus_tolerance, R_min, R_max, xtol=precision, maxiter=200)
    return max_R


def find_next_valid_tolerance(
    eltwise_op, initial_tolerance, tolerance_step=0.001, max_tolerance=1.0, R_min=0.1, R_max=100, precision=1e-3
):
    tolerance = initial_tolerance
    while tolerance <= max_tolerance:
        max_R = find_max_R(eltwise_op, tolerance, R_min, R_max, precision)
        if max_R is not None:
            return tolerance, max_R
        else:
            tolerance += tolerance_step
    return None, None


if __name__ == "__main__":
    initial_tolerance = 0.001
    tolerance_step = 0.001
    max_tolerance = 0.1

    R_min = 0.01
    R_max = 3.3912069e38 / 2

    tolerance, max_R = find_next_valid_tolerance(
        ttnn.cos,
        initial_tolerance,
        tolerance_step,
        max_tolerance,
        R_min,
        R_max,
        precision=1e-3,
    )

    if max_R is not None:
        print(f"The widest range is [-{max_R:.5f}, {max_R:.5f}] where the maximum error is within {tolerance}")
    else:
        print("Could not find a range where the error is within the tolerance.")
