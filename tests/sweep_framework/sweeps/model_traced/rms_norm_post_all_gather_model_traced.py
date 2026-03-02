# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.master_config_loader import MasterConfigLoader


TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("rms_norm_post_all_gather", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype=None,
    input_b_memory_config=None,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    if isinstance(input_shape, dict) and "self" in input_shape:
        shape = input_shape["self"] if isinstance(input_shape["self"], tuple) else tuple(input_shape["self"])
        # "other" is the stats tensor shape (not weight), use it to determine n_devices
        stats_shape_from_trace = input_shape.get("other")
        if stats_shape_from_trace is not None:
            stats_shape_from_trace = (
                tuple(stats_shape_from_trace) if isinstance(stats_shape_from_trace, list) else stats_shape_from_trace
            )
    elif isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape) if isinstance(input_shape, list) else input_shape
        stats_shape_from_trace = None
    else:
        shape = (1, 1, 32, 32)
        stats_shape_from_trace = None

    eps = 1e-5
    hidden_dim = shape[-1]

    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        shape
    )

    # Weight shape: [1, 1, hidden_dim//32, 32] in ROW_MAJOR_LAYOUT, matching model usage
    weight_sticks = max(hidden_dim // 32, 1)
    weight_4d_shape = (1, 1, weight_sticks, 32)
    weight_size = weight_sticks * 32
    torch_gamma_1d = torch.randn(weight_size, dtype=torch.float32)
    torch_gamma_4d = torch_gamma_1d.reshape(weight_4d_shape)

    # Golden: output = x * gamma / sqrt(mean(x^2) + eps)
    torch_output = (
        torch_input * torch_gamma_1d[:hidden_dim] / torch.sqrt(torch.mean(torch_input**2, dim=-1, keepdim=True) + eps)
    )

    input_tensor = ttnn.from_torch(
        torch_input, dtype=input_a_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    weight_tensor = ttnn.from_torch(
        torch_gamma_4d,
        dtype=input_b_dtype or input_a_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Determine n_devices from the traced stats shape
    # Stats shape is (batch..., 32 * n_devices), each device contributes a tile-width (32) of stats
    if stats_shape_from_trace and len(stats_shape_from_trace) >= 1:
        n_devices = max(stats_shape_from_trace[-1] // 32, 1)
    else:
        n_devices = 1

    # Construct stats tensor matching the gathered format that rms_norm_post_all_gather expects.
    # The full sum(x^2) needs to be split across n_devices slots as if each device
    # computed partial stats on 1/n_devices of the hidden_dim.
    sum_x2 = torch_input.pow(2).sum(dim=-1, keepdim=True)
    stats_width = 32 * n_devices
    stats_tensor_shape = list(shape[:-1]) + [stats_width]
    torch_stats = torch.zeros(stats_tensor_shape, dtype=torch.float32)
    per_device_sum = sum_x2 / n_devices
    for i in range(n_devices):
        torch_stats[..., i * 32 : i * 32 + 1] = per_device_sum

    stats_tensor = ttnn.from_torch(
        torch_stats, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.rms_norm_post_all_gather(input_tensor, stats_tensor, epsilon=eps, weight=weight_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
