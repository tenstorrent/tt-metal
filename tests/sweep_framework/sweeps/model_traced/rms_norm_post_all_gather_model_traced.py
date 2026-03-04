# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader, dict_to_compute_kernel_config
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)


TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("rms_norm_post_all_gather")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
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


def mesh_device_fixture():
    mesh_shape = get_mesh_shape()
    if mesh_shape:
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0)
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype=None,
    input_b_memory_config=None,
    output_memory_config=None,
    memory_config=None,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")

    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    if isinstance(input_a_shape, dict) and "self" in input_a_shape:
        shape = input_a_shape["self"] if isinstance(input_a_shape["self"], tuple) else tuple(input_a_shape["self"])
        # "other" is the stats tensor shape (not weight), use it to determine n_devices
        stats_shape_from_trace = input_a_shape.get("other")
        if stats_shape_from_trace is not None:
            stats_shape_from_trace = (
                tuple(stats_shape_from_trace) if isinstance(stats_shape_from_trace, list) else stats_shape_from_trace
            )
    elif isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape) if isinstance(input_a_shape, list) else input_a_shape
        stats_shape_from_trace = None
    else:
        shape = (1, 1, 32, 32)
        stats_shape_from_trace = None

    eps = kwargs.get("epsilon", 1e-5)
    compute_kernel_config = kwargs.get("compute_kernel_config", None)
    if isinstance(compute_kernel_config, dict):
        compute_kernel_config = dict_to_compute_kernel_config(compute_kernel_config)
    hidden_dim = shape[-1]

    # rms_norm_post_all_gather only supports BFLOAT16 and BFLOAT8_B input dtypes
    if input_a_dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
        input_a_dtype = ttnn.bfloat16

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

    if is_mesh_device:
        input_tensor = create_tensor_on_mesh(
            torch_input, device, input_a_dtype, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, input_a_tensor_placement
        )
        weight_tensor = create_tensor_on_mesh(
            torch_gamma_4d,
            device,
            input_b_dtype or input_a_dtype,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
            kwargs.get("weight_tensor_placement", input_a_tensor_placement),
        )
    else:
        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=input_a_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

    if is_mesh_device:
        stats_tensor = create_tensor_on_mesh(
            torch_stats, device, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, input_a_tensor_placement
        )
    else:
        stats_tensor = ttnn.from_torch(
            torch_stats,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    start_time = start_measuring_time()
    op_kwargs = {"epsilon": eps, "weight": weight_tensor}
    if compute_kernel_config is not None:
        op_kwargs["compute_kernel_config"] = compute_kernel_config
    output_tensor = ttnn.rms_norm_post_all_gather(input_tensor, stats_tensor, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
