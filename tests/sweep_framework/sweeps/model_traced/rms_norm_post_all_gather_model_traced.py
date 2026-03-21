# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    infer_mesh_shape_from_params,
    detect_mesh_shape_from_hardware,
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
    if not mesh_shape:
        mesh_shape = infer_mesh_shape_from_params(model_traced_params)
    if not mesh_shape:
        mesh_shape = detect_mesh_shape_from_hardware()
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


def _is_sharded_memory_config(mem_config):
    """Check if a memory config uses a sharded layout."""
    if mem_config is None:
        return False
    return hasattr(mem_config, "memory_layout") and "SHARDED" in str(mem_config.memory_layout)


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
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

    if isinstance(input_a_shape, dict) and "self" in input_a_shape:
        shape = input_a_shape["self"] if isinstance(input_a_shape["self"], tuple) else tuple(input_a_shape["self"])
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

    # MasterConfigLoader provides the stats tensor (arg1) shape as input_b_shape.
    # Prefer it over the legacy "other" field from dict-based input_a_shape.
    if input_b_shape is not None:
        stats_shape_from_trace = tuple(input_b_shape) if isinstance(input_b_shape, list) else input_b_shape

    eps = kwargs.get("epsilon", 1e-5)
    op_kwargs = build_op_kwargs(kwargs, exclude={"epsilon"}, output_memory_config=output_memory_config)
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

    input_a_target_sharded = None
    input_a_create_config = input_a_memory_config if input_a_memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    if _is_sharded_memory_config(input_a_memory_config):
        input_a_target_sharded = input_a_memory_config
        input_a_create_config = ttnn.DRAM_MEMORY_CONFIG

    if is_mesh_device:
        input_tensor = create_tensor_on_mesh(
            torch_input, device, input_a_dtype, ttnn.TILE_LAYOUT, input_a_create_config, input_a_tensor_placement
        )
        weight_tensor = create_tensor_on_mesh(
            torch_gamma_4d,
            device,
            input_b_dtype if input_b_dtype is not None else ttnn.bfloat16,
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
            memory_config=input_a_create_config,
        )
        weight_tensor = ttnn.from_torch(
            torch_gamma_4d,
            dtype=input_b_dtype if input_b_dtype is not None else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    if input_a_target_sharded is not None:
        input_tensor = ttnn.to_memory_config(input_tensor, input_a_target_sharded)

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

    input_b_target_sharded = None
    input_b_create_config = input_b_memory_config if input_b_memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    if _is_sharded_memory_config(input_b_memory_config):
        input_b_target_sharded = input_b_memory_config
        input_b_create_config = ttnn.DRAM_MEMORY_CONFIG

    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", input_a_tensor_placement)
    stats_dtype = input_b_dtype if input_b_dtype is not None else ttnn.bfloat16

    if is_mesh_device:
        stats_tensor = create_tensor_on_mesh(
            torch_stats, device, stats_dtype, ttnn.TILE_LAYOUT, input_b_create_config, input_b_tensor_placement
        )
    else:
        stats_tensor = ttnn.from_torch(
            torch_stats,
            dtype=stats_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_b_create_config,
        )

    if input_b_target_sharded is not None:
        stats_tensor = ttnn.to_memory_config(stats_tensor, input_b_target_sharded)

    start_time = start_measuring_time()
    output_tensor = ttnn.rms_norm_post_all_gather(
        input_tensor, stats_tensor, epsilon=eps, weight=weight_tensor, **op_kwargs
    )
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
