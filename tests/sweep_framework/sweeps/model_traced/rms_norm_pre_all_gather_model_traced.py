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
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)


TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("rms_norm_pre_all_gather")

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
    input_b_memory_config=None,
    program_config=None,
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
    elif isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape) if isinstance(input_a_shape, list) else input_a_shape
    else:
        shape = (1, 1, 32, 32)

    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        shape
    )

    torch_expected_stats = torch_input.pow(2).sum(dim=-1, keepdim=True)

    # Create tensor in DRAM first, then move to target memory config
    if is_mesh_device:
        input_tensor = create_tensor_on_mesh(
            torch_input, device, input_a_dtype, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, input_a_tensor_placement
        )
    else:
        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=input_a_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # If the traced config specifies a sharded memory config, move the tensor there
    is_sharded = False
    if not is_mesh_device and hasattr(input_a_memory_config, "memory_layout"):
        mem_layout = str(input_a_memory_config.memory_layout)
        if "SHARDED" in mem_layout:
            is_sharded = True
            input_tensor = ttnn.to_memory_config(input_tensor, input_a_memory_config)

    # Build program_config if provided from traced JSON
    ttnn_program_config = None
    if program_config and isinstance(program_config, dict):
        compute_grid = program_config.get("compute_with_storage_grid_size", {})
        ttnn_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(compute_grid.get("x", 1), compute_grid.get("y", 1)),
            subblock_w=program_config.get("subblock_w", 1),
            block_h=program_config.get("block_h", 1),
            block_w=program_config.get("block_w", 1),
            inplace=bool(program_config.get("inplace", 0)),
        )

    start_time = start_measuring_time()
    op_kwargs = {"dtype": ttnn.bfloat16}
    if ttnn_program_config is not None:
        op_kwargs["program_config"] = ttnn_program_config
    tt_stats = ttnn.rms_norm_pre_all_gather(input_tensor, **op_kwargs)
    tt_stats_torch = mesh_tensor_to_torch(tt_stats, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    tt_sum_x2 = tt_stats_torch[..., 0:1]

    return [check_with_pcc(torch_expected_stats, tt_sum_x2, 0.99), e2e_perf]
