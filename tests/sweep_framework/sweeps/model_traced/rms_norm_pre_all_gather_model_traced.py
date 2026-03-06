# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader, dict_to_compute_kernel_config, parse_dtype
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

    if isinstance(input_a_shape, dict) and "self" in input_a_shape:
        shape = input_a_shape["self"] if isinstance(input_a_shape["self"], tuple) else tuple(input_a_shape["self"])
    elif isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape) if isinstance(input_a_shape, list) else input_a_shape
    else:
        shape = (1, 1, 32, 32)

    # rms_norm_pre_all_gather only supports BFLOAT16 and BFLOAT8_B input dtypes
    if input_a_dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
        input_a_dtype = ttnn.bfloat16

    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype)(
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

    ttnn_program_config = None
    if program_config and isinstance(program_config, dict):
        config_type = program_config.get("type", "")
        config_value = program_config.get("value", "")

        if "ShardedMultiCore" in config_type and isinstance(config_value, str):
            x_m = re.search(r"x=(\d+)", config_value)
            y_m = re.search(r"y=(\d+)", config_value)
            sw_m = re.search(r"subblock_w=(\d+)", config_value)
            bh_m = re.search(r"block_h=(\d+)", config_value)
            bw_m = re.search(r"block_w=(\d+)", config_value)
            inp_m = re.search(r"inplace=(\d+)", config_value)
            ttnn_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(
                    int(x_m.group(1)) if x_m else 1,
                    int(y_m.group(1)) if y_m else 1,
                ),
                subblock_w=int(sw_m.group(1)) if sw_m else 1,
                block_h=int(bh_m.group(1)) if bh_m else 1,
                block_w=int(bw_m.group(1)) if bw_m else 1,
                inplace=bool(int(inp_m.group(1))) if inp_m else False,
            )
        elif "Default" in config_type:
            pass
        elif "compute_with_storage_grid_size" in program_config:
            compute_grid = program_config.get("compute_with_storage_grid_size", {})
            ttnn_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(compute_grid.get("x", 1), compute_grid.get("y", 1)),
                subblock_w=program_config.get("subblock_w", 1),
                block_h=program_config.get("block_h", 1),
                block_w=program_config.get("block_w", 1),
                inplace=bool(program_config.get("inplace", 0)),
            )

    # Parse compute_kernel_config and dtype from traced config
    compute_kernel_config = kwargs.get("compute_kernel_config", None)
    if isinstance(compute_kernel_config, dict):
        compute_kernel_config = dict_to_compute_kernel_config(compute_kernel_config)
    output_dtype = kwargs.get("dtype", ttnn.bfloat16)
    if isinstance(output_dtype, dict):
        output_dtype = parse_dtype(output_dtype)
    if output_dtype is None:
        output_dtype = ttnn.bfloat16

    start_time = start_measuring_time()
    op_kwargs = {"dtype": output_dtype}
    if ttnn_program_config is not None:
        op_kwargs["program_config"] = ttnn_program_config
    if compute_kernel_config is not None:
        op_kwargs["compute_kernel_config"] = compute_kernel_config
    tt_stats = ttnn.rms_norm_pre_all_gather(input_tensor, **op_kwargs)
    tt_stats_torch = mesh_tensor_to_torch(tt_stats, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    tt_sum_x2 = tt_stats_torch[..., 0:1]

    # Use 0.95 PCC threshold: this operation computes intermediate stats (sum(x^2))
    # which can have lower precision in bfloat16 accumulation, especially without fp32_dest_acc_en.
    # The final model accuracy is maintained by rms_norm_post_all_gather.
    pcc_threshold = 0.99 if compute_kernel_config is not None else 0.95
    return [check_with_pcc(torch_expected_stats, tt_sum_x2, pcc_threshold), e2e_perf]
