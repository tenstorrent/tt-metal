# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_named_tensor_kwargs
import re


def dict_to_layernorm_program_config(cfg):
    if cfg is None or not isinstance(cfg, dict):
        return cfg
    cfg_type = cfg.get("type", "")
    val_str = str(cfg.get("value", ""))

    if "LayerNormShardedMultiCoreProgramConfig" in cfg_type:
        m = re.search(r"x\s*=\s*(\d+).*?y\s*=\s*(\d+)", val_str)
        grid = ttnn.CoreCoord(int(m.group(1)), int(m.group(2))) if m else ttnn.CoreCoord(8, 4)

        def _int(name, default=0):
            p = re.search(rf"(?<![a-zA-Z_]){name}\s*=\s*(\d+)", val_str)
            return int(p.group(1)) if p else default

        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid,
            subblock_w=_int("subblock_w", 4),
            block_h=_int("block_h", 1),
            block_w=_int("block_w", 4),
            inplace=bool(_int("inplace", 0)),
        )

    if "LayerNormDefaultProgramConfig" in cfg_type:
        return ttnn.LayerNormDefaultProgramConfig()

    return cfg


TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("rms_norm")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "weight_dtype": [ttnn.bfloat16],
        "weight_layout": [ttnn.TILE_LAYOUT],
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
            device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")

    # Build op kwargs — auto-parses compute_kernel_config, memory_config dicts;
    # auto-filters weight_* named tensor kwargs and infrastructure keys.
    # Exclude program_config because it needs custom parsing (LayerNorm-specific).
    op_kwargs = build_op_kwargs(kwargs, exclude={"program_config"}, output_memory_config=output_memory_config)

    # Handle program_config with custom parser
    program_config = kwargs.get("program_config")
    if isinstance(program_config, dict):
        program_config = dict_to_layernorm_program_config(program_config)
    if program_config is not None:
        op_kwargs["program_config"] = program_config

    # Use named memory_config for output if output_memory_config not set
    if output_memory_config is None and "memory_config" in op_kwargs:
        output_memory_config = op_kwargs.pop("memory_config")
    # If output_memory_config is explicitly set, remove duplicate memory_config from op_kwargs
    elif "memory_config" in op_kwargs:
        op_kwargs.pop("memory_config")
    if output_memory_config is not None:
        op_kwargs["memory_config"] = output_memory_config

    input_shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    # Extract weight named tensor kwargs
    weight_info = extract_named_tensor_kwargs(kwargs, "weight")
    if weight_info and weight_info["shape"] is not None:
        w_shape = (
            tuple(weight_info["shape"]) if isinstance(weight_info["shape"], (list, tuple)) else weight_info["shape"]
        )
        w_dtype = weight_info["dtype"]
        w_layout = weight_info["layout"]
        w_mem = weight_info["memory_config"] or ttnn.DRAM_MEMORY_CONFIG
        w_placement = weight_info["tensor_placement"]
    else:
        w_shape = (input_shape[-1],)
        w_dtype = kwargs.get("weight_dtype", None)
        w_layout = kwargs.get("weight_layout", None)
        w_mem = ttnn.DRAM_MEMORY_CONFIG
        w_placement = None

    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        input_shape
    )

    torch_weight = torch.randn(w_shape, dtype=torch.float32)

    # PyTorch golden: RMS norm = x * weight / sqrt(mean(x^2) + eps)
    # Need 1D weight matching input's last dim for broadcasting
    if len(w_shape) > 1:
        weight_size = input_shape[-1]
        torch_weight_1d = torch_weight.flatten()[:weight_size]
    else:
        torch_weight_1d = torch_weight

    eps = float(op_kwargs.get("epsilon", 1e-5))
    rms = torch.sqrt(torch.mean(torch_input**2, dim=-1, keepdim=True) + eps)
    torch_output = torch_input * torch_weight_1d / rms

    is_host = storage_type and "HOST" in str(storage_type)

    # Create input tensor
    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor = create_tensor_on_mesh(
                torch_input,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            input_tensor = ttnn.from_torch(
                torch_input,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor = ttnn.from_torch(torch_input, dtype=input_a_dtype, layout=input_a_layout)

    # Reshape weight for TILE layout compatibility
    if w_layout == ttnn.TILE_LAYOUT and len(w_shape) >= 2:
        weight_size = input_shape[-1]
        torch_weight_reshaped = torch_weight.flatten()[:weight_size].reshape([1, 1, 1, weight_size])
    elif len(w_shape) == 1:
        torch_weight_reshaped = (
            torch_weight.reshape([1, 1, 1, w_shape[0]]) if w_layout == ttnn.TILE_LAYOUT else torch_weight
        )
    else:
        torch_weight_reshaped = torch_weight

    if is_mesh_device and w_placement:
        weight_tensor = create_tensor_on_mesh(
            torch_weight_reshaped,
            device,
            w_dtype,
            w_layout,
            w_mem,
            w_placement,
        )
    else:
        weight_tensor = ttnn.from_torch(
            torch_weight_reshaped,
            dtype=w_dtype,
            layout=w_layout,
            device=device,
            memory_config=w_mem,
        )

    start_time = start_measuring_time()
    output_tensor = ttnn.rms_norm(input_tensor, weight=weight_tensor, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output, output_tensor, 0.999)
    return [pcc, e2e_perf]
