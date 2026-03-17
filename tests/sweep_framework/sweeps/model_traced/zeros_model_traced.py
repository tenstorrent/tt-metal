# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    mesh_tensor_to_torch,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("zeros")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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
    input_a_shape=None,
    input_a_dtype=None,
    input_a_layout=None,
    input_a_memory_config=None,
    output_memory_config=None,
    memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # ttnn.zeros has no tensor inputs; the model trace provides shape/dtype/layout/memory_config
    # directly as kwargs rather than as input_a_* prefixed names.
    shape_val = input_a_shape or kwargs.pop("shape", None) or (1, 1, 32, 32)
    dtype_val = input_a_dtype or kwargs.pop("dtype", None) or ttnn.bfloat16
    layout_val = input_a_layout or kwargs.pop("layout", None) or ttnn.TILE_LAYOUT
    mem_cfg = (
        input_a_memory_config or kwargs.pop("memory_config", None)
        if input_a_memory_config is not None
        else kwargs.pop("memory_config", None)
    )

    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    if output_memory_config is None and mem_cfg is not None:
        output_memory_config = mem_cfg
    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    if isinstance(dtype_val, str):
        from tests.sweep_framework.master_config_loader_v2 import parse_dtype

        dtype_val = parse_dtype(dtype_val)

    if isinstance(layout_val, str):
        from tests.sweep_framework.master_config_loader_v2 import parse_layout

        layout_val = parse_layout(layout_val)

    if isinstance(output_memory_config, dict):
        from tests.sweep_framework.master_config_loader_v2 import dict_to_memory_config

        output_memory_config = dict_to_memory_config(output_memory_config)

    shape = tuple(shape_val) if isinstance(shape_val, (list, tuple)) else shape_val

    torch_output_tensor = torch.zeros(shape, dtype=torch.float32)

    start_time = start_measuring_time()
    output_tensor = ttnn.zeros(
        shape=shape,
        dtype=dtype_val,
        layout=layout_val,
        device=device,
        memory_config=output_memory_config,
        **op_kwargs,
    )
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
