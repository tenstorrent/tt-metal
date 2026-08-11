# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    get_mesh_composer,
    reconcile_golden_to_actual,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

import re as _re_mod


def _parse_mesh_shape(s):
    """Parse a mesh_device_shape string like '[4, 8]' or '(4, 8)' into a list of ints.

    Avoids ast.literal_eval (Cycode SAST CWE-400) by extracting integer tokens
    directly. mesh_device_shape comes from the trace's machine_info and is
    always a short list/tuple of small positive ints.
    """
    if not isinstance(s, str) or len(s) > 64:
        return None
    matches = _re_mod.findall(r"-?\d+", s)
    if not matches:
        return None
    return [int(m) for m in matches]


TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("unsqueeze_to_4D")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 32, 32)],
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
    mesh_shape = get_model_traced_mesh_shape()
    device = create_mesh_device(mesh_shape)
    device_name = ttnn.get_arch_name()
    yield (device, device_name)
    ttnn.close_mesh_device(device)


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config=None,
    memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    # Check kwargs for storage_type — V2 loader provides input_a_storage_type for HOST tensors
    if storage_type == "StorageType::DEVICE":
        if "input_a_storage_type" in kwargs:
            storage_type = kwargs["input_a_storage_type"]
        elif "storage_type" in kwargs:
            storage_type = kwargs["storage_type"]

    # Some master configs were traced under a different mesh_device_shape than the
    # fixture's (e.g. cfg 361 was traced on [1, 32] while the rest of gpt_oss is
    # [4, 8]). config_hash bakes in mesh_device_shape, so we reshape the live mesh
    # to the placement's target shape, run the op, then reshape back.
    _saved_mesh_shape = None
    if is_mesh_device and isinstance(input_a_tensor_placement, dict):
        _target_str = input_a_tensor_placement.get("mesh_device_shape")
        if isinstance(_target_str, str):
            _target = _parse_mesh_shape(_target_str)
            _current = list(device.shape) if hasattr(device, "shape") else None
            if _target and _current and _target != _current:
                _saved_mesh_shape = _current
                device.reshape(ttnn.MeshShape(*_target))
    op_kwargs = build_op_kwargs(
        kwargs, output_memory_config=output_memory_config
    )  # op_kwargs available but op does not accept extra kwargs

    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    # Some traced models record 1D tensors with TILE layout (e.g. tt_dit/wan2.2
    # tail-end vectors). Preserve the original rank so the trace's original_shape
    # matches master exactly. ttnn handles 1D TILE tensors internally.

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # PyTorch reference: unsqueeze to 4D
    torch_output_tensor = torch_input_tensor_a
    while len(torch_output_tensor.shape) < 4:
        torch_output_tensor = torch.unsqueeze(torch_output_tensor, 0)

    is_host = storage_type and "HOST" in str(storage_type)
    # V2 loader doesn't expose storage_type directly; detect HOST from memory_config=None
    # (HOST tensors have no device memory config)
    if input_a_memory_config is None:
        is_host = True

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor_a = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        # HOST tensor — create without placing on device.  On mesh devices the
        # master trace still records mesh metadata (distribution_shape, mesh_device_shape).
        # Pass mesh_mapper so the tensor carries the correct topology.
        if is_mesh_device and input_a_tensor_placement:
            from tests.sweep_framework.sweep_utils.mesh_tensor_utils import replicate_with_topology

            device_tensor = replicate_with_topology(
                torch_input_tensor_a,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config or ttnn.DRAM_MEMORY_CONFIG,
                input_a_tensor_placement,
            )
            input_tensor_a = ttnn.from_device(device_tensor)
        else:
            input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    try:
        output_tensor = ttnn.unsqueeze_to_4D(input_tensor_a)
        mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
        output_tensor = mesh_tensor_to_torch(
            output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer
        )
        e2e_perf = stop_measuring_time(start_time)
        if is_mesh_device:
            torch_output_tensor = reconcile_golden_to_actual(
                torch_output_tensor, output_tensor, input_a_tensor_placement
            )
        pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    finally:
        if _saved_mesh_shape is not None:
            device.reshape(ttnn.MeshShape(*_saved_mesh_shape))

    return [pcc, e2e_perf]
