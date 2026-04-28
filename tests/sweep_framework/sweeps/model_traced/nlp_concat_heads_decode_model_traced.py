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
    replicate_with_topology,
    mesh_tensor_to_torch,
    get_mesh_composer,
)

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::nlp_concat_heads_decode")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 12, 32, 64)],
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
    num_heads=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    if isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape)
    else:
        shape = input_a_shape

    # num_heads is required - try to get from op_kwargs first, then kwargs, then infer.
    # op_kwargs is the authoritative source because it mirrors the master trace kwargs.
    if num_heads is None:
        num_heads = op_kwargs.pop("num_heads", None)
    if num_heads is None:
        num_heads = kwargs.get("num_heads", None)
    if num_heads is None:
        # Fallback: infer from input shape (may be tile-padded — less reliable)
        if len(shape) == 4 and shape[1] == 1:
            num_heads = shape[2]
        else:
            num_heads = 16

    # The V2 vector may contain a tile-padded shape (e.g. dim-2 = 32) while the
    # master trace records the original logical shape (e.g. dim-2 = 8 = num_heads).
    # Override dim-2 with the true num_heads so the tracer records the correct shape.
    if len(shape) == 4 and num_heads is not None and shape[2] != num_heads:
        shape = (shape[0], shape[1], num_heads, shape[3])

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # Proper torch reference from test_nlp_concat_heads_decode.py (line 95)
    # Input shape: [1, batch, padded_heads, head_dim]
    # Output shape: [1, 1, batch, head_dim * num_heads]
    # The operation takes first num_heads from padded_heads dimension and concatenates them

    if len(shape) == 4:
        _, batch, padded_heads, head_dim = shape
        # Take first num_heads from the padded_heads dimension and reshape
        # Input: (1, batch, padded_heads, head_dim) -> Output: (1, 1, batch, head_dim * num_heads)
        torch_output_tensor = torch_input_tensor_a[:, :, :num_heads, :].reshape(1, 1, batch, head_dim * num_heads)
    else:
        torch_output_tensor = torch_input_tensor_a.clone()

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            # The input tensor shape is (1, batch, num_heads, head_dim) per-device.
            # The master model creates it per-device (replicated) so logical_shape()
            # returns the per-device shape with dim[2]=num_heads.
            # create_tensor_on_mesh would expand+shard, causing logical_shape() to
            # return the global shape (dim[2]=num_heads*mesh_factor), mismatching
            # the master trace.  Replicate to preserve the per-device shape.
            input_tensor_a = replicate_with_topology(
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
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    output_tensor = ttnn.experimental.nlp_concat_heads_decode(input_tensor_a, num_heads=num_heads, **op_kwargs)
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    # Unpad the output - TTNN output may be padded to tile size (32)
    # We need to extract only the actual batch size
    if len(shape) == 4:
        _, batch, _, _ = shape
        output_tensor = output_tensor[:, :, :batch, :]

    # Check with PCC - using standard threshold
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    return [pcc, e2e_perf]
