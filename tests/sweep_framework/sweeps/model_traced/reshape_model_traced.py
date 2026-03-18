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
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("reshape")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "target_shape": [(1, 32, 1, 32)],
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
    target_shape=None,
    shape=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"arg1", "arg2"}, output_memory_config=output_memory_config)

    # v2 tracer puts target shape in arg1 or shape; arg2 may hold a
    # secondary shape (e.g. padded output shape) used by some internal paths.
    # Filter out "__ABSENT__" sentinel values from the V2 loader.
    def _clean_absent(val):
        """Return None if val is the __ABSENT__ sentinel."""
        return None if val == "__ABSENT__" else val

    tgt_shape = _clean_absent(target_shape) or _clean_absent(shape) or _clean_absent(kwargs.get("arg1", None))
    if tgt_shape is None:
        tgt_shape = (1, 32, 1, 32)  # fallback for sample

    if isinstance(tgt_shape, list):
        tgt_shape = tuple(tgt_shape)
    elif isinstance(tgt_shape, dict) and "value" in tgt_shape:
        import re

        m = re.search(r"\[([0-9, ]+)\]", str(tgt_shape["value"]))
        if m:
            tgt_shape = tuple(int(x) for x in m.group(1).split(","))
    elif isinstance(tgt_shape, dict):
        # Unrecognized dict format -- try to extract shape from any string repr
        import re

        m = re.search(r"\[([0-9, ]+)\]", str(tgt_shape))
        if m:
            tgt_shape = tuple(int(x) for x in m.group(1).split(","))

    # arg2 may be a padded output shape; extract if present
    arg2 = _clean_absent(kwargs.get("arg2", None))
    if arg2 is not None and isinstance(arg2, dict) and "value" in arg2:
        import re

        m = re.search(r"\[([0-9, ]+)\]", str(arg2["value"]))
        if m:
            arg2 = tuple(int(x) for x in m.group(1).split(","))
        else:
            arg2 = None  # Could not parse dict-style arg2
    elif arg2 is not None and isinstance(arg2, dict):
        import re

        m = re.search(r"\[([0-9, ]+)\]", str(arg2))
        if m:
            arg2 = tuple(int(x) for x in m.group(1).split(","))
        else:
            arg2 = None  # Could not parse dict-style arg2
    if isinstance(arg2, list):
        arg2 = tuple(arg2)
    # Final guard: arg2 must be a tuple of ints if present
    if arg2 is not None and not isinstance(arg2, tuple):
        arg2 = None

    in_shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        in_shape
    )

    import math

    input_numel = math.prod(in_shape)
    tgt_numel = math.prod(tgt_shape)
    has_padded_shape = tgt_numel != input_numel and arg2 is not None and math.prod(arg2) == input_numel
    if has_padded_shape:
        torch_output = torch.reshape(torch_input, arg2)
        slices = tuple(slice(0, s) for s in tgt_shape)
        torch_output = torch_output[slices]
    else:
        torch_output = torch.reshape(torch_input, tgt_shape)

    is_host = storage_type and "HOST" in str(storage_type)

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

    start_time = start_measuring_time()
    if tgt_numel != input_numel and arg2 is not None:
        output_tensor = ttnn.reshape(input_tensor, tgt_shape, arg2, **op_kwargs)
    else:
        output_tensor = ttnn.reshape(input_tensor, tgt_shape, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output, output_tensor, 0.999)
    return [pcc, e2e_perf]
