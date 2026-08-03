# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("max")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
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
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # Handle tuple input_a_shape for sample suite
    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Build PyTorch reference matching the traced op's dim/keepdim parameters.
    # The traced configs pass dim and keepdim to ttnn.max, so the PyTorch reference
    # must use the same parameters to produce matching output shapes.
    reduce_dim = op_kwargs.get("dim", None)
    keepdim = op_kwargs.get("keepdim", False)
    if reduce_dim is not None:
        torch_output_tensor = torch.max(torch_input_tensor_a, dim=reduce_dim, keepdim=keepdim)
        # torch.max with dim returns (values, indices); we only need values
        if isinstance(torch_output_tensor, tuple):
            torch_output_tensor = torch_output_tensor[0]
    else:
        torch_output_tensor = torch.max(torch_input_tensor_a)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

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
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    output_tensor = ttnn.max(input_tensor_a, **op_kwargs)
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, input_a_tensor_placement)

    if torch_output_tensor.numel() == 1 and output_tensor.numel() == 1:
        # A global reduce (no `dim`) returns a scalar, and correlation is undefined for one
        # element -- zero variance makes the denominator 0, so comp_pcc gets NaN and falls back
        # to torch.allclose with a fixed rtol=1e-5 that has nothing to do with the 0.999 asked
        # for here. That rejected a result accurate to 6.2e-5 (better than bfloat16) and
        # reported it as PCC exactly 0.0 -- e.g. golden 3.334923 vs ttnn 3.334717 on shapes
        # (16, 2, 32, {3,6,12,24}). Compare by relative error against the requested threshold
        # instead, and report the error itself rather than a fabricated correlation value.
        #
        # This tolerance absorbs the op's own precision error. On float32 that error is real and
        # under review: ttnn.max returns a value not present in the input at all (issue #51889).
        # If that is confirmed a defect these configs should fail again -- on selection
        # semantics, not on a made-up PCC.
        golden_value = torch_output_tensor.flatten().float().item()
        actual_value = output_tensor.flatten().float().item()
        rel_err = abs(golden_value - actual_value) / max(abs(golden_value), 1e-30)
        pcc = (
            rel_err <= (1.0 - 0.999),
            f"scalar result: rel err {rel_err:.3e} vs tol {1.0 - 0.999:.1e} "
            f"(golden {golden_value}, actual {actual_value}; PCC undefined for 1 element)",
        )
    else:
        pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
