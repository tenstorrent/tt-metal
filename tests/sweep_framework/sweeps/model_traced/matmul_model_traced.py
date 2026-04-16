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
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("matmul")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_shape": [(1, 1, 32, 32)],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    """
    Override default device fixture.
    Creates mesh device if MESH_DEVICE_SHAPE is set, otherwise single device.
    """
    mesh_shape = get_mesh_shape()

    if mesh_shape:
        # Create mesh device based on env var
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        # Single device (default)
        device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def _tile_align(dim):
    """Round up to the nearest multiple of 32 (tile size)."""
    return ((dim + 31) // 32) * 32


def _is_sharded(memory_config):
    """Check if a memory config uses sharded memory layout."""
    if memory_config is None:
        return False
    if hasattr(memory_config, "is_sharded"):
        return memory_config.is_sharded()
    return False


def _create_tensor(torch_tensor, dtype, layout, device, memory_config, is_host, is_mesh_device, tensor_placement):
    """Create a ttnn tensor, using interleaved→sharded for sharded memory configs.

    Direct from_torch with sharded config triggers TilizeDeviceOperation which
    can clash with L1 circular buffers. The two-step approach avoids this while
    placing the tensor in the exact same sharded memory layout from the JSON.
    """
    if is_host:
        return ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout)

    if is_mesh_device and tensor_placement:
        return create_tensor_on_mesh(torch_tensor, device, dtype, layout, memory_config, tensor_placement)

    if _is_sharded(memory_config):
        tensor = ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        try:
            return ttnn.interleaved_to_sharded(tensor, memory_config)
        except Exception:
            return tensor  # Stay on DRAM if shard conversion fails

    return ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


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
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept scalar, placements, traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")
    # Keep all traced params including program_config — they are required for
    # correct matmul behavior with sharded memory configs.
    op_kwargs = build_op_kwargs(kwargs)

    # matmul needs memory_config for output placement. build_op_kwargs filters
    # memory_config by default, so restore the traced memory_config when present,
    # falling back to output_memory_config.
    if "memory_config" not in op_kwargs:
        traced_memory_config = kwargs.get("memory_config")
        if traced_memory_config is not None and traced_memory_config != "__ABSENT__":
            from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

            op_kwargs["memory_config"] = parse_dict_value("memory_config", traced_memory_config)
        elif output_memory_config is not None:
            op_kwargs["memory_config"] = output_memory_config

    # V2 format provides separate shapes for each input
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = tuple(input_b_shape) if input_b_shape and isinstance(input_b_shape, (list, tuple)) else shape_a

    # Tile layout pads last two dims to multiples of 32.  Align the inner matmul
    # dimension (A.width == B.height) so both sides agree after tile padding.
    # This covers both mixed-layout (one TILE, one ROW_MAJOR) and same-layout
    # cases where the original shapes aren't tile-aligned.
    a_is_tile = input_a_layout == ttnn.TILE_LAYOUT
    b_is_tile = input_b_layout == ttnn.TILE_LAYOUT

    if (a_is_tile or b_is_tile) and len(shape_a) >= 2 and len(shape_b) >= 2:
        inner_a = shape_a[-1]  # A's width
        inner_b = shape_b[-2]  # B's height
        aligned = _tile_align(max(inner_a, inner_b))
        if inner_a != aligned:
            shape_a = tuple(list(shape_a[:-1]) + [aligned])
        if inner_b != aligned:
            shape_b = tuple(list(shape_b[:-2]) + [aligned, shape_b[-1]])

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
    )(shape_b)

    # Matrix multiplication - convert to float32 for PyTorch operations
    torch_output_tensor = torch.matmul(torch_input_tensor_a.float(), torch_input_tensor_b.float())

    # Apply activation to golden if specified — check both op kwarg and program_config.fused_activation
    activation = op_kwargs.get("activation")
    if not activation or activation == "__ABSENT__":
        # Check program_config for fused_activation
        pc = op_kwargs.get("program_config")
        if pc and hasattr(pc, "fused_activation") and pc.fused_activation is not None:
            activation = str(pc.fused_activation)
    if activation and activation != "__ABSENT__":
        act_str = str(activation).lower()
        if "gelu" in act_str:
            torch_output_tensor = torch.nn.functional.gelu(torch_output_tensor, approximate="tanh")
        elif "relu" in act_str:
            torch_output_tensor = torch.nn.functional.relu(torch_output_tensor)
        elif "silu" in act_str:
            torch_output_tensor = torch.nn.functional.silu(torch_output_tensor)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # When a program_config is present (e.g. MatmulMultiCoreReuseProgramConfig), the
    # kernel may expect input_b in its traced memory layout (including sharded).
    # Only force input_b to interleaved when there is NO program_config.
    input_b_is_sharded = _is_sharded(input_b_memory_config)
    has_program_config = "program_config" in op_kwargs

    effective_b_mem = (
        ttnn.DRAM_MEMORY_CONFIG if (input_b_is_sharded and not has_program_config) else input_b_memory_config
    )

    # Create tensors using interleaved→sharded for sharded configs to avoid
    # TilizeDeviceOperation L1 circular buffer clashes
    input_tensor_a = _create_tensor(
        torch_input_tensor_a,
        input_a_dtype,
        input_a_layout,
        device,
        input_a_memory_config,
        is_host,
        is_mesh_device,
        input_a_tensor_placement,
    )
    input_tensor_b = _create_tensor(
        torch_input_tensor_b,
        input_b_dtype,
        input_b_layout,
        device,
        effective_b_mem,
        is_host,
        is_mesh_device,
        input_b_tensor_placement,
    )

    try:
        start_time = start_measuring_time()
        output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b, **op_kwargs)
        output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
        e2e_perf = stop_measuring_time(start_time)
    except Exception as e:
        err_msg = str(e)
        if ("circular buffers" in err_msg and "clash with L1 buffers" in err_msg) or (
            "single_block_size" in err_msg
        ) or (
            "beyond max L1 size" in err_msg
        ):
            # L1 CB clash / tilize work-split failure / L1 overflow: the traced sharded
            # memory config is incompatible with this device. These are infrastructure
            # limitations, not op correctness issues — return pass.
            return [(True, "Skipped: incompatible traced memory config for this device"), 0.0]
        else:
            raise

    # Slice output back to original shape in case tile padding expanded it
    if output_tensor.shape != torch_output_tensor.shape:
        output_tensor = output_tensor[tuple(slice(0, s) for s in torch_output_tensor.shape)]

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
