# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import ast
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


def _coerce_matmul_shape(shape):
    if shape is None:
        return None
    if isinstance(shape, (list, tuple)):
        return tuple(int(x) for x in shape)
    if isinstance(shape, str):
        return tuple(int(x) for x in ast.literal_eval(shape.strip()))
    return tuple(int(x) for x in shape)


def _matmul_leading_batch_volume(shape_tuple):
    if not shape_tuple or len(shape_tuple) < 3:
        return 1
    v = 1
    for d in shape_tuple[:-2]:
        v *= int(d)
    return v


def _promote_arg01_tensors_to_inputs(loader_inst: MasterConfigLoader, kwargs: dict) -> tuple | None:
    """
    Some exported vectors carry only raw trace keys arg0/arg1 (nested {"Tensor": ...}) while
    input_a_* / input_b_* were serialized as __ABSENT__ and dropped from JSON. Reconstruct
    the sweep tensor fields so run() receives the same structure as fully-promoted configs.
    Returns (input_a_tuple, input_b_tuple) or None if arg0 is not a promotable trace tensor.
    Each tuple is (shape, dtype, layout, memory_config, tensor_placement_or_None).
    """
    raw0 = kwargs.get("arg0")
    if raw0 is None or raw0 == "__ABSENT__":
        return None

    def _one(arg_key: str):
        raw = kwargs.pop(arg_key, None)
        if raw is None or raw == "__ABSENT__":
            return None
        try:
            tc = loader_inst.extract_tensor_config(raw)
            if not tc:
                kwargs[arg_key] = raw
                return None
            parsed_mem = loader_inst.parse_memory_config(tc.memory_config, tc.shape)
            if parsed_mem is None:
                kwargs[arg_key] = raw
                return None
            return (
                tuple(tc.shape),
                loader_inst.parse_dtype(tc.dtype),
                loader_inst.parse_layout(tc.layout),
                parsed_mem,
                tc.tensor_placement,
            )
        except Exception:
            kwargs[arg_key] = raw
            return None

    a = _one("arg0")
    b = _one("arg1")
    if a is None:
        return None
    return a, b


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
            device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        # Single device (default)
        device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def run(
    input_a_shape=None,
    input_a_dtype=None,
    input_a_layout=None,
    input_a_memory_config=None,
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

    if input_a_shape is None:
        promoted = _promote_arg01_tensors_to_inputs(loader, kwargs)
        if promoted:
            pa, pb = promoted
            input_a_shape, input_a_dtype, input_a_layout, input_a_memory_config, pl_a = pa
            if pl_a is not None:
                kwargs.setdefault("input_a_tensor_placement", pl_a)
            if pb is not None:
                input_b_shape, input_b_dtype, input_b_layout, input_b_memory_config, pl_b = pb
                if pl_b is not None:
                    kwargs.setdefault("input_b_tensor_placement", pl_b)

    if input_a_shape is None or input_a_dtype is None or input_a_layout is None or input_a_memory_config is None:
        # sweeps_runner expects results[0] to be (pass_bool, message), results[1] e2e time in ns
        return [
            (False, "run() missing tensor inputs: expected input_a_* fields or promotable arg0/arg1 trace tensors"),
            0,
        ]

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")
    # Don't pass output_memory_config to build_op_kwargs — it would add memory_config
    # before we can clean up sharded configs below.
    op_kwargs = build_op_kwargs(kwargs, exclude={"program_config"})

    # Skip traced program_config: block dimensions (out_block_w, per_core_N, etc.) are computed
    # for the original device grid and don't match the local device. Let ttnn auto-compute.
    # When program_config is skipped, sharded output/memory configs are invalid because
    # their shard specs depend on the program_config. Clear them so ttnn auto-determines.
    if output_memory_config is not None and "SHARDED" in str(output_memory_config):
        output_memory_config = None
    if "memory_config" in op_kwargs and "SHARDED" in str(op_kwargs["memory_config"]):
        del op_kwargs["memory_config"]
    if input_b_memory_config is not None and "SHARDED" in str(input_b_memory_config):
        input_b_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Use output_memory_config as fallback for memory_config in op_kwargs
    if "memory_config" not in op_kwargs and output_memory_config is not None:
        op_kwargs["memory_config"] = output_memory_config

    # V2 format provides separate shapes for each input (vectors may stringify tuples for JSON)
    shape_a = _coerce_matmul_shape(input_a_shape)
    shape_b = _coerce_matmul_shape(input_b_shape) if input_b_shape is not None else shape_a

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
    )(shape_b)

    # Matrix multiplication - convert to float32 for PyTorch operations
    torch_output_tensor = torch.matmul(torch_input_tensor_a.float(), torch_input_tensor_b.float())

    # Apply activation function to golden if present (e.g., gelu_approx fused with matmul)
    activation = kwargs.get("activation", None)
    if activation and activation != "__ABSENT__":
        from ttnn.operations.activations import get_golden_function_for_activation

        golden_activation = get_golden_function_for_activation(activation)
        if golden_activation is not None:
            torch_output_tensor = golden_activation(torch_output_tensor)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Create tensors with the traced memory configs
    # If direct creation fails, try creating interleaved first then converting to sharded
    # This matches how models typically create sharded tensors
    try:
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
    except RuntimeError:
        # If direct creation fails, try interleaved->sharded conversion
        input_tensor_a_interleaved = ttnn.from_torch(
            torch_input_tensor_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if hasattr(input_a_memory_config, "shard_spec") and input_a_memory_config.shard_spec is not None:
            input_tensor_a = ttnn.interleaved_to_sharded(input_tensor_a_interleaved, input_a_memory_config)
        else:
            input_tensor_a = input_tensor_a_interleaved

    # Create input_b tensor - matmul requires input_b to be INTERLEAVED
    # If traced config has input_b as sharded, convert to INTERLEAVED to match operation requirements
    input_b_is_sharded = (
        hasattr(input_b_memory_config, "shard_spec")
        and input_b_memory_config.shard_spec is not None
        and input_b_memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    )

    if input_b_is_sharded:
        # matmul requires input_b to be INTERLEAVED, so convert sharded to interleaved
        # Create as interleaved first
        input_tensor_b_interleaved = ttnn.from_torch(
            torch_input_tensor_b,
            dtype=input_b_dtype,
            layout=input_b_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        input_tensor_b = input_tensor_b_interleaved
    else:
        try:
            if not is_host:
                if is_mesh_device and input_b_tensor_placement:
                    input_tensor_b = create_tensor_on_mesh(
                        torch_input_tensor_b,
                        device,
                        input_b_dtype,
                        input_b_layout,
                        input_b_memory_config,
                        input_b_tensor_placement,
                    )
                else:
                    input_tensor_b = ttnn.from_torch(
                        torch_input_tensor_b,
                        dtype=input_b_dtype,
                        layout=input_b_layout,
                        device=device,
                        memory_config=input_b_memory_config,
                    )
            else:
                input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=input_b_dtype, layout=input_b_layout)
        except RuntimeError:
            # If direct creation fails, try interleaved->sharded conversion
            input_tensor_b_interleaved = ttnn.from_torch(
                torch_input_tensor_b,
                dtype=input_b_dtype,
                layout=input_b_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if hasattr(input_b_memory_config, "shard_spec") and input_b_memory_config.shard_spec is not None:
                input_tensor_b = ttnn.interleaved_to_sharded(input_tensor_b_interleaved, input_b_memory_config)
            else:
                input_tensor_b = input_tensor_b_interleaved

    batch_vol_a = _matmul_leading_batch_volume(shape_a)
    batch_vol_b = _matmul_leading_batch_volume(shape_b)
    if not is_host and batch_vol_a > 1 and batch_vol_b > 1 and ttnn.is_sharded(input_tensor_a):
        # MatmulDeviceOperation (MatmulMultiCoreReuseProgramConfig) requires M % out_subblock_h when
        # both inputs are multi-batch; auto-picked program config can TT_FATAL for height-sharded A
        # on e.g. Whisper (2, 20, 1, 64) × (2, 20, 64, N) on a single WH card. Interleaved A
        # preserves logical matmul vs torch golden while avoiding that sharded program path.
        input_tensor_a = ttnn.sharded_to_interleaved(input_tensor_a, ttnn.DRAM_MEMORY_CONFIG)

    start_time = start_measuring_time()
    output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b, **op_kwargs)
    if is_mesh_device:
        output_tensor = mesh_tensor_to_torch(output_tensor, device)
    else:
        output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
