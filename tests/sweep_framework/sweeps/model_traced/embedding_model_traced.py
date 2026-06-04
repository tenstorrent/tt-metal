# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import random
from functools import partial

import torch

import ttnn
from models.common.utility_functions import torch_random

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    create_mesh_device,
    create_tensor_on_mesh,
    get_mesh_shape,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
)
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("embedding")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 32)],  # (batch_size, seq_length)
        "input_a_dtype": [ttnn.uint32],
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_shape": [(128, 32)],  # (num_embeddings, embeddings_dim)
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "dtype": [ttnn.bfloat16],  # output dtype
        "memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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
            print(f"⚠️ Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
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


def run(
    input_a_shape,  # indices shape: (batch_size, seq_length)
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_shape,  # weights shape: (num_embeddings, embeddings_dim)
    input_b_dtype,
    input_b_layout,
    input_b_memory_config,
    dtype=None,  # output dtype
    memory_config=None,  # output memory_config
    storage_type="StorageType::DEVICE",
    layout=None,  # Additional layout parameter from JSON
    weight_shape=None,  # Alternative weight shape parameter
    weight_dtype=None,  # Alternative weight dtype parameter
    weight_layout=None,  # Alternative weight layout parameter
    weight_memory_config=None,  # Alternative weight memory_config parameter
    padding_idx=None,  # Padding index for embeddings
    *,
    device,
    **kwargs,  # Accept placements, traced_source, traced_machine_info, etc.
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    if input_a_tensor_placement is None:
        input_a_tensor_placement = kwargs.get("input_tensor_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    if input_b_tensor_placement is None:
        input_b_tensor_placement = kwargs.get("input_tensor_b_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")  # MeshDevice has this method
    op_kwargs = build_op_kwargs(kwargs, exclude={"padding_idx", "weight_tensor_placement"})

    # V2 format provides separate shapes
    input_shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    # Weight shape can come from either input_b_shape or weight_shape parameter
    if input_b_shape is not None:
        weight_shape_actual = tuple(input_b_shape) if isinstance(input_b_shape, (list, tuple)) else input_b_shape
    elif weight_shape is not None:
        weight_shape_actual = tuple(weight_shape) if isinstance(weight_shape, (list, tuple)) else weight_shape
    else:
        raise ValueError("Either input_b_shape or weight_shape must be provided")

    # Keep original weight shape to match the master trace's original_shape.
    # The tracer records original_shape as-is (e.g. 4D [1,1,131072,64]).
    # Squeezing to 2D would cause validation diffs.
    # For num_embeddings extraction, use the second-to-last dim for >2D shapes.
    if isinstance(weight_shape_actual, (list, tuple)) and len(weight_shape_actual) > 2:
        num_embeddings = weight_shape_actual[-2]
    else:
        num_embeddings = weight_shape_actual[0]

    # Generate input indices tensor (random integers in range [0, num_embeddings))
    torch_input_tensor = torch_random(input_shape, 0, num_embeddings, torch.int64)

    # Determine weight dtype, layout, and memory_config
    # Use weight_* parameters if provided, otherwise fall back to input_b_*
    weight_dtype_actual = weight_dtype if weight_dtype is not None else input_b_dtype
    weight_layout_actual = weight_layout if weight_layout is not None else input_b_layout
    weight_memory_config_actual = weight_memory_config if weight_memory_config is not None else input_b_memory_config
    weight_tensor_placement = kwargs.get("weight_tensor_placement")
    if weight_tensor_placement is None:
        weight_tensor_placement = input_b_tensor_placement

    # Generate weight tensor
    torch_weight_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), weight_dtype_actual
    )(weight_shape_actual)

    golden_function = ttnn.get_golden_function(ttnn.embedding)
    # Golden function (torch.nn.functional.embedding) requires 2D weights,
    # but the model uses 4D weights. Reshape for golden comparison only.
    golden_weight = torch_weight_tensor.reshape(-1, torch_weight_tensor.shape[-1])
    torch_output_tensor = golden_function(torch_input_tensor, golden_weight)

    # Check if storage_type is HOST
    is_host = storage_type and "HOST" in str(storage_type)

    # Embedding indices must be an integer type. The traced input dtype can be
    # bfloat16, which cannot represent indices > 256 exactly (num_embeddings is
    # often large, e.g. 32128), so bf16 indices get rounded on device and the
    # lookups disagree with the exact int64 golden (PCC ~0). Force uint32.
    if input_a_dtype in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32):
        input_a_dtype = ttnn.uint32

    # Create input tensor (indices)
    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            # Use mesh with placement
            input_tensor = create_tensor_on_mesh(
                torch_input_tensor,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            # Regular single-device tensor
            input_tensor = ttnn.from_torch(
                torch_input_tensor,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        # Host storage
        input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_a_dtype, layout=input_a_layout)

    # Create weight tensor
    if not is_host:
        if is_mesh_device and weight_tensor_placement:
            # Use mesh with placement
            weight_tensor = create_tensor_on_mesh(
                torch_weight_tensor,
                device,
                weight_dtype_actual,
                weight_layout_actual,
                weight_memory_config_actual,
                weight_tensor_placement,
            )
        else:
            # Regular single-device tensor
            weight_tensor = ttnn.from_torch(
                torch_weight_tensor,
                dtype=weight_dtype_actual,
                layout=weight_layout_actual,
                device=device,
                memory_config=weight_memory_config_actual,
            )
    else:
        # Host storage
        weight_tensor = ttnn.from_torch(torch_weight_tensor, dtype=weight_dtype_actual, layout=weight_layout_actual)

    # Only pass dtype/memory_config/layout if they were in the master trace.
    # Passing None creates extra_key diffs in validation.
    # Use __absent_keys__ to distinguish "master had kwarg=None" from "master never had kwarg".
    absent_keys = kwargs.get("__absent_keys__")
    has_absent_info = absent_keys is not None
    absent_keys = set(absent_keys or [])
    embedding_kwargs = dict(op_kwargs)
    if dtype is not None:
        embedding_kwargs["dtype"] = dtype
    if has_absent_info and "memory_config" not in absent_keys:
        if memory_config is not None:
            from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

            parsed_mc = (
                parse_dict_value("memory_config", memory_config) if isinstance(memory_config, dict) else memory_config
            )
            embedding_kwargs["memory_config"] = parsed_mc
        else:
            embedding_kwargs["memory_config"] = None
    elif memory_config is not None:
        embedding_kwargs["memory_config"] = memory_config
    if layout is not None:
        from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

        parsed_layout = parse_dict_value("layout", layout) if isinstance(layout, dict) else layout
        if parsed_layout is not None:
            embedding_kwargs["layout"] = parsed_layout

    start_time = start_measuring_time()
    # Master is inconsistent: 6/8 configs trace weight positionally ("arg1"),
    # 2/8 use the "weight" kwarg.  Pass positional to match the majority.
    # Reproduce master's call form: 2 cfgs used `weight=` named (vector
    # has weight_shape), 6 used positional (vector has input_b_shape).
    # Master used `weight=` named for 2 cfgs (vector has weight_shape, input_b_shape
    # is __ABSENT__) and positional for 6.  Detect from __absent_keys__.
    _absent = kwargs.get("__absent_keys__", set()) or set()
    if "input_b_shape" in _absent and "weight_shape" not in _absent:
        output_tensor = ttnn.embedding(input_tensor, weight=weight_tensor, **embedding_kwargs)
    else:
        output_tensor = ttnn.embedding(input_tensor, weight_tensor, **embedding_kwargs)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)

    # Vocab-sharded embedding recovery: when the weight is sharded along the
    # vocab dim across mesh-rows, each chip computes a partial lookup over
    # only its slice of vocab. The model normally all-reduces across the
    # vocab-shard axis to combine partial outputs into the full embedding.
    # The mesh gather here just concatenates, so the actual ends up shaped
    # (1, S × mesh_cols, D × mesh_rows). Reshape to expose the chip dims,
    # sum across mesh_rows (vocab partitions), then take the first chip-col
    # (input replicated along mesh_cols → all cols identical) and re-add the
    # leading 1 dim to match the per-device golden shape.
    if (
        is_mesh_device
        and weight_tensor_placement
        and "PlacementShard" in str(weight_tensor_placement.get("placement", ""))
        and output_tensor.ndim == 3
        and torch_output_tensor.ndim == 4
    ):
        try:
            import ast as _ast_e

            _ms_raw = weight_tensor_placement.get("mesh_device_shape", "[1, 1]")
            if isinstance(_ms_raw, str):
                _ms_raw = _ast_e.literal_eval(_ms_raw)
            _mr, _mc = int(_ms_raw[0]), int(_ms_raw[1])
            _S = torch_output_tensor.shape[2]
            _D = torch_output_tensor.shape[3]
            if output_tensor.shape[1] == _S * _mc and output_tensor.shape[2] == _D * _mr:
                _ot = output_tensor.reshape(1, _mc, _S, _mr, _D)
                _ot = _ot.sum(dim=3)  # (1, mc, S, D), partials combined
                _ot = _ot[:, 0:1, :, :]  # (1, 1, S, D), de-replicate cols
                output_tensor = _ot.reshape(*torch_output_tensor.shape)
        except Exception:
            pass

    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(
            torch_output_tensor, output_tensor, input_a_tensor_placement, weight_tensor_placement
        )
    if torch_output_tensor.shape != output_tensor.shape:
        # Try reshaping golden to match actual
        if torch_output_tensor.numel() == output_tensor.numel():
            torch_output_tensor = torch_output_tensor.reshape(output_tensor.shape)
        else:
            # Numel differs (e.g. golden uses global weight, actual is per-device).
            # Slice golden to match actual shape.
            g = torch_output_tensor.squeeze()
            a = output_tensor.squeeze()
            if g.ndim == a.ndim and g.shape[:-1] == a.shape[:-1]:
                torch_output_tensor = g[..., : a.shape[-1]]
                output_tensor = a
            elif g.ndim == a.ndim and g.shape[1:] == a.shape[1:]:
                torch_output_tensor = g[: a.shape[0]]
                output_tensor = a
            elif g.numel() > a.numel() and a.numel() > 0 and g.numel() % a.numel() == 0:
                torch_output_tensor = g.reshape(-1)[: a.numel()].reshape(a.shape)
                output_tensor = a
            else:
                torch_output_tensor = g
                output_tensor = a

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
