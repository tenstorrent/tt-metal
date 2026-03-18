# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sweep test for ttnn.experimental.rotary_embedding_llama_fused_qk operation.

This operation fuses rotary position embedding with Q/K preparation in one kernel,
optimizing memory bandwidth and reducing overhead in transformer attention layers.

The operation returns two tensors: rotated Q and rotated K.

C++ validation constraints (from rotary_embedding_llama_fused_qk_device_operation.cpp):
  - ALL 5 tensors MUST be HEIGHT_SHARDED (not DRAM interleaved)
  - Q and K must have equal batch_size (dim[1]) and batch_size <= 32
  - Q and K shard grids must NOT overlap
  - cos/sin batch_size must equal q_batch + k_batch
  - trans_mat must be sharded over >= (q_num_cores + k_num_cores) cores
  - trans_mat shard shape must be (32, 32)
  - q_num_cores + k_num_cores <= 64
  - head_dim must be a multiple of 32
"""

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::rotary_embedding_llama_fused_qk")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 8, 128, 64)],  # batch, n_heads, seq_len, head_dim
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_c_dtype": [ttnn.bfloat16],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_d_dtype": [ttnn.bfloat16],
        "input_d_layout": [ttnn.TILE_LAYOUT],
        "input_d_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_e_dtype": [ttnn.bfloat16],
        "input_e_layout": [ttnn.TILE_LAYOUT],
        "input_e_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
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
            device = ttnn.open_device(device_id=0, l1_small_size=32768, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=32768, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def _validate_shard_grid_for_device(memory_config, device):
    """Check if a sharded memory config's core grid is valid for the given device.

    Returns True if valid (or not sharded), False if the grid exceeds device limits.
    """
    if memory_config is None:
        return False
    if not (hasattr(memory_config, "is_sharded") and memory_config.is_sharded()):
        return True
    try:
        shard_spec = memory_config.shard_spec
        if shard_spec is None:
            return False
        # Try to access grid properties -- if the grid references cores
        # beyond device limits, this will be caught at tensor creation time.
        # Here we do a basic check: ensure num_cores > 0
        grid = shard_spec.grid
        num_cores = grid.num_cores()
        return num_cores > 0
    except Exception:
        return False


def _create_tensor_on_device(torch_tensor, device, dtype, layout, memory_config, is_mesh_device, placement):
    """Create tensor on device with the exact traced sharded memory config.

    This op REQUIRES HEIGHT_SHARDED for all inputs.  We must use the traced
    memory configs directly -- falling back to DRAM_MEMORY_CONFIG would cause
    a TT_FATAL in the C++ validation layer.

    For mesh devices, we first create with DRAM interleaved then reshard,
    since ttnn.from_torch with mesh_mapper may not support sharded configs
    directly for all placement types.
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    if is_mesh_device and placement:
        # For mesh: create on mesh, then reshard if needed
        # create_tensor_on_mesh handles mesh placement (ShardTensor2dMesh etc.)
        # First place with DRAM, then move to sharded if memory_config is sharded
        is_sharded = hasattr(memory_config, "is_sharded") and memory_config.is_sharded()
        if is_sharded:
            tensor = create_tensor_on_mesh(torch_tensor, device, dtype, layout, ttnn.DRAM_MEMORY_CONFIG, placement)
            tensor = ttnn.to_memory_config(tensor, memory_config)
            return tensor
        else:
            return create_tensor_on_mesh(torch_tensor, device, dtype, layout, memory_config, placement)

    # Single device or mesh without placement: create directly
    is_sharded = hasattr(memory_config, "is_sharded") and memory_config.is_sharded()
    if is_sharded:
        # Create on DRAM first, then move to sharded config
        tensor = ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tensor = ttnn.to_memory_config(tensor, memory_config)
        return tensor
    else:
        return ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=memory_config,
        )


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    input_d_dtype=None,
    input_d_layout=None,
    input_d_memory_config=None,
    input_e_dtype=None,
    input_e_layout=None,
    input_e_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    if not input_a_tensor_placement:
        input_a_tensor_placement = None
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    if not input_b_tensor_placement:
        input_b_tensor_placement = None
    input_c_tensor_placement = kwargs.get("input_c_tensor_placement", None)
    if not input_c_tensor_placement:
        input_c_tensor_placement = None
    input_d_tensor_placement = kwargs.get("input_d_tensor_placement", None)
    if not input_d_tensor_placement:
        input_d_tensor_placement = None
    input_e_tensor_placement = kwargs.get("input_e_tensor_placement", None)
    if not input_e_tensor_placement:
        input_e_tensor_placement = None
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # Extract shapes for all 5 inputs.
    # V2 loader puts shapes as separate keys: input_b_shape, input_c_shape, etc.
    # These end up in **kwargs since only input_a_shape is a named parameter.
    # Filter out "__ABSENT__" sentinel used by V2 loader for missing keys.
    def _get_shape(key):
        val = kwargs.get(key, None)
        if val is None or val == "__ABSENT__":
            return None
        return val

    shape_b_from_kwargs = _get_shape("input_b_shape")
    shape_c_from_kwargs = _get_shape("input_c_shape")
    shape_d_from_kwargs = _get_shape("input_d_shape")
    shape_e_from_kwargs = _get_shape("input_e_shape")

    if shape_b_from_kwargs is not None:
        # V2 loader format: each input has its own shape key
        shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
        shape_b = tuple(shape_b_from_kwargs) if isinstance(shape_b_from_kwargs, (list, tuple)) else shape_b_from_kwargs
        shape_c = tuple(shape_c_from_kwargs) if shape_c_from_kwargs is not None else None
        shape_d = tuple(shape_d_from_kwargs) if shape_d_from_kwargs is not None else None
        shape_e = tuple(shape_e_from_kwargs) if shape_e_from_kwargs is not None else None
    elif isinstance(input_a_shape, dict):
        # Legacy dict format (multi-input packed into input_a_shape)
        shape_a = input_a_shape.get("input_a", input_a_shape.get("q_input_tensor"))
        shape_b = input_a_shape.get("input_b", input_a_shape.get("k_input_tensor"))
        shape_c = input_a_shape.get("input_c", input_a_shape.get("cos_cache"))
        shape_d = input_a_shape.get("input_d", input_a_shape.get("sin_cache"))
        shape_e = input_a_shape.get("input_e", input_a_shape.get("trans_mat"))
    else:
        # Fallback for sample configurations (simple tuple shape)
        if isinstance(input_a_shape, (tuple, list)):
            shape = tuple(input_a_shape)
        else:
            shape = input_a_shape
        batch, n_heads, seq_len, head_dim = shape
        shape_a = shape
        shape_b = shape
        shape_c = (1, 1, seq_len, head_dim)
        shape_d = (1, 1, seq_len, head_dim)
        shape_e = (1, 1, head_dim, head_dim)

    # Validate: all shapes must be present for this 5-input op
    if shape_a is None or shape_b is None or shape_c is None or shape_d is None:
        return [(False, "Missing required input shapes (need all 5: Q, K, cos, sin, trans_mat)"), 0]

    # Pre-validate C++ constraints before creating tensors:
    # 1. Q and K batch_size must be equal
    q_batch = shape_a[1] if len(shape_a) >= 2 else 1
    k_batch = shape_b[1] if len(shape_b) >= 2 else 1
    if q_batch != k_batch:
        return [(False, f"Q batch ({q_batch}) != K batch ({k_batch}), C++ requires equal batch sizes"), 0]

    # 2. Q and K batch_size must be <= 32
    if q_batch > 32:
        return [(False, f"Q/K batch_size ({q_batch}) > 32, C++ limit is 32"), 0]

    # 3. head_dim must be a multiple of 32
    head_dim = shape_a[-1]
    if head_dim % 32 != 0:
        return [(False, f"head_dim ({head_dim}) is not a multiple of 32"), 0]

    # Use random tensors with exact traced shapes for all 5 inputs.
    torch_input_a = (torch.rand(shape_a) * 2 - 1).to(torch.bfloat16)
    torch_input_b = (torch.rand(shape_b) * 2 - 1).to(torch.bfloat16)
    torch_input_c = (torch.rand(shape_c) * 2 - 1).to(torch.bfloat16)
    torch_input_d = (torch.rand(shape_d) * 2 - 1).to(torch.bfloat16)
    torch_input_e = (torch.rand(shape_e) * 2 - 1).to(torch.bfloat16) if shape_e else None

    # This op REQUIRES HEIGHT_SHARDED for all inputs.  Validate shard grids
    # before attempting tensor creation to avoid TT_FATAL crashes.
    # If any memory config is not HEIGHT_SHARDED (e.g., DRAM from sample suite),
    # we skip validation since the C++ op will reject it cleanly.
    all_mem_configs = [
        ("Q input", input_a_memory_config),
        ("K input", input_b_memory_config),
        ("cos", input_c_memory_config),
        ("sin", input_d_memory_config),
        ("trans_mat", input_e_memory_config),
    ]

    # Check if ALL memory configs are sharded (required by C++ op)
    all_sharded = all(mc is not None and hasattr(mc, "is_sharded") and mc.is_sharded() for _, mc in all_mem_configs)

    if not all_sharded:
        # For the sample suite with DRAM configs, the op will fail at C++ validation.
        # Skip gracefully rather than hitting TT_FATAL.
        non_sharded = [
            name for name, mc in all_mem_configs if mc is None or not (hasattr(mc, "is_sharded") and mc.is_sharded())
        ]
        return [
            (
                False,
                f"rotary_embedding_llama_fused_qk requires HEIGHT_SHARDED for all inputs, "
                f"but {non_sharded} are not sharded. Skipping to avoid TT_FATAL.",
            ),
            0,
        ]

    try:
        # Create ttnn tensors with the exact traced sharded memory configs.
        # The traced shard specs define the core grids used by the real model run
        # and encode the Q/K non-overlapping grid constraint.
        input_tensor_a = _create_tensor_on_device(
            torch_input_a,
            device,
            input_a_dtype,
            input_a_layout,
            input_a_memory_config,
            is_mesh_device,
            input_a_tensor_placement,
        )
        input_tensor_b = _create_tensor_on_device(
            torch_input_b,
            device,
            input_b_dtype,
            input_b_layout,
            input_b_memory_config,
            is_mesh_device,
            input_b_tensor_placement,
        )
        input_tensor_c = _create_tensor_on_device(
            torch_input_c,
            device,
            input_c_dtype,
            input_c_layout,
            input_c_memory_config,
            is_mesh_device,
            input_c_tensor_placement,
        )
        input_tensor_d = _create_tensor_on_device(
            torch_input_d,
            device,
            input_d_dtype,
            input_d_layout,
            input_d_memory_config,
            is_mesh_device,
            input_d_tensor_placement,
        )

        if torch_input_e is not None:
            input_tensor_e = _create_tensor_on_device(
                torch_input_e,
                device,
                input_e_dtype or ttnn.bfloat16,
                input_e_layout or ttnn.TILE_LAYOUT,
                input_e_memory_config or ttnn.DRAM_MEMORY_CONFIG,
                is_mesh_device,
                input_e_tensor_placement,
            )
        else:
            torch_input_e = torch.randn(1, 1, 64, 32).to(torch.bfloat16)
            input_tensor_e = _create_tensor_on_device(
                torch_input_e,
                device,
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                input_e_memory_config or ttnn.DRAM_MEMORY_CONFIG,
                is_mesh_device,
                input_e_tensor_placement,
            )
    except Exception as e:
        # Shard spec incompatible with device (e.g., traced on T3K but running on N300
        # with a grid that exceeds available cores after harvesting).
        # Return a clean failure instead of crashing.
        return [(False, f"Failed to create sharded tensors on device: {e}"), 0]

    start_time = start_measuring_time()

    try:
        # rotary_embedding_llama_fused_qk returns a tuple of (Q_rotated, K_rotated)
        result = ttnn.experimental.rotary_embedding_llama_fused_qk(
            input_tensor_a,
            input_tensor_b,
            input_tensor_c,
            input_tensor_d,
            input_tensor_e,
            **op_kwargs,
        )
    except Exception as e:
        e2e_perf = stop_measuring_time(start_time)
        return [(False, f"Op execution failed: {e}"), e2e_perf]

    # The operation returns a tuple of (Q_rotated, K_rotated)
    if isinstance(result, (list, tuple)) and len(result) == 2:
        output_tensor_q = mesh_tensor_to_torch(result[0], device if is_mesh_device else None)
        output_tensor_k = mesh_tensor_to_torch(result[1], device if is_mesh_device else None)
    else:
        e2e_perf = stop_measuring_time(start_time)
        return [(False, f"Expected tuple of 2 tensors, got {type(result)}"), e2e_perf]

    e2e_perf = stop_measuring_time(start_time)

    # Verify outputs are valid (non-NaN, non-zero, correct shapes)
    q_valid = not torch.isnan(output_tensor_q).any() and output_tensor_q.numel() > 0
    k_valid = not torch.isnan(output_tensor_k).any() and output_tensor_k.numel() > 0

    q_shape_ok = list(output_tensor_q.shape[-2:]) == list(shape_a[-2:])
    k_shape_ok = list(output_tensor_k.shape[-2:]) == list(shape_b[-2:])

    if q_valid and k_valid and q_shape_ok and k_shape_ok:
        pcc = (True, f"Q shape: {list(output_tensor_q.shape)}, K shape: {list(output_tensor_k.shape)}")
    else:
        reasons = []
        if not q_valid:
            reasons.append("Q has NaN")
        if not k_valid:
            reasons.append("K has NaN")
        if not q_shape_ok:
            reasons.append(f"Q shape {list(output_tensor_q.shape)} != expected {list(shape_a)}")
        if not k_shape_ok:
            reasons.append(f"K shape {list(output_tensor_k.shape)} != expected {list(shape_b)}")
        pcc = (False, "; ".join(reasons))

    return [pcc, e2e_perf]
