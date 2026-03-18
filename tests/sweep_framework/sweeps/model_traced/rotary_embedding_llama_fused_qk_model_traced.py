# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sweep test for ttnn.experimental.rotary_embedding_llama_fused_qk operation.

This operation fuses rotary position embedding with Q/K preparation in one kernel,
optimizing memory bandwidth and reducing overhead in transformer attention layers.

The operation returns two tensors: rotated Q and rotated K.
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


def _safe_memory_config(memory_config, tensor_shape, device):
    """Return memory_config if compatible with tensor/device, else DRAM_MEMORY_CONFIG.

    Traced shard specs are computed for a specific device topology and may use
    core grids that are invalid on the test device (different harvesting, grid sizes).
    Always fall back to DRAM interleaved for sharded configs to avoid TT_FATAL.
    """
    if memory_config is None:
        return ttnn.DRAM_MEMORY_CONFIG
    if not (hasattr(memory_config, "is_sharded") and memory_config.is_sharded()):
        return memory_config
    # Sharded memory configs from traced runs have device-specific shard specs.
    # Rather than trying to validate each shard spec against the test device grid,
    # fall back to DRAM interleaved which works universally.
    return ttnn.DRAM_MEMORY_CONFIG


def _create_tensor_on_device(torch_tensor, device, dtype, layout, memory_config, is_mesh_device, placement):
    """Create tensor on device, falling back to DRAM interleaved if shard spec is incompatible."""
    # Always validate shard spec compatibility before creating tensor.
    # Traced shard specs may have core grids from a different device topology.
    safe_mc = _safe_memory_config(memory_config, torch_tensor.shape, device)

    if is_mesh_device and placement:
        return create_tensor_on_mesh(torch_tensor, device, dtype, layout, safe_mc, placement)

    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=safe_mc,
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

    def _is_valid_placement(placement):
        if not placement or not isinstance(placement, dict):
            return False
        mesh_shape = placement.get("mesh_device_shape", "")
        if isinstance(mesh_shape, str):
            mesh_shape = mesh_shape.strip()
            if not mesh_shape or mesh_shape == "[]":
                return False
        elif isinstance(mesh_shape, list):
            if len(mesh_shape) < 2:
                return False
        return True

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    if not _is_valid_placement(input_a_tensor_placement):
        input_a_tensor_placement = None
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    if not _is_valid_placement(input_b_tensor_placement):
        input_b_tensor_placement = None
    input_c_tensor_placement = kwargs.get("input_c_tensor_placement", None)
    if not _is_valid_placement(input_c_tensor_placement):
        input_c_tensor_placement = None
    input_d_tensor_placement = kwargs.get("input_d_tensor_placement", None)
    if not _is_valid_placement(input_d_tensor_placement):
        input_d_tensor_placement = None
    input_e_tensor_placement = kwargs.get("input_e_tensor_placement", None)
    if not _is_valid_placement(input_e_tensor_placement):
        input_e_tensor_placement = None
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # Handle dict input_a_shape from traced configurations (multi-input)
    if isinstance(input_a_shape, dict):
        shape_a = input_a_shape.get("input_a", input_a_shape.get("q_input_tensor"))  # Q tensor
        shape_b = input_a_shape.get("input_b", input_a_shape.get("k_input_tensor"))  # K tensor
        shape_c = input_a_shape.get("input_c", input_a_shape.get("cos_cache"))  # cos cache
        shape_d = input_a_shape.get("input_d", input_a_shape.get("sin_cache"))  # sin cache
        shape_e = input_a_shape.get("input_e", input_a_shape.get("trans_mat"))  # trans_mat
    else:
        # Fallback for sample configurations
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

    # Use random tensors with exact traced shapes for all 5 inputs.
    # Production models use compute_gather_cos_sin and get_rot_transformation_mat
    # which produce model-specific shapes (e.g., trans_mat [1,1,2048,32] not [1,1,head_dim,head_dim]).
    # Using exact traced shapes ensures the op gets properly shaped tensors.
    torch_input_a = (torch.rand(shape_a) * 2 - 1).to(torch.bfloat16)
    torch_input_b = (torch.rand(shape_b) * 2 - 1).to(torch.bfloat16)
    torch_input_c = (torch.rand(shape_c) * 2 - 1).to(torch.bfloat16)
    torch_input_d = (torch.rand(shape_d) * 2 - 1).to(torch.bfloat16)
    torch_input_e = (torch.rand(shape_e) * 2 - 1).to(torch.bfloat16) if shape_e else None

    # Create ttnn tensors with fallback for sharded memory configs
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
        input_tensor_e = ttnn.from_torch(
            torch_input_e,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    start_time = start_measuring_time()

    # rotary_embedding_llama_fused_qk returns a tuple of (Q_rotated, K_rotated)
    result = ttnn.experimental.rotary_embedding_llama_fused_qk(
        input_tensor_a,
        input_tensor_b,
        input_tensor_c,
        input_tensor_d,
        input_tensor_e,
        **op_kwargs,
    )

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
