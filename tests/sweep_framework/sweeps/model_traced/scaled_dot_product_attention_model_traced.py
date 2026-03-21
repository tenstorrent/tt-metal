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

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("transformer::scaled_dot_product_attention")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 8, 32, 64)],  # Batch, heads, seq_len, head_dim
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_c_dtype": [ttnn.bfloat16],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> tuple:
    """
    Filter out configs that are known to cause timeouts or resource issues.
    """
    input_shape = test_vector.get("input_a_shape")

    # Extract Q shape - handle both dict (V1) and tuple/list (V2) formats
    shape_q = None
    if isinstance(input_shape, dict):
        shape_q = input_shape.get("input_a") or input_shape.get("self")
    elif isinstance(input_shape, (list, tuple)) and len(input_shape) >= 4:
        shape_q = input_shape

    if shape_q and isinstance(shape_q, (list, tuple)) and len(shape_q) >= 4:
        batch, num_heads, seq_len, head_dim = shape_q[0], shape_q[1], shape_q[2], shape_q[3]

        # Filter very large sequence lengths that cause device hangs/OOM
        if seq_len > 4096:
            return True, f"Sequence length {seq_len} too large (timeout/OOM risk)"

        if num_heads > 64:
            return True, f"Number of heads {num_heads} too large (timeout risk)"

        # Large batch sizes can OOM on single device (N150/N300)
        if batch > 16:
            return True, f"Batch size {batch} too large for single device (OOM risk)"

        # Filter configs with very large total attention computation
        total_elements = batch * num_heads * seq_len * seq_len * head_dim
        if total_elements > 1024 * 1024 * 1024:  # 1B elements
            return (
                True,
                f"Attention computation too large: {total_elements / (1024**3):.2f}B elements (timeout risk)",
            )

    return False, None


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
            device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_shape=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    raw_placement_a = kwargs.get("input_a_tensor_placement", None)
    input_a_tensor_placement = raw_placement_a
    raw_placement_b = kwargs.get("input_b_tensor_placement", None)
    input_b_tensor_placement = raw_placement_b
    raw_placement_c = kwargs.get("input_c_tensor_placement", None)
    input_c_tensor_placement = raw_placement_c
    is_mesh_device = hasattr(device, "get_num_devices")
    is_causal = kwargs.get("is_causal", False)
    if is_causal is None:
        is_causal = False

    # Clear sharded memory configs - shard specs from traced configs have galaxy-specific
    # core grids that are invalid on N150/N300 (different harvesting, grid sizes).
    if input_a_memory_config is not None and "SHARDED" in str(input_a_memory_config):
        input_a_memory_config = ttnn.DRAM_MEMORY_CONFIG
    if input_b_memory_config is not None and "SHARDED" in str(input_b_memory_config):
        input_b_memory_config = ttnn.DRAM_MEMORY_CONFIG
    if input_c_memory_config is not None and "SHARDED" in str(input_c_memory_config):
        input_c_memory_config = ttnn.DRAM_MEMORY_CONFIG
    if output_memory_config is not None and "SHARDED" in str(output_memory_config):
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    op_kwargs = build_op_kwargs(kwargs, exclude={"is_causal"}, output_memory_config=output_memory_config)

    # Clear sharded memory_config from op_kwargs too

    # Validate program_config grid fits current device
    pc = op_kwargs.get("program_config")
    if pc is not None:
        try:
            device_grid = device.compute_with_storage_grid_size()
            pc_grid = pc.compute_with_storage_grid_size
            if pc_grid.x > device_grid.x or pc_grid.y > device_grid.y:
                del op_kwargs["program_config"]
        except Exception:
            del op_kwargs["program_config"]

    # Handle shape extraction — V2 loader provides separate input_b_shape, input_c_shape
    if isinstance(input_a_shape, dict):
        # Traced configuration with multiple inputs (Q, K, V)
        shape_q = input_a_shape.get("input_a", input_a_shape.get("self"))
        shape_k = input_a_shape.get("input_b", input_a_shape.get("other"))
        shape_v = input_a_shape.get("input_c")
        if shape_v is None:
            shape_v = shape_k
    else:
        shape_q = tuple(input_a_shape) if isinstance(input_a_shape, (tuple, list)) else input_a_shape
        shape_k = (
            tuple(input_b_shape)
            if input_b_shape is not None and input_b_shape != "__ABSENT__" and isinstance(input_b_shape, (tuple, list))
            else shape_q
        )
        shape_v = (
            tuple(input_c_shape)
            if input_c_shape is not None and input_c_shape != "__ABSENT__" and isinstance(input_c_shape, (tuple, list))
            else shape_k
        )

    def _or_default(val, default):
        return val if val is not None and val != "__ABSENT__" else default

    dtype_q = input_a_dtype
    dtype_k = _or_default(input_b_dtype, dtype_q)
    dtype_v = _or_default(input_c_dtype, dtype_k)

    layout_q = input_a_layout
    layout_k = _or_default(input_b_layout, layout_q)
    layout_v = _or_default(input_c_layout, layout_k)

    mem_config_q = input_a_memory_config
    mem_config_k = _or_default(input_b_memory_config, mem_config_q)
    mem_config_v = _or_default(input_c_memory_config, mem_config_k)

    batch_size, num_heads_q, seq_len, head_dim = shape_q
    _, num_heads_k, _, _ = shape_k
    _, num_heads_v, _, _ = shape_v

    # Create Q, K, V tensors
    torch_q = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_q)(shape_q)
    torch_k = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_k)(shape_k)
    torch_v = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_v)(shape_v)

    # Handle GQA (Grouped Query Attention) - if K/V have fewer heads, replicate them
    if num_heads_k < num_heads_q:
        repeat_factor = num_heads_q // num_heads_k
        torch_k = torch_k.repeat(1, repeat_factor, 1, 1)
        if num_heads_q % num_heads_k != 0:
            remaining = num_heads_q - (repeat_factor * num_heads_k)
            torch_k = torch.cat([torch_k, torch_k[:, -num_heads_k : -num_heads_k + remaining, :, :]], dim=1)

    if num_heads_v < num_heads_q:
        repeat_factor = num_heads_q // num_heads_v
        torch_v = torch_v.repeat(1, repeat_factor, 1, 1)
        if num_heads_q % num_heads_v != 0:
            remaining = num_heads_q - (repeat_factor * num_heads_v)
            torch_v = torch.cat([torch_v, torch_v[:, -num_heads_v : -num_heads_v + remaining, :, :]], dim=1)

    # Quantize inputs to target dtype - both PyTorch and TTNN use same quantized inputs.
    # Always use DRAM interleaved for the round-trip (safe on any device).
    # On mesh devices, use ReplicateTensorToMesh for the round-trip; then extract
    # device 0's copy via get_device_tensors to get a clean single-device torch tensor.
    def _quantize_roundtrip(torch_tensor, dtype, layout):
        if is_mesh_device:
            t = ttnn.from_torch(
                torch_tensor,
                dtype=dtype,
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
            return ttnn.to_torch(ttnn.get_device_tensors(t)[0])
        else:
            return ttnn.to_torch(
                ttnn.from_torch(
                    torch_tensor, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            )

    torch_q = _quantize_roundtrip(torch_q, dtype_q, layout_q)
    torch_k = _quantize_roundtrip(torch_k, dtype_k, layout_k)
    torch_v = _quantize_roundtrip(torch_v, dtype_v, layout_v)

    # Ensure all tensors have the same dtype for PyTorch SDPA
    torch_q = torch_q.to(torch.float32)
    torch_k = torch_k.to(torch.float32)
    torch_v = torch_v.to(torch.float32)

    # PyTorch reference
    torch_output_golden = torch.nn.functional.scaled_dot_product_attention(
        torch_q, torch_k, torch_v, attn_mask=None, dropout_p=0.0, is_causal=bool(is_causal)
    )

    # TTNN execution
    if is_mesh_device and input_a_tensor_placement:
        q_tensor = create_tensor_on_mesh(torch_q, device, dtype_q, layout_q, mem_config_q, input_a_tensor_placement)
        k_tensor = create_tensor_on_mesh(torch_k, device, dtype_k, layout_k, mem_config_k, input_b_tensor_placement)
        v_tensor = create_tensor_on_mesh(torch_v, device, dtype_v, layout_v, mem_config_v, input_c_tensor_placement)
    else:
        q_tensor = ttnn.from_torch(torch_q, dtype=dtype_q, layout=layout_q, device=device, memory_config=mem_config_q)
        k_tensor = ttnn.from_torch(torch_k, dtype=dtype_k, layout=layout_k, device=device, memory_config=mem_config_k)
        v_tensor = ttnn.from_torch(torch_v, dtype=dtype_v, layout=layout_v, device=device, memory_config=mem_config_v)

    start_time = start_measuring_time()
    output_tensor = ttnn.transformer.scaled_dot_product_attention(
        q_tensor, k_tensor, v_tensor, is_causal=bool(is_causal), **op_kwargs
    )
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Compare raw golden (float32) against TTNN output.
    # Do NOT requantize the golden — that introduces double-quantization error.
    # LoFi compute kernels have lower precision — use relaxed threshold.
    ckc = op_kwargs.get("compute_kernel_config")
    is_lofi = False
    if ckc is not None:
        try:
            is_lofi = ckc.math_fidelity == ttnn.MathFidelity.LoFi
        except Exception:
            pass
    pcc_threshold = 0.98 if is_lofi else 0.99
    pcc = check_with_pcc(torch_output_golden, output_tensor, pcc_threshold)

    return [pcc, e2e_perf]
