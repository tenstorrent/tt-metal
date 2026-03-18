# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_named_tensor_kwargs

TIMEOUT = 300


def page_cache(cache, page_block_size, permutation):
    """Convert K or V cache to paged format.

    Args:
        cache: [b, nkv, s, d] tensor
        page_block_size: size of each page block
        permutation: permutation for shuffling pages

    Returns:
        paged_cache: [max_num_blocks, nkv, page_block_size, d]
    """
    b, nkv, s, d = cache.shape
    max_num_blocks_per_seq = s // page_block_size
    max_num_blocks = b * max_num_blocks_per_seq

    paged_cache = (
        cache.reshape(b, nkv, max_num_blocks_per_seq, page_block_size, d)
        .transpose(1, 2)
        .reshape(max_num_blocks, nkv, page_block_size, d)
    )

    shuffled_page_cache = paged_cache[permutation]
    return shuffled_page_cache


loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("transformer::chunked_scaled_dot_product_attention")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 8, 32, 64)],
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
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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


def run(
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    is_mesh_device = hasattr(device, "get_num_devices")

    # --- Extract tensor parameters ---
    # V2 model_traced suite provides named tensor kwargs:
    #   input_tensor_q_*, input_tensor_k_*, input_tensor_v_*, page_table_tensor_*
    # Sample suite provides positional tensor params:
    #   input_a_*, input_b_*, input_c_*, input_d_*
    q_kwargs = extract_named_tensor_kwargs(kwargs, "input_tensor_q")
    k_kwargs = extract_named_tensor_kwargs(kwargs, "input_tensor_k")
    v_kwargs = extract_named_tensor_kwargs(kwargs, "input_tensor_v")
    pt_kwargs = extract_named_tensor_kwargs(kwargs, "page_table_tensor")

    if q_kwargs and q_kwargs.get("shape") is not None:
        # V2 path: named tensor kwargs from traced configurations
        input_a_shape = q_kwargs["shape"]
        input_a_dtype = q_kwargs.get("dtype", ttnn.bfloat16)
        input_a_layout = q_kwargs.get("layout", ttnn.TILE_LAYOUT)
        input_a_memory_config = q_kwargs.get("memory_config", ttnn.DRAM_MEMORY_CONFIG)
        input_a_tensor_placement = q_kwargs.get("tensor_placement")

        input_b_shape = k_kwargs["shape"] if k_kwargs else None
        input_b_dtype = k_kwargs.get("dtype", ttnn.bfloat16) if k_kwargs else ttnn.bfloat16
        input_b_layout = k_kwargs.get("layout", ttnn.TILE_LAYOUT) if k_kwargs else ttnn.TILE_LAYOUT
        input_b_memory_config = (
            k_kwargs.get("memory_config", ttnn.DRAM_MEMORY_CONFIG) if k_kwargs else ttnn.DRAM_MEMORY_CONFIG
        )
        input_b_tensor_placement = k_kwargs.get("tensor_placement") if k_kwargs else None

        input_c_dtype = v_kwargs.get("dtype", ttnn.bfloat16) if v_kwargs else ttnn.bfloat16
        input_c_layout = v_kwargs.get("layout", ttnn.TILE_LAYOUT) if v_kwargs else ttnn.TILE_LAYOUT
        input_c_memory_config = (
            v_kwargs.get("memory_config", ttnn.DRAM_MEMORY_CONFIG) if v_kwargs else ttnn.DRAM_MEMORY_CONFIG
        )
        input_c_tensor_placement = v_kwargs.get("tensor_placement") if v_kwargs else None

        input_d_tensor_placement = pt_kwargs.get("tensor_placement") if pt_kwargs else None
    else:
        # Sample suite path: positional tensor params (input_a_*, input_b_*, etc.)
        input_a_shape = kwargs.get("input_a_shape", (1, 8, 32, 64))
        input_a_dtype = kwargs.get("input_a_dtype", ttnn.bfloat16)
        input_a_layout = kwargs.get("input_a_layout", ttnn.TILE_LAYOUT)
        input_a_memory_config = kwargs.get("input_a_memory_config", ttnn.DRAM_MEMORY_CONFIG)
        input_a_tensor_placement = kwargs.get("input_a_tensor_placement")

        input_b_shape = kwargs.get("input_b_shape")
        input_b_dtype = kwargs.get("input_b_dtype", ttnn.bfloat16)
        input_b_layout = kwargs.get("input_b_layout", ttnn.TILE_LAYOUT)
        input_b_memory_config = kwargs.get("input_b_memory_config", ttnn.DRAM_MEMORY_CONFIG)
        input_b_tensor_placement = kwargs.get("input_b_tensor_placement")

        input_c_dtype = kwargs.get("input_c_dtype", ttnn.bfloat16)
        input_c_layout = kwargs.get("input_c_layout", ttnn.TILE_LAYOUT)
        input_c_memory_config = kwargs.get("input_c_memory_config", ttnn.DRAM_MEMORY_CONFIG)
        input_c_tensor_placement = kwargs.get("input_c_tensor_placement")

        input_d_tensor_placement = kwargs.get("input_d_tensor_placement")

    output_memory_config = kwargs.get("output_memory_config", ttnn.DRAM_MEMORY_CONFIG)

    chunk_start_idx = kwargs.get("chunk_start_idx", 0)
    if chunk_start_idx is None:
        chunk_start_idx = 0
    op_kwargs = build_op_kwargs(kwargs, exclude={"chunk_start_idx"}, output_memory_config=output_memory_config)

    # Extract shapes for Q and K/V paged from separate inputs or dict fallback
    if isinstance(input_a_shape, dict):
        shape_q = input_a_shape.get("input_a", (1, 8, 32, 64))
        shape_k_paged = input_a_shape.get("input_b", (64, 1, 64, 64))
    else:
        shape_q = tuple(input_a_shape) if isinstance(input_a_shape, (tuple, list)) else (1, 8, 32, 64)
        if input_b_shape is not None:
            shape_k_paged = tuple(input_b_shape)
        else:
            b, nh, sq, d = shape_q
            page_block_size = 64
            max_num_blocks_per_seq = max(1, (sq + page_block_size - 1) // page_block_size)
            nkv = 1
            shape_k_paged = (b * max_num_blocks_per_seq, nkv, page_block_size, d)

    # Extract dimensions from Q
    b, nh, sq, d = shape_q

    # For paged K, V: [num_pages, nkv, page_block_size, head_dim]
    num_pages, nkv, page_block_size, _ = shape_k_paged

    # Calculate unpaged sequence length from paged dimensions
    max_num_blocks_per_seq = num_pages // b
    s = max_num_blocks_per_seq * page_block_size

    # Create unpaged K and V for torch reference
    torch_k_unpaged = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_b_dtype)(
        (b, nkv, s, d)
    )
    torch_v_unpaged = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_c_dtype)(
        (b, nkv, s, d)
    )

    # Create Q
    torch_q = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype)(shape_q)

    # Create page table and permutation
    max_num_blocks = b * max_num_blocks_per_seq
    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    torch_page_table = reverse_permutation.reshape(b, max_num_blocks_per_seq)

    # Convert K, V to paged format for TTNN operation
    torch_k_paged = page_cache(torch_k_unpaged, page_block_size, permutation)
    torch_v_paged = page_cache(torch_v_unpaged, page_block_size, permutation)

    # Torch reference: Use unpaged K, V with chunked causal attention
    # chunk_start_idx tells us where in the full sequence this chunk starts
    # Q attends to K/V from positions [0, chunk_start_idx + sq]
    kv_end = min(chunk_start_idx + sq, s)
    torch_k_chunk = torch_k_unpaged[:, :, :kv_end, :]
    torch_v_chunk = torch_v_unpaged[:, :, :kv_end, :]

    if nkv < nh:
        K_repeated = torch.cat(
            [torch_k_chunk[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
        )
        V_repeated = torch.cat(
            [torch_v_chunk[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
        )
    else:
        K_repeated = torch_k_chunk
        V_repeated = torch_v_chunk

    # Build causal mask for chunked attention:
    # Q position i (absolute: chunk_start_idx + i) can attend to K position j if j <= chunk_start_idx + i
    q_positions = torch.arange(chunk_start_idx, chunk_start_idx + sq).unsqueeze(1)  # [sq, 1]
    k_positions = torch.arange(kv_end).unsqueeze(0)  # [1, kv_end]
    causal_mask = (k_positions <= q_positions).unsqueeze(0).unsqueeze(0)  # [1, 1, sq, kv_end]
    # Convert to float mask (0 for allowed, -inf for masked)
    attn_mask = torch.where(causal_mask, 0.0, float("-inf"))

    # Ensure all tensors are float32 for torch SDPA
    torch_output = torch.nn.functional.scaled_dot_product_attention(
        torch_q.to(torch.float32),
        K_repeated.to(torch.float32),
        V_repeated.to(torch.float32),
        attn_mask=attn_mask.to(torch.float32),
        dropout_p=0.0,
        is_causal=False,  # We provide explicit mask
    )

    # Create TTNN tensors with paged K, V
    dtype_k = input_b_dtype
    dtype_v = input_c_dtype
    layout_k = input_b_layout
    layout_v = input_c_layout
    mem_k = input_b_memory_config
    mem_v = input_c_memory_config

    if is_mesh_device and input_a_tensor_placement:
        q_tensor = create_tensor_on_mesh(
            torch_q, device, input_a_dtype, input_a_layout, input_a_memory_config, input_a_tensor_placement
        )
        k_tensor = create_tensor_on_mesh(torch_k_paged, device, dtype_k, layout_k, mem_k, input_b_tensor_placement)
        v_tensor = create_tensor_on_mesh(torch_v_paged, device, dtype_v, layout_v, mem_v, input_c_tensor_placement)
        page_table_tensor = create_tensor_on_mesh(
            torch_page_table,
            device,
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
            input_a_memory_config,
            input_d_tensor_placement,
        )
    else:
        q_tensor = ttnn.from_torch(
            torch_q, dtype=input_a_dtype, layout=input_a_layout, device=device, memory_config=input_a_memory_config
        )
        k_tensor = ttnn.from_torch(
            torch_k_paged,
            dtype=dtype_k,
            layout=layout_k,
            device=device,
            memory_config=mem_k,
        )
        v_tensor = ttnn.from_torch(
            torch_v_paged,
            dtype=dtype_v,
            layout=layout_v,
            device=device,
            memory_config=mem_v,
        )
        page_table_tensor = ttnn.from_torch(
            torch_page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_a_memory_config,
        )

    start_time = start_measuring_time()
    output_tensor = ttnn.transformer.chunked_scaled_dot_product_attention(
        q_tensor, k_tensor, v_tensor, page_table_tensor, chunk_start_idx, **op_kwargs
    )
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Comparison - use 0.998 PCC threshold as in unit test
    return [check_with_pcc(torch_output, output_tensor, 0.998), e2e_perf]
