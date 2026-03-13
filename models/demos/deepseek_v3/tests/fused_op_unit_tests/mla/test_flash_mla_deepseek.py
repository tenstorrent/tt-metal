# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import nearest_y
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_with_pcc


def _debug_mtp_enabled() -> bool:
    return os.getenv("DEBUG_MTP", "0") == "1"


def create_paged_kvpe_cache(device, num_users, max_seq_len, head_dim, num_blocks, block_size, mapping):
    """Create a paged KVPE cache for testing."""
    # Per-user sequence length must match paging: num_blocks * block_size total slots, num_users -> (num_blocks * block_size) // num_users per user
    seq_len_per_user = (num_blocks * block_size) // num_users
    assert (
        seq_len_per_user == max_seq_len
    ), f"Paging invariant: (num_blocks * block_size) // num_users must equal max_seq_len; got {seq_len_per_user} vs {max_seq_len}"
    cache_shape = (num_users, 1, seq_len_per_user, head_dim)
    cache = torch.randn(cache_shape, dtype=torch.bfloat16) * 0.1

    paged_cache = cache.reshape(num_users, 1, -1, block_size, head_dim)
    paged_cache = paged_cache.transpose(1, 2)
    paged_cache = paged_cache.reshape(num_blocks, 1, block_size, head_dim)
    inverse_mapping = torch.argsort(mapping.view(-1))
    paged_cache = paged_cache[inverse_mapping]

    # Convert to ttnn with DRAM memory (matching model configuration)
    tt_cache = ttnn.from_torch(
        paged_cache,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return tt_cache, cache


def create_page_table(device, num_users, num_blocks):
    """Create a page table mapping logical to physical blocks."""
    # Page table shape: [num_users, max_num_blocks_per_user]
    # For simplicity, use identity mapping
    blocks_per_user = num_blocks // num_users
    page_table = torch.randperm(num_blocks, dtype=torch.int32).reshape(num_users, blocks_per_user)

    # Convert to ttnn
    tt_page_table = ttnn.from_torch(
        page_table,
        dtype=ttnn.int32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return tt_page_table, page_table


def scaled_dot_product_attention_reference(Q, K, V, start_indices, padded_layer_len, scale, is_causal=True):
    b, nh, _, _ = Q.shape  # b, nh, 1, d
    _, nkv, _, _ = K.shape

    attn_mask = None
    if is_causal:
        attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
        for i in range(b):
            start_idx = start_indices[i]
            attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min
    else:
        assert False, "Non-causal attention is not supported in this function."

    Q_slice = Q[:, :nh, :, :]  # b, nh, 1, d
    K_slice = K[:, :nkv, :padded_layer_len, :]  # b, nkv, S, d
    K_slice = torch.cat(
        [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S
    V_slice = V[:, :, :padded_layer_len, :]  # b, nkv, S, d
    V_slice = torch.cat(
        [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S
    attn_mask_slice = attn_mask[:, :nh, :, :]  # b, nh, 1, S
    out = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d

    return out


def _reconstruct_logical_cache_from_paged(
    paged_cache: torch.Tensor,
    page_table: torch.Tensor,
    block_size: int,
    head_dim: int,
) -> torch.Tensor:
    num_rows, blocks_per_user = page_table.shape
    cache = torch.empty((num_rows, 1, blocks_per_user * block_size, head_dim), dtype=paged_cache.dtype)
    for row in range(num_rows):
        blocks = [paged_cache[int(block_idx.item()), 0] for block_idx in page_table[row]]
        cache[row, 0] = torch.cat(blocks, dim=0)
    return cache


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "op_name, q_shape, cache_params, output_shape, shard_shape, num_cores",
    [
        (
            "flash_mla_decode",
            [1, 4, 128, 576],  # Q shape after all-to-all: [1, bsz_local, num_heads, kv_lora_rank + qk_rope_head_dim]
            {
                "num_users": 4,  # Per device: 32 users / 8 devices = 4
                "max_seq_len": 1024,
                "head_dim": 576,  # kv_lora_rank + qk_rope_head_dim
                "num_blocks": 128,  # (num_blocks * block_size) // num_users == max_seq_len; 128*32//4 == 1024
                "block_size": 32,
                "kv_lora_rank": 512,
            },
            [1, 4, 128, 512],  # Output: [1, bsz_local, num_heads, kv_lora_rank]
            [32, 576],  # HEIGHT_SHARDED shard shape for input Q
            64,  # (32/8) * 16 = 4 * 16 = 64 cores
        ),
    ],
    ids=["flash_mla_decode"],
)
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 5500000,  # Larger trace region for flash attention
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_flash_mla_trace_mode(
    device,
    batch_size,
    op_name,
    q_shape,
    cache_params,
    output_shape,
    shard_shape,
    num_cores,
    warmup_iters,
    num_iters,
):
    """
    Test the paged_flash_multi_latent_attention_decode operation from mla1d.py with trace mode.

    This operation performs paged flash attention for MLA (Multi-head Latent Attention):
    - flash_mla_decode (line 1287): Paged flash MLA decode operation
      Input Q: [1, 4, 128, 576] height sharded
      Output: [1, 4, 128, 512] height sharded

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - HEIGHT_SHARDED memory layout
    - k_chunk_size: 128
    - Scale: (qk_head_dim)**-0.5
    """
    torch.manual_seed(0)

    num_users = cache_params["num_users"]
    max_seq_len = cache_params["max_seq_len"]
    head_dim = cache_params["head_dim"]
    num_blocks = cache_params["num_blocks"]
    block_size = cache_params["block_size"]
    kv_lora_rank = cache_params["kv_lora_rank"]

    # Create Q tensor
    torch_q = torch.randn(q_shape, dtype=torch.bfloat16) * 0.1

    # Create page table
    tt_page_table, torch_page_table = create_page_table(device, num_users, num_blocks)

    # Create paged KVPE cache
    tt_kvpe_cache, torch_kvpe_cache = create_paged_kvpe_cache(
        device, num_users, max_seq_len, head_dim, num_blocks, block_size, torch_page_table
    )

    # Create position indices (current position for each user)
    position_idxs = np.linspace(0, max_seq_len // 2, num_users, dtype=np.int32)

    # Convert Q to ttnn with L1 interleaved first
    tt_q = ttnn.from_torch(
        torch_q,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Create HEIGHT_SHARDED memory config for Q
    grid_size = device.compute_with_storage_grid_size()
    q_core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)

    q_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tt_q = ttnn.to_memory_config(tt_q, q_sharded_mem_config)

    # Convert position indices to ttnn
    tt_position_idxs = ttnn.from_torch(
        position_idxs,
        dtype=ttnn.int32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Flash MLA configuration matching mla1d.py decode config
    qk_nope_head_dim = 192
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 256
    scale = qk_head_dim**-0.5

    # Output memory config - height sharded with kv_lora_rank width
    output_shard_shape = [shard_shape[0], kv_lora_rank]  # [32, 512]
    output_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=output_shard_shape,
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Program config
    q_chunk_size = 0  # Unused in decode mode
    k_chunk_size = 128
    padded_layer_len = nearest_y(max_seq_len // 2 + 1, k_chunk_size)
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    # Compute kernel config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Compile run
    logger.info(f"Compiling flash MLA operation: {op_name}")
    logger.info(f"  Q shape: {q_shape}")
    logger.info(
        f"  Cache shape: [{num_users}, 1, {max_seq_len}, {head_dim}] (num_blocks={num_blocks}, block_size={block_size})"
    )
    logger.info(f"  Output shape: {output_shape}")
    logger.info(f"  Q shard shape: {shard_shape}")
    logger.info(f"  Output shard shape: {output_shard_shape}")
    logger.info(f"  Num cores: {num_cores}")
    logger.info(f"  Scale: {scale}")

    # Pass only K cache; V is read from first head_dim_v (kv_lora_rank) dims of K (MLA semantics, matches mla1d.py).
    tt_output = ttnn.transformer.paged_flash_multi_latent_attention_decode(
        tt_q,
        tt_kvpe_cache,
        page_table_tensor=tt_page_table,
        cur_pos_tensor=tt_position_idxs,
        head_dim_v=kv_lora_rank,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=output_sharded_mem_config,
    )
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            tt_q,
            tt_kvpe_cache,
            page_table_tensor=tt_page_table,
            cur_pos_tensor=tt_position_idxs,
            head_dim_v=kv_lora_rank,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=output_sharded_mem_config,
        )
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            tt_q,
            tt_kvpe_cache,
            page_table_tensor=tt_page_table,
            cur_pos_tensor=tt_position_idxs,
            head_dim_v=kv_lora_rank,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=output_sharded_mem_config,
        )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler = BenchmarkProfiler()
    profiler.start("warmup")
    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)
    profiler.end("warmup")
    ttnn.synchronize_device(device)

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("main")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    profiler.end("main")
    signpost("stop")
    ttnn.synchronize_device(device)

    # Verify output shape and correctness against PyTorch reference.
    # KVPE cache: K uses full head_dim 576; V is first kv_lora_rank (512) dims (MLA semantics).
    # Device Q has batch 1 (single user); run reference for user 0 only so shapes and semantics match.
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    torch_output = scaled_dot_product_attention_reference(
        torch_q.permute(1, 2, 0, 3),
        torch_kvpe_cache,  # K: user 0 only, [1, 1, 2048, 576]
        torch_kvpe_cache[..., :kv_lora_rank],  # V: user 0 only, first 512 dims
        position_idxs,
        padded_layer_len,
        scale,
    )
    torch_output = torch_output.permute(2, 0, 1, 3)
    assert list(tt_output.shape) == output_shape, f"Shape mismatch: {list(tt_output.shape)} != {output_shape}"
    assert_with_pcc(torch_output, tt_output, pcc=0.99)

    # PCC check allows for bfloat16 / accumulation differences vs PyTorch reference
    logger.info(f"✓ Trace mode {op_name} test passed with correct output shape")


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_paged_flash_mla_verify_aliasing_two_step_mesh_sharded_pos_idxs(mesh_device):
    """
    Replay two consecutive aliased verify steps with mesh-sharded position/update
    tensors to check both the single-step mesh-sharded read path and the
    multi-step cache handoff in one test.
    """
    torch.manual_seed(0)
    enable_debug = _debug_mtp_enabled()

    mesh_shape = tuple(mesh_device.shape)
    dp_factor = mesh_shape[1]
    num_devices = mesh_shape[0] * mesh_shape[1]

    num_users = 2
    batch_per_shard = 2 * num_users
    total_batch = batch_per_shard * dp_factor

    num_heads = 128
    head_dim = 576
    kv_lora_rank = 512
    block_size = ttnn.TILE_SIZE
    blocks_per_user = 4
    max_seq_len = blocks_per_user * block_size
    max_num_blocks = batch_per_shard * blocks_per_user

    base_page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(batch_per_shard, blocks_per_user)
    alias_page_table = base_page_table.clone()
    for row in range(1, batch_per_shard, 2):
        alias_page_table[row] = alias_page_table[row - 1]

    tt_page_table = ttnn.from_torch(
        alias_page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    torch_cache = (torch.randn((max_num_blocks, 1, block_size, head_dim), dtype=torch.float32) * 0.05).to(
        torch.bfloat16
    )
    tt_cache = ttnn.from_torch(
        torch_cache,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    grid_size = mesh_device.compute_with_storage_grid_size()
    update_core_grid = ttnn.num_cores_to_corerangeset(batch_per_shard, grid_size, row_wise=True)
    update_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(nearest_y(1, ttnn.TILE_SIZE), head_dim),
        core_grid=update_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    q_core_grid = ttnn.num_cores_to_corerangeset(64, grid_size, row_wise=True)
    q_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(32, head_dim),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    output_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(32, kv_lora_rank),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_q_steps = []
    tt_q_steps = []
    for step_idx in range(2):
        torch_q = (torch.randn((1, total_batch, num_heads, head_dim), dtype=torch.float32) * 0.05).to(torch.bfloat16)
        torch_q = torch_q + torch.tensor(0.02 * step_idx, dtype=torch.bfloat16)
        tt_q = ttnn.from_torch(
            torch_q,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=q_sharded_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=mesh_shape),
        )
        torch_q_steps.append(torch_q)
        tt_q_steps.append(tt_q)

    scale = (192 + 64) ** -0.5
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    mesh_coords = {ttnn.MeshCoordinate(r, c) for r in range(mesh_shape[0]) for c in range(mesh_shape[1])}
    ref_cache_by_device = [torch_cache.clone() for _ in range(num_devices)]
    step_positions = [
        torch.tensor([1, 2, 1, 2], dtype=torch.int32),
        torch.tensor([2, 3, 2, 3], dtype=torch.int32),
    ]

    for step_idx, positions_local in enumerate(step_positions):
        prompt_update_idxs_local = positions_local.clone()
        prompt_update_idxs_local[1::2] = -1
        spec_update_idxs_local = positions_local.clone()
        spec_update_idxs_local[0::2] = -1

        prompt_update_idxs_global = prompt_update_idxs_local.repeat(num_devices)
        spec_update_idxs_global = spec_update_idxs_local.repeat(num_devices)
        positions_global = positions_local.repeat(num_devices)

        update_tensor = torch.zeros((1, total_batch, 1, head_dim), dtype=torch.bfloat16)
        for shard in range(dp_factor):
            base = shard * batch_per_shard
            step_base = 0.25 * step_idx + 0.05 * shard
            update_tensor[0, base + 0, 0, :] = 0.10 + step_base
            update_tensor[0, base + 1, 0, :] = 0.20 + step_base
            update_tensor[0, base + 2, 0, :] = 0.30 + step_base
            update_tensor[0, base + 3, 0, :] = 0.40 + step_base

        tt_update = ttnn.from_torch(
            update_tensor,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=update_mem_cfg,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=mesh_shape),
        )
        tt_prompt_update_idxs = ttnn.from_torch(
            prompt_update_idxs_global,
            device=mesh_device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_spec_update_idxs = ttnn.from_torch(
            spec_update_idxs_global,
            device=mesh_device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_position_idxs = ttnn.from_torch(
            positions_global,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        ttnn.experimental.paged_update_cache(
            tt_cache,
            tt_update,
            update_idxs_tensor=tt_prompt_update_idxs,
            page_table=tt_page_table,
            mesh_coords=mesh_coords,
        )
        ttnn.experimental.paged_update_cache(
            tt_cache,
            tt_update,
            update_idxs_tensor=tt_spec_update_idxs,
            page_table=tt_page_table,
            mesh_coords=mesh_coords,
        )
        ttnn.synchronize_device(mesh_device)

        tt_output = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            tt_q_steps[step_idx],
            tt_cache,
            page_table_tensor=tt_page_table,
            cur_pos_tensor=tt_position_idxs,
            head_dim_v=kv_lora_rank,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=output_sharded_mem_config,
        )
        ttnn.synchronize_device(mesh_device)

        tt_cache_torch = ttnn.to_torch(tt_cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).reshape(
            num_devices, max_num_blocks, 1, block_size, head_dim
        )
        tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).reshape(
            num_devices, 1, batch_per_shard, num_heads, kv_lora_rank
        )

        for device_idx in range(num_devices):
            shard_idx = device_idx % dp_factor
            local_update = update_tensor[0, shard_idx * batch_per_shard : (shard_idx + 1) * batch_per_shard, 0, :]
            for local_idx, pos in enumerate(positions_local.tolist()):
                block = pos // block_size
                offset = pos % block_size
                physical_block = int(alias_page_table[local_idx, block].item())
                ref_cache_by_device[device_idx][physical_block, 0, offset, :] = local_update[local_idx, :]

            assert_with_pcc(ref_cache_by_device[device_idx], tt_cache_torch[device_idx], pcc=0.999)

            logical_cache = _reconstruct_logical_cache_from_paged(
                ref_cache_by_device[device_idx],
                alias_page_table,
                block_size=block_size,
                head_dim=head_dim,
            )
            q_local = torch_q_steps[step_idx][:, shard_idx * batch_per_shard : (shard_idx + 1) * batch_per_shard]
            torch_output = scaled_dot_product_attention_reference(
                q_local.permute(1, 2, 0, 3),
                logical_cache,
                logical_cache[..., :kv_lora_rank],
                positions_local.tolist(),
                max_seq_len,
                scale,
            ).permute(2, 0, 1, 3)
            assert_with_pcc(torch_output, tt_output_torch[device_idx], pcc=0.99)

        if enable_debug:
            logger.info(
                "two-step alias flash step={} positions={} device0 cache row2 pos{}[:4]={}",
                step_idx,
                positions_local.tolist(),
                int(positions_local[2].item()),
                tt_cache_torch[0, int(alias_page_table[2, 0].item()), 0, int(positions_local[2].item()), :4]
                .to(torch.float32)
                .tolist(),
            )
            logger.info(
                "two-step alias flash step={} device0 output slice={}",
                step_idx,
                tt_output_torch[0, 0, :, 0, :4].to(torch.float32).tolist(),
            )
