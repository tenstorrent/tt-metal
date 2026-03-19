# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN DecoderBlock Test
Tests decoder fused operation with full pipeline:
- CCL Broadcast -> RMSNorm -> Matmul -> Gather -> RMSNorm2 -> Matmul2 (shuffled) -> Matmul3 (Qnope only) & RoPE (Qrope only) -> Interleaved Pre-SDPA Output
- Qnope output: [64, 1, 512] after matmul3
- Qrope output: [64, 1, 64] after RoPE
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.tt.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.fused_ops.attention_block.op import AttentionBlock
from models.demos.deepseek_v3_b1.fused_ops.decoder_block.op import DecoderBlock
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode
from models.demos.deepseek_v3_b1.prepare_weights import (
    create_gate_indices_tensor,
    load_moe_decoder_layer,
    load_moe_routed_experts,
    prepare_dense_layer_weights,
    prepare_moe_layer_weights,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import (
    DENSE_LAYER_IDX,
    DENSE_SHARED_N,
    ROUTED_EXPERT_LAYER_IDX,
    RoutedExpert,
    SharedExpert,
    extract_routed_expert_output,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.test_post_sdpa import compute_forwarder_scratch_size
from models.demos.deepseek_v3_b1.tests.unit_tests.test_pre_sdpa import deinterleave_kv_cache
from models.demos.deepseek_v3_b1.utils import get_pinned_optimal_dram_bank_to_logical_worker_assignment


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


# ============================================================================
# Unified decoder block tensor setup (single BDW instance for all L1 weights)
# ============================================================================
def create_decoder_block_tensors(
    submesh,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    position_id,
    state_dict,
    layer_idx,
    max_seq_len,
    reduce_root_coord=ttnn.MeshCoordinate(1, 1),
    *,
    is_moe: bool = True,
    num_routed_experts: int = 0,
    preloaded_weights=None,
    rigged_experts: bool = False,
    input_override: torch.Tensor | None = None,
):
    """Create all tensors required by DecoderBlock.op().

    Three modes of operation:
    - **preloaded_weights mode** (production): pass a DeepSeekV3MoELayerWeights from
      load_moe_layer. Skips BDW allocation, weight processing, and golden tensors.
    - **state_dict + is_moe=True** (MoE tests): allocates BDW, calls
      prepare_moe_layer_weights, builds MoE golden tensors.
    - **state_dict + is_moe=False** (dense tests): allocates BDW, calls
      prepare_dense_layer_weights, builds dense golden tensors.

    Returns a dict with all attention + FFN + shared expert + reduce tensors.
    """
    if preloaded_weights is None and state_dict is None:
        raise ValueError("Either state_dict or preloaded_weights must be provided")
    torch.manual_seed(0)
    num_devices = mesh_rows * mesh_cols
    device_grid_size = submesh.compute_with_storage_grid_size()
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)

    # ── Constants for runtime tensors ──
    QNOPE_HEAD_DIM = 128
    QROPE_HEAD_DIM = 64
    QNOPE_OUT_DIM = 512
    KNOPE_DIM = 512
    KROPE_DIM = 64

    M = 1
    K = 7168
    output_size = 7168
    shape = (1, K)
    scale = (QNOPE_HEAD_DIM + QROPE_HEAD_DIM) ** -0.5
    kvpe_dim = KNOPE_DIM + KROPE_DIM

    QNOPE_GRID_COLS = 8
    QROPE_GRID_COLS = 4
    matmul2_grid_y = 8
    qrope_num_cores = QROPE_GRID_COLS * matmul2_grid_y

    NUM_SDPA_WORKERS = 8
    SDPA_L_HEIGHT = 8
    SDPA_L_WIDTH = 512 * NUM_SDPA_WORKERS
    SDPA_MS_WIDTH = 32 * NUM_SDPA_WORKERS

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    tile = ttnn.Tile([1, 32])

    kv_cache_branch_start_offset = (0, 8)
    kv_cache_branch_rope_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(8 + kv_cache_branch_start_offset[0], kv_cache_branch_start_offset[1]),
                ttnn.CoreCoord(8 + kv_cache_branch_start_offset[0], 1 + kv_cache_branch_start_offset[1]),
            )
        }
    )

    # ── SDPA KV cache buffer ──
    kv_cache_num_cores_x = device_grid_size.x
    kv_cache_num_cores_y = device_grid_size.y
    kv_cache_num_cores = kv_cache_num_cores_x * kv_cache_num_cores_y
    kv_cache_shard_height = 256
    kv_cache_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(kv_cache_num_cores_x - 1, kv_cache_num_cores_y - 1))}
        ),
        (kv_cache_shard_height, kvpe_dim),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_kv_cache_buffer = ttnn.from_torch(
        torch.randn((kv_cache_shard_height * kv_cache_num_cores, kvpe_dim), dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
        ),
        mesh_mapper=mesh_mapper,
    )

    # ── SDPA output intermediate buffer ──
    sdpa_out_interm_num_cores = device_grid_size.x * device_grid_size.y
    sdpa_out_interm_num_slots = 5  # MoE needs 36864 bytes/shard; 5 slots × 17 tiles × 512 = 43520
    sdpa_out_interm_shard_height = sdpa_out_interm_num_slots * 8
    sdpa_out_interm_shard_width = 17 * 32
    sdpa_out_interm_total_height = sdpa_out_interm_shard_height * sdpa_out_interm_num_cores
    sdpa_out_interm_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
        ),
        (sdpa_out_interm_shard_height, sdpa_out_interm_shard_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_out_interm_buffer = ttnn.from_torch(
        torch.zeros((sdpa_out_interm_total_height, sdpa_out_interm_shard_width), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_out_interm_shard_spec
        ),
        mesh_mapper=mesh_mapper,
        tile=ttnn.Tile([8, 32]),
    )

    if input_override is not None:
        torch_input = input_override.reshape(shape).to(torch.bfloat16)
    elif rigged_experts:
        # Use deterministic RMS-normalized input to avoid oversized constant-direction activations.
        # (all-ones can produce brittle saturation in downstream low-precision paths)
        torch_input_f32 = torch.randn(shape, dtype=torch.float32)
        torch_input_f32 = torch_input_f32 / torch.sqrt(torch_input_f32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        torch_input = torch_input_f32.to(torch.bfloat16)
    else:
        torch_input = torch.randn(shape, dtype=torch.bfloat16)

    rigged_group_ids = None
    rigged_expert_ids = None
    if rigged_experts and is_moe and preloaded_weights is None:
        # Deterministically pick 4 groups and 2 experts/group, then bias-rig gate selection.
        g = torch.Generator()
        g.manual_seed(2026)
        rigged_group_ids = torch.randperm(8, generator=g)[:4].tolist()
        rigged_expert_ids = {grp: torch.randperm(32, generator=g)[:2].tolist() for grp in rigged_group_ids}

        rigged_bias = torch.full((8, 32), -10.0, dtype=torch.bfloat16)
        for grp in rigged_group_ids:
            for exp in rigged_expert_ids[grp]:
                rigged_bias[grp, exp] = 10.0
        state_dict[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"] = rigged_bias.reshape(-1).contiguous()
        logger.info(
            f"Rigged experts enabled: groups={rigged_group_ids}, "
            f"experts={[(grp, rigged_expert_ids[grp]) for grp in rigged_group_ids]}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # All weights via a single BDW instance or preloaded
    # ══════════════════════════════════════════════════════════════════════════
    if preloaded_weights is not None:
        layer = preloaded_weights
    else:
        bdw = BlitzDecodeWeights(submesh)
        if is_moe:
            layer = prepare_moe_layer_weights(
                bdw,
                state_dict,
                layer_idx,
                num_routed_experts=num_routed_experts,
                move_to_device=True,
            )
        else:
            layer = prepare_dense_layer_weights(bdw, state_dict, layer_idx, move_to_device=True)

    # ── FFN final output config (DRAM streaming matmul output grid) ──
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(submesh, gate_proj_noc)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in gate_proj_worker_cores])
    num_gate_proj_cores = len(gate_proj_worker_cores)
    final_output_width_per_core = RoutedExpert.FINAL_OUTPUT_WIDTH_PER_CORE
    final_output_total_width = final_output_width_per_core * num_gate_proj_cores
    num_banks = submesh.dram_grid_size().x
    tile_w = RoutedExpert.TILE_W
    down_proj_N_padded = ((K + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    per_core_down_proj_N = down_proj_N_padded // num_banks

    final_output_shard_spec = ttnn.ShardSpec(
        gate_proj_core_ranges,
        (1, final_output_width_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    final_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, final_output_shard_spec
    )

    # ── MoE-only: gate indices and output buffers ──
    if is_moe:
        input_core = ttnn.CoreCoord(device_grid_size.x - 1, RoutedExpert.INPUT_CORE_Y)
        input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(input_core, input_core)])

        ttnn_gate_indices = create_gate_indices_tensor(submesh, input_core_grid, mesh_mapper=mesh_mapper)

        tile_1x16 = ttnn.Tile((1, 16))
        gate_output_shard_spec = ttnn.ShardSpec(input_core_grid, (1, 16), ttnn.ShardOrientation.ROW_MAJOR)
        gate_output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gate_output_shard_spec
        )
        gate_output_scores_tensor = ttnn.from_torch(
            torch.zeros((1, 16), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=gate_output_mem_config,
            tile=tile_1x16,
            mesh_mapper=mesh_mapper,
        )
        gate_output_indices_tensor = ttnn.from_torch(
            torch.zeros((1, 16), dtype=torch.uint16),
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=gate_output_mem_config,
            tile=tile_1x16,
            mesh_mapper=mesh_mapper,
        )
        moe_ref_gate_output_scores = ttnn.from_torch(
            torch.zeros((1, 16), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=gate_output_mem_config,
            tile=tile_1x16,
            mesh_mapper=mesh_mapper,
        )
        moe_ref_gate_output_indices = ttnn.from_torch(
            torch.zeros((1, 16), dtype=torch.uint16),
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=gate_output_mem_config,
            tile=tile_1x16,
            mesh_mapper=mesh_mapper,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Attention input/intermediate/output mesh tensors
    # ══════════════════════════════════════════════════════════════════════════
    sender_coord = ttnn.MeshCoordinate(sender_row, sender_col)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}), shape, ttnn.ShardOrientation.ROW_MAJOR
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    device_tensors = []
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            if row == sender_row and col == sender_col:
                device_tensors.append(torch_input)
            else:
                device_tensors.append(torch.zeros_like(torch_input))

    input_tensor_mesh = ttnn.from_torch(
        torch.cat(device_tensors, dim=0),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )

    # ── RoPE TTNN tensors ──
    qrope_dram_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    position_ids = torch.tensor([position_id])
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, QROPE_HEAD_DIM, 2, dtype=torch.float32) / QROPE_HEAD_DIM))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    torch_cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)
    torch_sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)
    torch_trans_mat = get_rot_transformation_mat()

    ttnn_qrope_cos = ttnn.from_torch(
        torch_cos.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_dram_mem,
        tile=tile,
        mesh_mapper=mesh_mapper,
    )
    ttnn_qrope_sin = ttnn.from_torch(
        torch_sin.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_dram_mem,
        tile=tile,
        mesh_mapper=mesh_mapper,
    )
    ttnn_krope_cos = ttnn.from_torch(
        torch_cos.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_dram_mem,
        tile=tile,
        mesh_mapper=mesh_mapper,
    )
    ttnn_krope_sin = ttnn.from_torch(
        torch_sin.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_dram_mem,
        tile=tile,
        mesh_mapper=mesh_mapper,
    )

    # ── Trans mat ──
    qrope_grid = ttnn.CoreRange(
        ttnn.CoreCoord(QNOPE_GRID_COLS, 0), ttnn.CoreCoord(QNOPE_GRID_COLS + QROPE_GRID_COLS - 1, matmul2_grid_y - 1)
    )
    trans_mat_crs = kv_cache_branch_rope_crs.merge(ttnn.CoreRangeSet({qrope_grid}))
    trans_tile = ttnn.Tile((ttnn.TILE_SIZE, ttnn.TILE_SIZE))
    trans_shard_spec = ttnn.ShardSpec(trans_mat_crs, (ttnn.TILE_SIZE, ttnn.TILE_SIZE), ttnn.ShardOrientation.ROW_MAJOR)
    trans_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, trans_shard_spec)
    trans_mat_replicated = torch_trans_mat.repeat(1, 1, qrope_num_cores + kv_cache_branch_rope_crs.num_cores(), 1)
    ttnn_trans_mat = ttnn.from_torch(
        trans_mat_replicated,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=trans_mem,
        tile=trans_tile,
        mesh_mapper=mesh_mapper,
    )

    # ── Position IDs ──
    pos_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
    )
    pos_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(pos_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_position_ids = ttnn.from_torch(
        torch.full((device_grid_size.x * device_grid_size.y, 1), position_id, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=pos_mem,
        mesh_mapper=mesh_mapper,
    )

    # ── KV Cache (ND sharded DRAM) ──
    program_config = FlashMLADecode.ProgramConfig(k_chunk_size=128, exp_approx_mode=False)
    grid = program_config.grid
    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, program_config.k_chunk_size, kvpe_dim],
        grid=grid.optimal_dram_grid(),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=kv_nd_shard_spec)
    num_sp = mesh_rows
    dcs = program_config.device_chunk_size
    torch_kv_cache = torch.zeros((1, 1, max_seq_len, kvpe_dim), dtype=torch.bfloat16)
    torch_kv_cache[:, :, :position_id, :] = torch.randn(1, 1, position_id, kvpe_dim, dtype=torch.bfloat16)
    torch_kv_cache_shuffled = deinterleave_kv_cache(torch_kv_cache, dcs, num_sp)
    kv_cache_2d_mesh_mapper = ttnn.ShardTensor2dMesh(submesh, mesh_shape=(mesh_rows, mesh_cols), dims=(2, None))
    ttnn_kv_cache = ttnn.from_torch(
        torch_kv_cache_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=kv_mem,
        mesh_mapper=kv_cache_2d_mesh_mapper,
    )
    kv_cache_bfp8_before_op = ttnn.to_torch(ttnn_kv_cache, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    # ── KV cache clone for standalone AttentionBlock.op sanity check ──
    ttnn_kv_cache_attn_ref = ttnn.from_torch(
        torch_kv_cache_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=kv_mem,
        mesh_mapper=kv_cache_2d_mesh_mapper,
    )

    # ── SDPA output tensor ──
    s1_cores, _ = FlashMLADecode.ProgramConfig.grid.BLOCKS[0]
    sdpa_input_output_grid_crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in s1_cores]
    )
    HEADS_PER_ROW = 8
    SDPA_INPUT_NUM_CORES = len(s1_cores)
    sdpa_tile = ttnn.Tile([8, 32])
    sdpa_input_output_shard_spec = ttnn.ShardSpec(
        sdpa_input_output_grid_crs, (HEADS_PER_ROW, QNOPE_OUT_DIM), ttnn.ShardOrientation.ROW_MAJOR
    )
    sdpa_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_input_output_shard_spec
    )
    ttnn_sdpa_output = ttnn.from_torch(
        torch.zeros((SDPA_INPUT_NUM_CORES * HEADS_PER_ROW, QNOPE_OUT_DIM), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=sdpa_mem,
        mesh_mapper=mesh_mapper,
        tile=sdpa_tile,
    )

    # ── Post-SDPA tensors ──
    a_tile = ttnn.Tile([M, 32])
    shard_mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    gather_core = ttnn.CoreCoord(12, 9)
    gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

    # ── Attention block output / MoE residual input (overlapped with sdpa_kv_cache_buffer) ──
    # These are temporally disjoint: the kv cache on core (12,9) is done after SDPA,
    # so the attention output and MoE residual input can reuse that L1 region.
    a_tile_size = a_tile.get_tile_size(ttnn.bfloat16)  # 1×32 tile → 64 bytes
    num_output_tiles = output_size // 32  # 7168 / 32 = 224 tiles
    output_shard_spec = ttnn.ShardSpec(gather_core_grid, (M, output_size), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    mesh_output_torch = torch.cat([torch.zeros((M, output_size), dtype=torch.bfloat16)] * num_devices, dim=0)
    attn_output = ttnn.from_torch(
        mesh_output_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=output_mem_config,
        mesh_mapper=shard_mesh_mapper,
    )
    attn_ref_output = ttnn.from_torch(
        mesh_output_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=output_mem_config,
        mesh_mapper=shard_mesh_mapper,
    )

    # ── SDPA worker/forwarder tensors ──
    sdpa_output_cores = FlashMLADecode.ProgramConfig.grid.output_cores(0, NUM_SDPA_WORKERS)
    sdpa_worker_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sdpa_output_cores]
    )
    sdpa_l_per_worker = SDPA_L_WIDTH // NUM_SDPA_WORKERS
    sdpa_ms_per_worker = SDPA_MS_WIDTH // NUM_SDPA_WORKERS

    sdpa_recv_per_worker = sdpa_l_per_worker + sdpa_ms_per_worker
    sdpa_recv_shard_shape = (2 * SDPA_L_HEIGHT, sdpa_recv_per_worker)
    sdpa_recv_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sdpa_worker_grid, sdpa_recv_shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )
    sdpa_recv_full_width = sdpa_recv_per_worker * NUM_SDPA_WORKERS
    mesh_recv = torch.cat(
        [torch.zeros((2 * SDPA_L_HEIGHT, sdpa_recv_full_width), dtype=torch.bfloat16)] * num_devices, dim=0
    )
    ttnn_sdpa_intermediate_recv = ttnn.from_torch(
        mesh_recv,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_recv_mem,
        tile=sdpa_tile,
        mesh_mapper=shard_mesh_mapper,
    )

    sdpa_forwarder_cores = [ttnn.CoreCoord(9, 8), ttnn.CoreCoord(10, 8)]
    sdpa_forwarder_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in sdpa_forwarder_cores])
    sdpa_fwd_buffer_bytes = compute_forwarder_scratch_size(
        batch_size=SDPA_L_HEIGHT,
        l_width=sdpa_l_per_worker,
        num_cores=NUM_SDPA_WORKERS,
    )
    sdpa_fwd_total_elements = sdpa_fwd_buffer_bytes // 2
    sdpa_fwd_per_forwarder = sdpa_fwd_total_elements // 2
    sdpa_forwarder_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sdpa_forwarder_grid, (1, sdpa_fwd_per_forwarder), ttnn.ShardOrientation.ROW_MAJOR),
    )
    mesh_fwd_scratch = torch.cat([torch.zeros((1, sdpa_fwd_total_elements), dtype=torch.bfloat16)] * num_devices, dim=0)
    ttnn_sdpa_forwarder_scratch = ttnn.from_torch(
        mesh_fwd_scratch,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=sdpa_forwarder_mem,
        mesh_mapper=shard_mesh_mapper,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Reduce-to-one tensors
    # ══════════════════════════════════════════════════════════════════════════
    reduce_mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh.shape)
    reduce_mesh_mapper = ttnn.create_mesh_mapper(submesh, reduce_mesh_mapper_config)
    tile_1x32 = ttnn.Tile([1, 32])

    # Single intermediate tensor with 3x shard width for all 3 reduction rounds
    orig_shard_spec = final_output_mem_config.shard_spec
    intermediate_shard_shape = [orig_shard_spec.shape[0], orig_shard_spec.shape[1] * 3]
    intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            orig_shard_spec.grid,
            intermediate_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    intermediate_tensors = ttnn.from_torch(
        torch.zeros([4, 2, final_output_total_width * 3], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=intermediate_mem_config,
        tile=tile_1x32,
        mesh_mapper=reduce_mesh_mapper,
    )

    shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
    aggregator_core = shard_cores_list[0]
    reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(aggregator_core, aggregator_core)})
    reduce_output_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(reduce_output_shard_grid, (1, final_output_total_width), ttnn.ShardOrientation.ROW_MAJOR),
    )
    reduce_output_tensor = ttnn.from_torch(
        torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=reduce_output_mem,
        tile=tile_1x32,
        mesh_mapper=reduce_mesh_mapper,
    )

    # ── Standalone MoE ref reduce tensors (MoE only) ──
    if is_moe:
        moe_ref_reduce_intermediate = ttnn.from_torch(
            torch.zeros([4, 2, final_output_total_width * 3], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=intermediate_mem_config,
            tile=tile_1x32,
            mesh_mapper=reduce_mesh_mapper,
        )
        moe_ref_reduce_output = ttnn.from_torch(
            torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=reduce_output_mem,
            tile=tile_1x32,
            mesh_mapper=reduce_mesh_mapper,
        )

    sender_core_from_residual = attn_output.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core_from_residual)])

    # ══════════════════════════════════════════════════════════════════════════
    # Golden model PyTorch tensors (only when state_dict is available)
    # ══════════════════════════════════════════════════════════════════════════
    golden = {}
    if state_dict is not None:

        def _sd_key(suffix):
            return f"model.layers.{layer_idx}.{suffix}"

        golden_torch_gamma = state_dict[_sd_key("input_layernorm.weight")].unsqueeze(0)
        golden_torch_matmul_weights = state_dict[_sd_key("self_attn.q_a_proj.weight")].T.contiguous()
        golden_torch_rmsnorm2_gamma = state_dict[_sd_key("self_attn.q_a_layernorm.weight")].unsqueeze(0)
        golden_torch_matmul2_weights = state_dict[_sd_key("self_attn.q_b_proj.weight")].T.contiguous()
        golden_torch_dkv_matmul_weights = state_dict[_sd_key("self_attn.kv_a_proj_with_mqa.weight")].T.contiguous()
        golden_torch_dkv_rmsnorm_gamma = state_dict[_sd_key("self_attn.kv_a_layernorm.weight")].unsqueeze(0)
        golden_torch_o_proj_weights = state_dict[_sd_key("self_attn.o_proj.weight")].T.contiguous()

        V_HEAD_DIM = 128
        KV_B_PROJ_HEAD_DIM = QNOPE_HEAD_DIM + V_HEAD_DIM  # 256
        kv_b_proj_raw = state_dict[_sd_key("self_attn.kv_b_proj.weight")]
        kv_b_out, kv_lora_rank = kv_b_proj_raw.shape
        total_kv_heads = kv_b_out // KV_B_PROJ_HEAD_DIM
        kv_b_3d = kv_b_proj_raw.reshape(total_kv_heads, KV_B_PROJ_HEAD_DIM, kv_lora_rank).contiguous()
        golden_kv_b1 = kv_b_3d[:, :QNOPE_HEAD_DIM, :].reshape(-1, kv_lora_rank)
        golden_kv_b2 = kv_b_3d[:, QNOPE_HEAD_DIM:, :].reshape(-1, kv_lora_rank).T.contiguous()
        golden_torch_matmul3_weights = golden_kv_b1.reshape(total_kv_heads, QNOPE_HEAD_DIM, kv_lora_rank)

        golden_total_qnope_heads = total_kv_heads
        golden_total_qrope_heads = total_kv_heads

        # ── Golden FFN tensors (MoE vs dense differ in key paths and weight layout) ──
        golden_moe_rmsnorm_gamma = (
            state_dict[_sd_key("post_attention_layernorm.weight")].reshape(1, K).to(torch.bfloat16).float()
        )
        if is_moe:
            golden_moe_shared_gate = state_dict[_sd_key("mlp.shared_experts.gate_proj.weight")].T.contiguous()
            golden_moe_shared_up = state_dict[_sd_key("mlp.shared_experts.up_proj.weight")].T.contiguous()
            golden_moe_shared_down = state_dict[_sd_key("mlp.shared_experts.down_proj.weight")].T.contiguous()
            golden_moe_routing_weights = state_dict[_sd_key("mlp.gate.weight")].T.contiguous()
            golden_moe_bias = (
                state_dict[_sd_key("mlp.gate.e_score_correction_bias")]
                .reshape(1, 8, 32)
                .contiguous()
                .to(torch.bfloat16)
            )
            golden_moe_gate_proj_dict = {}
            golden_moe_up_proj_dict = {}
            golden_moe_down_proj_dict = {}
            for e in range(num_routed_experts):
                w_g = state_dict[_sd_key(f"mlp.experts.{e}.gate_proj.weight")].T.contiguous()
                golden_moe_gate_proj_dict[e] = w_g.reshape(1, 1, K, -1)
                w_u = state_dict[_sd_key(f"mlp.experts.{e}.up_proj.weight")].T.contiguous()
                golden_moe_up_proj_dict[e] = w_u.reshape(1, 1, K, -1)
                w_d = state_dict[_sd_key(f"mlp.experts.{e}.down_proj.weight")].T.contiguous()
                golden_moe_down_proj_dict[e] = w_d.reshape(1, 1, -1, K)
        else:
            gate_full = state_dict[_sd_key("mlp.gate_proj.weight")].T.contiguous()
            up_full = state_dict[_sd_key("mlp.up_proj.weight")].T.contiguous()
            down_full = state_dict[_sd_key("mlp.down_proj.weight")].T.contiguous()
            golden_moe_shared_gate = gate_full[:, :DENSE_SHARED_N].contiguous()
            golden_moe_shared_up = up_full[:, :DENSE_SHARED_N].contiguous()
            golden_moe_shared_down = down_full[:DENSE_SHARED_N, :].contiguous()
            golden_moe_routing_weights = None
            golden_moe_bias = None
            golden_moe_gate_proj_dict = {}
            golden_moe_up_proj_dict = {}
            golden_moe_down_proj_dict = {}
            for e in range(8):
                start = DENSE_SHARED_N + e * RoutedExpert.GATE_PROJ_N
                end = start + RoutedExpert.GATE_PROJ_N
                golden_moe_gate_proj_dict[e] = gate_full[:, start:end].reshape(1, 1, K, -1)
                golden_moe_up_proj_dict[e] = up_full[:, start:end].reshape(1, 1, K, -1)
                golden_moe_down_proj_dict[e] = down_full[start:end, :].reshape(1, 1, -1, K)

        golden = {
            "golden_torch_input": torch_input,
            "golden_torch_gamma": golden_torch_gamma,
            "golden_torch_matmul_weights": golden_torch_matmul_weights,
            "golden_torch_rmsnorm2_gamma": golden_torch_rmsnorm2_gamma,
            "golden_torch_matmul2_weights": golden_torch_matmul2_weights,
            "golden_torch_matmul3_weights": golden_torch_matmul3_weights,
            "golden_torch_sin": torch_sin,
            "golden_torch_cos": torch_cos,
            "golden_position_ids": position_ids,
            "golden_torch_dkv_matmul_weights": golden_torch_dkv_matmul_weights,
            "golden_torch_dkv_rmsnorm_gamma": golden_torch_dkv_rmsnorm_gamma,
            "golden_torch_kv_cache": torch_kv_cache,
            "golden_scale": scale,
            "golden_torch_kv_b2_proj_weights": golden_kv_b2,
            "golden_torch_o_proj_weights": golden_torch_o_proj_weights,
            "golden_total_qnope_heads": golden_total_qnope_heads,
            "golden_total_qrope_heads": golden_total_qrope_heads,
            "golden_kv_cache_bfp8_before_op": kv_cache_bfp8_before_op,
            "golden_sdpa_slice_size": SDPA_INPUT_NUM_CORES * HEADS_PER_ROW,
            "golden_moe_rmsnorm_gamma": golden_moe_rmsnorm_gamma,
            "golden_moe_shared_gate": golden_moe_shared_gate,
            "golden_moe_shared_up": golden_moe_shared_up,
            "golden_moe_shared_down": golden_moe_shared_down,
            "golden_moe_routing_weights": golden_moe_routing_weights,
            "golden_moe_bias": golden_moe_bias,
            "golden_moe_gate_proj_dict": golden_moe_gate_proj_dict,
            "golden_moe_up_proj_dict": golden_moe_up_proj_dict,
            "golden_moe_down_proj_dict": golden_moe_down_proj_dict,
            "rigged_group_ids": rigged_group_ids,
            "rigged_expert_ids": rigged_expert_ids,
        }

    # ── Routed weight tensors differ between MoE (list) and dense (single tensor) ──
    routed_gate = layer.routed_gate_proj[0] if is_moe else layer.routed_gate_proj
    routed_up = layer.routed_up_proj[0] if is_moe else layer.routed_up_proj
    routed_down = layer.routed_down_proj[0] if is_moe else layer.routed_down_proj

    result = {
        # Attention weights (from prepare_*_layer_weights via BDW)
        "gamma_overlapped": layer.attn_norm,
        "matmul_weights_overlapped": layer.q_a_proj,
        "rmsnorm2_gamma_overlapped": layer.q_norm,
        "matmul2_weights_overlapped": layer.q_b_proj,
        "matmul3_weights_overlapped": layer.kv_b1_proj,
        "dkv_matmul_weights_overlapped": layer.kv_a_proj,
        "dkv_rmsnorm_gamma_overlapped": layer.kv_norm,
        "kv_b2_overlapped": layer.kv_b2_proj,
        "o_proj_overlapped": layer.o_proj,
        "ffn_norm_overlapped": layer.ffn_norm,
        # Attention activation/buffer tensors
        "input_tensor_mesh": input_tensor_mesh,
        "ttnn_qrope_sin": ttnn_qrope_sin,
        "ttnn_qrope_cos": ttnn_qrope_cos,
        "ttnn_trans_mat": ttnn_trans_mat,
        "ttnn_krope_cos": ttnn_krope_cos,
        "ttnn_krope_sin": ttnn_krope_sin,
        "ttnn_kv_cache": ttnn_kv_cache,
        "ttnn_kv_cache_attn_ref": ttnn_kv_cache_attn_ref,
        "ttnn_position_ids": ttnn_position_ids,
        "scale": scale,
        "sdpa_kv_cache_buffer": sdpa_kv_cache_buffer,
        "sdpa_out_interm_buffer": sdpa_out_interm_buffer,
        "ttnn_sdpa_output": ttnn_sdpa_output,
        "sender_coord": sender_coord,
        "ttnn_sdpa_input_l": None,
        "ttnn_sdpa_input_ms": None,
        "ttnn_sdpa_output_l": None,
        "ttnn_sdpa_intermediate_recv": ttnn_sdpa_intermediate_recv,
        "ttnn_sdpa_forwarder_scratch": ttnn_sdpa_forwarder_scratch,
        "device_chunk_size": program_config.device_chunk_size,
        "ttnn_attention_block_output": attn_output,
        "ttnn_attn_ref_output": attn_ref_output,
        # FFN tensors (attn_output IS the FFN residual input — overlapped with kv cache)
        "ttnn_residual_mcast_src": attn_output,
        "gate_proj_weights": routed_gate,
        "up_proj_weights": routed_up,
        "down_proj_weights": routed_down,
        "final_output_mem_config": final_output_mem_config,
        "final_output_total_width": final_output_total_width,
        # Shared expert weights
        "shared_gate_weights_overlapped": layer.shared_gate_proj,
        "shared_up_weights_overlapped": layer.shared_up_proj,
        "shared_down_weights_tensor": layer.shared_down_proj,
        "shared_k_parallel": SharedExpert.K_PARALLEL,
        "shared_n_parallel": SharedExpert.N_PARALLEL,
        # Reduce-to-one
        "reduce_intermediate_tensors": intermediate_tensors,
        "reduce_output_tensor": reduce_output_tensor,
        "reduce_root_coord": reduce_root_coord,
        "num_gate_proj_cores": num_gate_proj_cores,
        "per_core_down_proj_N": per_core_down_proj_N,
        "mcast_grid": mcast_grid,
        **golden,
    }
    # MoE-only keys
    if is_moe:
        result.update(
            {
                "gate_mm_overlapped": layer.gate_mm,
                "ttnn_gate_bias": layer.gate_bias,
                "ttnn_gate_indices": ttnn_gate_indices,
                "gate_output_scores_tensor": gate_output_scores_tensor,
                "gate_output_indices_tensor": gate_output_indices_tensor,
                "moe_ref_gate_output_scores": moe_ref_gate_output_scores,
                "moe_ref_gate_output_indices": moe_ref_gate_output_indices,
                "moe_ref_reduce_intermediate": moe_ref_reduce_intermediate,
                "moe_ref_reduce_output": moe_ref_reduce_output,
            }
        )
    return result


@pytest.mark.parametrize(
    "sender_row, sender_col",
    [
        (1, 0),
    ],
)
@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("use_fp32", [False])
@pytest.mark.parametrize("bcast_cluster_axis", [0])
@pytest.mark.parametrize("bcast_secondary_cluster_axis", [1])
@pytest.mark.parametrize("reduce_cluster_axis", [1])
@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2)])
@pytest.mark.parametrize("num_iters", [(1)])
@pytest.mark.parametrize("max_seq_len", [32 * 1024])
@pytest.mark.parametrize(
    "position_id",
    [
        0,
        127,
        511,
        1023,
        2047,
        4096,  # (1 + partial,1,1,1): partial into dev0 (if SP = 4)
        pytest.param(6644, marks=pytest.mark.skip_post_commit),  # (2,2,1 + partial,1): partial into dev2 (if SP = 4)
        pytest.param(9916, marks=pytest.mark.skip_post_commit),  # (3,2 + partial,2,2): partial into dev1 (if SP = 4)
        pytest.param(11664, marks=pytest.mark.skip_post_commit),  # (3,3,3,2 + partial): partial into dev3 (if SP = 4)
    ],
)  # Must test 128 chunk aligned decode positions, add other tests when causal masks are in for SDPA
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
            "worker_l1_size": 1431568,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.parametrize("num_internal_iterations", [1])
@pytest.mark.parametrize("rigged_experts", [False, True], ids=["normal_gate", "fixed_experts"])
@pytest.mark.parametrize(
    "enable_routing, use_hardcoded_expert_index, num_routed_experts",
    [
        # (True, True, 8),
        pytest.param(True, False, 256, marks=pytest.mark.skip_post_commit),
    ],
    ids=[
        # "hardcoded_experts",
        "full_routing",
    ],
)
@pytest.mark.parametrize(
    "validate_standalone_mla",
    [pytest.param(True, marks=pytest.mark.skip_post_commit), False],
    ids=["validate_standalone_mla", "just_decoder_mla"],
)
@pytest.mark.parametrize(
    "validate_standalone_moe",
    [pytest.param(True, marks=pytest.mark.skip_post_commit), False],
    ids=["validate_standalone_moe", "just_decoder_moe"],
)
@pytest.mark.requires_grid_size((13, 10))
def test_decoder(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    epsilon,
    use_fp32,
    bcast_cluster_axis,
    bcast_secondary_cluster_axis,
    reduce_cluster_axis,
    num_iters,
    max_seq_len,
    position_id,
    noc_mode,
    num_internal_iterations,
    rigged_experts,
    enable_routing,
    use_hardcoded_expert_index,
    num_routed_experts,
    validate_standalone_mla,
    validate_standalone_moe,
    get_reference_model_state_dict,
):
    """Test TTNN decoder fused operation with CCL broadcast, kv cache, mla, reduce, residual add"""
    torch.manual_seed(0)
    num_devices = mesh_rows * mesh_cols
    logger.info(f"Number of devices: {num_devices}")
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than available")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()

    logger.info("Preparing model state dict...")
    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=num_routed_experts,
    )

    logger.info("Creating decoder block tensors...")
    d = create_decoder_block_tensors(
        submesh,
        mesh_rows,
        mesh_cols,
        sender_row,
        sender_col,
        position_id,
        state_dict,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        max_seq_len=max_seq_len,
        is_moe=True,
        num_routed_experts=num_routed_experts,
        rigged_experts=rigged_experts,
    )

    num_cores = device_grid_size.x * device_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, device_grid_size, row_wise=True)
    ttnn.synchronize_device(submesh)
    reduce_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(4)]
    persistent_next_iter_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 1)
    ttnn.synchronize_device(submesh)

    attn_semaphores = AttentionBlock.create_semaphores(submesh)
    moe_semaphores = MoeOp.create_semaphores(submesh)

    # ========================================================================
    # Run standalone AttentionBlock.op as sanity reference (uses cloned KV cache)
    # ========================================================================
    ttnn_attn_ref_output_torch = None
    if validate_standalone_mla:
        logger.info(f"Running standalone AttentionBlock.op with position_id={position_id}...")
        attn_ref_semaphores = AttentionBlock.create_semaphores(submesh)
        ttnn_attn_ref_result = AttentionBlock.op(
            d["input_tensor_mesh"],
            d["gamma_overlapped"],
            d["matmul_weights_overlapped"],
            d["rmsnorm2_gamma_overlapped"],
            d["matmul2_weights_overlapped"],
            d["matmul3_weights_overlapped"],
            d["ttnn_qrope_sin"],
            d["ttnn_qrope_cos"],
            d["ttnn_trans_mat"],
            d["ttnn_krope_cos"],
            d["ttnn_krope_sin"],
            d["dkv_matmul_weights_overlapped"],
            d["dkv_rmsnorm_gamma_overlapped"],
            d["ttnn_kv_cache_attn_ref"],
            d["ttnn_position_ids"],
            d["scale"],
            d["ttnn_sdpa_output"],
            d["sdpa_kv_cache_buffer"],
            d["sdpa_out_interm_buffer"],
            d["sender_coord"],
            d["kv_b2_overlapped"],
            d["o_proj_overlapped"],
            d["ttnn_sdpa_input_l"],
            d["ttnn_sdpa_input_ms"],
            d["ttnn_sdpa_output_l"],
            d["ttnn_sdpa_intermediate_recv"],
            d["ttnn_sdpa_forwarder_scratch"],
            d["device_chunk_size"],
            d["ttnn_attn_ref_output"],
            attn_ref_semaphores,
            bcast_cluster_axis,
            bcast_secondary_cluster_axis,
            reduce_cluster_axis,
            0,  # sdpa_cluster_axis
            1,  # num_links
            epsilon,
            use_fp32,
            False,  # skip_ccl
            noc_mode,
            num_iterations=1,
        )
        ttnn.synchronize_device(submesh)
        ttnn_attn_ref_output_torch = ttnn.to_torch(
            d["ttnn_attn_ref_output"], mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
        )
        logger.info("Standalone AttentionBlock.op completed.")

    # ========================================================================
    # Run decoder operation
    # ========================================================================
    logger.info(f"Running decoder operation with position_id={position_id}...")
    for i in range(num_iters):
        moe_final_output_tensor, attention_block_output_tensor = DecoderBlock.op(
            # AttentionBlock parameters
            d["input_tensor_mesh"],
            d["gamma_overlapped"],
            d["matmul_weights_overlapped"],
            d["rmsnorm2_gamma_overlapped"],
            d["matmul2_weights_overlapped"],
            d["matmul3_weights_overlapped"],
            d["ttnn_qrope_sin"],
            d["ttnn_qrope_cos"],
            d["ttnn_trans_mat"],
            d["ttnn_krope_cos"],
            d["ttnn_krope_sin"],
            d["dkv_matmul_weights_overlapped"],
            d["dkv_rmsnorm_gamma_overlapped"],
            d["ttnn_kv_cache"],
            d["ttnn_position_ids"],
            d["scale"],
            d["sdpa_kv_cache_buffer"],
            d["sdpa_out_interm_buffer"],
            d["sender_coord"],
            # Post-SDPA parameters
            # Post-SDPA
            d["kv_b2_overlapped"],
            d["o_proj_overlapped"],
            d["ttnn_sdpa_input_l"],
            d["ttnn_sdpa_input_ms"],
            d["ttnn_sdpa_output_l"],
            d["ttnn_sdpa_intermediate_recv"],
            d["ttnn_sdpa_forwarder_scratch"],
            d["device_chunk_size"],
            d["ttnn_attention_block_output"],
            attention_block_semaphores=attn_semaphores,
            # MoE parameters
            shared_residual_mcast_src_tensor=d["ttnn_residual_mcast_src"],
            gate_mm_weights_tensor=d["gate_mm_overlapped"],
            gate_bias_tensor=d["ttnn_gate_bias"],
            gate_indices_tensor=d["ttnn_gate_indices"],
            gate_output_scores_tensor=d["gate_output_scores_tensor"],
            gate_output_indices_tensor=d["gate_output_indices_tensor"],
            gate_proj_weights_tensor=d["gate_proj_weights"],
            up_proj_weights_tensor=d["up_proj_weights"],
            down_proj_weights_tensor=d["down_proj_weights"],
            moe_final_output_tensor=None,
            rmsnorm_gamma_tensor=d["ffn_norm_overlapped"],
            shared_gate_weights_overlapped=d["shared_gate_weights_overlapped"],
            shared_up_weights_overlapped=d["shared_up_weights_overlapped"],
            shared_down_weights_tensor=d["shared_down_weights_tensor"],
            shared_k_parallel=d["shared_k_parallel"],
            shared_n_parallel=d["shared_n_parallel"],
            moe_semaphores=moe_semaphores,
            reduce_intermediate_tensors=d["reduce_intermediate_tensors"],
            reduce_output_tensor=d["reduce_output_tensor"],
            reduce_semaphores=reduce_semaphores,
            reduce_root_coord=d["reduce_root_coord"],
            # Shared parameters
            enable_routing=True,
            bcast_cluster_axis=bcast_cluster_axis,
            bcast_secondary_cluster_axis=bcast_secondary_cluster_axis,
            reduce_cluster_axis=reduce_cluster_axis,
            sdpa_cluster_axis=0,  # sdpa_cluster_axis
            num_links=1,  # num_links
            epsilon=epsilon,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
            noc_mode=noc_mode,
            num_iterations=num_internal_iterations,
            upstream_socket=None,
            downstream_socket=None,
            persistent_next_iter_semaphore=persistent_next_iter_semaphore,
            persistent_mode=True,
        )
    ttnn.synchronize_device(submesh)

    kv_cache_output_torch = ttnn.to_torch(d["ttnn_kv_cache"], mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    ttnn_attention_output = ttnn.to_torch(
        attention_block_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
    )

    # ========================================================================
    # Extract decoder MoE output and reduce root info (always needed for golden)
    # ========================================================================
    decoder_moe_output = ttnn.to_torch(moe_final_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    root_coord_tuple = d["reduce_root_coord"]
    root_device_idx = root_coord_tuple[0] * mesh_cols + root_coord_tuple[1]
    decoder_moe_output_root = decoder_moe_output[root_device_idx]
    decoder_moe_output_valid = extract_routed_expert_output(
        decoder_moe_output_root.unsqueeze(0),
        d["num_gate_proj_cores"],
        RoutedExpert.FINAL_OUTPUT_WIDTH_PER_CORE,
        d["per_core_down_proj_N"],
    )

    # ========================================================================
    # Run standalone MoeOp.op on MLA output (validate golden MoE path)
    # ========================================================================
    moe_device_output_valid = None
    if validate_standalone_moe:
        logger.info(
            f"Running standalone MoeOp.op (enable_routing={enable_routing}, hardcoded={use_hardcoded_expert_index})..."
        )
        moe_ref_semaphores = MoeOp.create_semaphores(submesh)
        moe_ref_reduce_sems = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(4)]
        ttnn.synchronize_device(submesh)

        moe_ref_result = MoeOp.op(
            attention_block_output_tensor,
            gate_mm_weights_tensor=d["gate_mm_overlapped"],
            gate_bias_tensor=d["ttnn_gate_bias"],
            gate_indices_tensor=d["ttnn_gate_indices"],
            gate_output_scores_tensor=d["moe_ref_gate_output_scores"],
            gate_output_indices_tensor=d["moe_ref_gate_output_indices"],
            gate_proj_weights_tensor=d["gate_proj_weights"],
            up_proj_weights_tensor=d["up_proj_weights"],
            down_proj_weights_tensor=d["down_proj_weights"],
            rmsnorm_gamma_tensor=d["ffn_norm_overlapped"],
            shared_gate_weights_overlapped=d["shared_gate_weights_overlapped"],
            shared_up_weights_overlapped=d["shared_up_weights_overlapped"],
            shared_down_weights_tensor=d["shared_down_weights_tensor"],
            shared_k_parallel=d["shared_k_parallel"],
            shared_n_parallel=d["shared_n_parallel"],
            enable_routing=enable_routing,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            sdpa_kv_cache_buffer=d["sdpa_kv_cache_buffer"],
            sdpa_out_interm_buffer=d["sdpa_out_interm_buffer"],
            num_iterations=1,
            reduce_intermediate_tensors=d["moe_ref_reduce_intermediate"],
            reduce_output_tensor=d["moe_ref_reduce_output"],
            reduce_semaphores=moe_ref_reduce_sems,
            reduce_root_coord=ttnn.MeshCoordinate(d["reduce_root_coord"]),
            semaphores=moe_ref_semaphores,
            noc_mode=noc_mode,
        )
        ttnn.synchronize_device(submesh)
        logger.info("Standalone MoeOp.op completed.")

        if enable_routing:
            moe_ref_scores_tensor, moe_ref_indices_tensor, moe_ref_result = moe_ref_result
            moe_ref_scores_torch = ttnn.to_torch(
                moe_ref_scores_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
            )
            moe_ref_indices_torch = ttnn.to_torch(
                moe_ref_indices_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
            )
        else:
            moe_ref_scores_torch = None
            moe_ref_indices_torch = None

        moe_reduce_torch = ttnn.to_torch(moe_ref_result, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
        moe_reduce_root = moe_reduce_torch[root_device_idx]
        moe_device_output_valid = extract_routed_expert_output(
            moe_reduce_root.unsqueeze(0),
            d["num_gate_proj_cores"],
            RoutedExpert.FINAL_OUTPUT_WIDTH_PER_CORE,
            d["per_core_down_proj_N"],
        )

    # ========================================================================
    # Compute golden reference
    # ========================================================================
    logger.info("Computing golden reference...")

    device_chunk_size = d["device_chunk_size"]
    num_sp = mesh_rows
    owning_sp_device = (position_id // device_chunk_size) % num_sp

    QNOPE_HEAD_DIM = 128
    QROPE_HEAD_DIM = 64
    KNOPE_DIM = 512
    KROPE_DIM = 64
    HEADS_PER_ROW = 8

    full_q, golden_new_kv, mla_output, moe_scores, moe_indices, moe_output = DecoderBlock.golden(
        d["golden_torch_input"],
        d["golden_torch_gamma"],
        d["golden_torch_matmul_weights"],
        d["golden_torch_rmsnorm2_gamma"],
        d["golden_torch_matmul2_weights"],
        d["golden_torch_matmul3_weights"],
        d["golden_torch_sin"],
        d["golden_torch_cos"],
        d["golden_position_ids"],
        d["golden_torch_dkv_matmul_weights"],
        d["golden_torch_dkv_rmsnorm_gamma"],
        d["golden_torch_kv_cache"],
        d["golden_scale"],
        d["golden_torch_kv_b2_proj_weights"],
        d["golden_torch_o_proj_weights"],
        epsilon=epsilon,
        num_qnope_heads=d["golden_total_qnope_heads"],
        num_qrope_heads=d["golden_total_qrope_heads"],
        qnope_head_dim=QNOPE_HEAD_DIM,
        qrope_head_dim=QROPE_HEAD_DIM,
        heads_per_row=HEADS_PER_ROW,
        nope_dim=KNOPE_DIM,
        rope_dim=KROPE_DIM,
        # MoE golden parameters
        moe_shared_gate_weights=d["golden_moe_shared_gate"],
        moe_shared_up_weights=d["golden_moe_shared_up"],
        moe_shared_down_weights=d["golden_moe_shared_down"],
        moe_gate_proj_weights_dict=d["golden_moe_gate_proj_dict"],
        moe_up_proj_weights_dict=d["golden_moe_up_proj_dict"],
        moe_down_proj_weights_dict=d["golden_moe_down_proj_dict"],
        moe_rmsnorm_gamma=d["golden_moe_rmsnorm_gamma"],
        moe_rmsnorm_epsilon=epsilon,
        moe_routing_weights=d["golden_moe_routing_weights"],
        moe_bias=d["golden_moe_bias"],
        moe_gate_eps=RoutedExpert.GATE_EPS,
        moe_gate_scaling_factor=RoutedExpert.GATE_SCALING_FACTOR,
        moe_enable_routing=enable_routing,
    )

    logger.info(f"MLA output: {mla_output}")

    logger.info(f"Golden computed (owning_sp_device={owning_sp_device}, device_chunk_size={device_chunk_size})")

    def get_local_seq_len(sp_idx):
        """Return how many KV positions SP device sp_idx holds for the current global position_id."""
        sp_block = device_chunk_size * num_sp
        num_full_blocks = position_id // sp_block
        remainder = position_id % sp_block
        dev_start = sp_idx * device_chunk_size
        dev_end = dev_start + device_chunk_size
        dev_contrib = max(0, min(remainder, dev_end) - dev_start)
        return num_full_blocks * device_chunk_size + dev_contrib

    # ========================================================================
    # Validate KV cache outputs (per SP device)
    # ========================================================================
    for device_idx in range(mesh_rows * mesh_cols):
        sp_group = device_idx // mesh_cols
        local_seq_len = get_local_seq_len(sp_group)

        if local_seq_len == 0 and sp_group != owning_sp_device:
            logger.info(f"Device {device_idx} (SP={sp_group}) no data yet, skipped")
            continue

        assert torch.equal(
            d["golden_kv_cache_bfp8_before_op"][device_idx, ..., :local_seq_len, :],
            kv_cache_output_torch[device_idx, ..., :local_seq_len, :],
        ), f"Device {device_idx} (SP={sp_group}) KV Cache before and after op mismatch"
        logger.info(f"Device {device_idx} (SP={sp_group}) old cache validation passed")

        if sp_group == owning_sp_device:
            compare_kv_cache = kv_cache_output_torch[device_idx, ..., local_seq_len, :]
            expected_nope = golden_new_kv[..., :KNOPE_DIM]
            expected_rope = golden_new_kv[..., KNOPE_DIM:]
            compare_nope = compare_kv_cache[..., :KNOPE_DIM]
            compare_rope = compare_kv_cache[..., KNOPE_DIM:]

            nope_passing, nope_pcc = comp_pcc(compare_nope, expected_nope, 0.98)
            logger.info(f"Device {device_idx} (SP={sp_group}) KV Cache NOPE PCC: {nope_pcc}")
            assert nope_passing, f"Device {device_idx} (SP={sp_group}) KV Cache NOPE PCC check failed: {nope_pcc}"

            rope_passing, rope_pcc = comp_pcc(compare_rope, expected_rope, 0.98)
            logger.info(f"Device {device_idx} (SP={sp_group}) KV Cache ROPE PCC: {rope_pcc}")
            assert rope_passing, f"Device {device_idx} (SP={sp_group}) KV Cache ROPE PCC check failed: {rope_pcc}"

    if moe_scores is not None:
        logger.info(f"Golden MoE scores: {moe_scores}")
        logger.info(f"Golden MoE indices: {moe_indices}")

    # ========================================================================
    # Validate decoder MLA output (full pipeline)
    # ========================================================================
    for device_idx in range(mesh_rows * mesh_cols):
        received = ttnn_attention_output[device_idx : device_idx + 1, :]
        passing, pcc = comp_pcc(mla_output, received, 0.98)
        logger.info(f"Device {device_idx} DecoderBlock Output PCC: {pcc}")
        logger.info(f"Golden MLA output: {mla_output}")
        logger.info(f"DecoderBlock MLA output: {received}")
        if validate_standalone_mla:
            pure_mla = ttnn_attn_ref_output_torch[device_idx : device_idx + 1, :]
            pure_mla_passing, pure_mla_pcc = comp_pcc(mla_output, pure_mla, 0.98)
            logger.info(f"Pure MLA PCC: {pure_mla_pcc}")
            logger.info(f"Pure MLA output: {pure_mla}")
        assert passing, f"Device {device_idx} DecoderBlock Output PCC check failed: {pcc}"

    # ========================================================================
    # Validate MoE output vs DecoderBlock golden MoE output
    # ========================================================================
    # Golden with moe_num_devices>1 computes per-device golden with TP-sharded shared
    # weights and per-device expert indices, then sums — matching reduce-to-one exactly.
    passing, pcc = comp_pcc(moe_output.flatten(), decoder_moe_output_valid.flatten(), 0.97)
    logger.info(f"MoE PCC (decoder vs golden): {pcc}")
    logger.info(f"Golden MoE output: {moe_output.flatten()[:8]}")
    logger.info(f"DecoderBlock MoE output: {decoder_moe_output_valid.flatten()[:8]}")

    if validate_standalone_moe:
        pure_moe_passing, pure_moe_pcc = comp_pcc(moe_output.flatten(), moe_device_output_valid.flatten(), 0.97)
        logger.info(f"Pure MoE PCC (standalone vs golden): {pure_moe_pcc}")
        logger.info(f"Pure MoE output: {moe_device_output_valid.flatten()[:8]}")

        device_passing, device_pcc = comp_pcc(
            decoder_moe_output_valid.flatten(), moe_device_output_valid.flatten(), 0.996
        )
        logger.info(f"Pure MoE vs Decoder MoE PCC: {device_pcc}")

        if use_hardcoded_expert_index:
            assert pure_moe_passing, f"Standalone MoE PCC check failed: {pure_moe_pcc}"
        assert device_passing, f"Pure MoE vs Decoder MoE PCC check failed: {device_pcc}"
    assert passing, f"DecoderBlock MoE PCC check failed: {pcc}"
    logger.info("✓ DecoderBlock mesh test passed!")
    ttnn.synchronize_device(submesh)


@pytest.mark.parametrize(
    "sender_row, sender_col",
    [
        (1, 0),
    ],
)
@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("use_fp32", [False])
@pytest.mark.parametrize("bcast_cluster_axis", [0])
@pytest.mark.parametrize("bcast_secondary_cluster_axis", [1])
@pytest.mark.parametrize("reduce_cluster_axis", [1])
@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2)])
@pytest.mark.parametrize("num_iters", [(1)])
@pytest.mark.parametrize("max_seq_len", [32 * 1024])
@pytest.mark.parametrize(
    "position_id",
    [
        0,
        127,
        pytest.param(511, marks=pytest.mark.skip_post_commit),
        pytest.param(1023, marks=pytest.mark.skip_post_commit),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
            "worker_l1_size": 1374544,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.parametrize("num_internal_iterations", [1])
@pytest.mark.requires_grid_size((13, 10))
def test_decoder_mlp(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    epsilon,
    use_fp32,
    bcast_cluster_axis,
    bcast_secondary_cluster_axis,
    reduce_cluster_axis,
    num_iters,
    max_seq_len,
    position_id,
    noc_mode,
    num_internal_iterations,
    get_reference_model_state_dict,
):
    """Test TTNN decoder fused operation for a dense (MLP) layer with enable_routing=False."""
    torch.manual_seed(0)
    num_devices = mesh_rows * mesh_cols
    logger.info(f"Number of devices: {num_devices}")
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than available")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()

    logger.info("Preparing dense MLP model state dict...")
    state_dict = get_reference_model_state_dict(
        layer_idx=DENSE_LAYER_IDX,
        is_moe=False,
        seed=RoutedExpert.SEED,
    )

    logger.info("Creating dense decoder block tensors...")
    d = create_decoder_block_tensors(
        submesh,
        mesh_rows,
        mesh_cols,
        sender_row,
        sender_col,
        position_id,
        state_dict,
        layer_idx=DENSE_LAYER_IDX,
        max_seq_len=max_seq_len,
        is_moe=False,
    )

    num_cores = device_grid_size.x * device_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, device_grid_size, row_wise=True)
    ttnn.synchronize_device(submesh)
    reduce_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(4)]
    persistent_next_iter_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 1)
    ttnn.synchronize_device(submesh)

    attn_semaphores = AttentionBlock.create_semaphores(submesh)
    moe_semaphores = MoeOp.create_semaphores(submesh)

    logger.info(f"Running dense decoder operation with position_id={position_id}...")
    for i in range(num_iters):
        moe_final_output_tensor, attention_block_output_tensor = DecoderBlock.op(
            # AttentionBlock parameters
            d["input_tensor_mesh"],
            d["gamma_overlapped"],
            d["matmul_weights_overlapped"],
            d["rmsnorm2_gamma_overlapped"],
            d["matmul2_weights_overlapped"],
            d["matmul3_weights_overlapped"],
            d["ttnn_qrope_sin"],
            d["ttnn_qrope_cos"],
            d["ttnn_trans_mat"],
            d["ttnn_krope_cos"],
            d["ttnn_krope_sin"],
            d["dkv_matmul_weights_overlapped"],
            d["dkv_rmsnorm_gamma_overlapped"],
            d["ttnn_kv_cache"],
            d["ttnn_position_ids"],
            d["scale"],
            d["sdpa_kv_cache_buffer"],
            d["sdpa_out_interm_buffer"],
            d["sender_coord"],
            # Post-SDPA parameters
            d["kv_b2_overlapped"],
            d["o_proj_overlapped"],
            d["ttnn_sdpa_input_l"],
            d["ttnn_sdpa_input_ms"],
            d["ttnn_sdpa_output_l"],
            d["ttnn_sdpa_intermediate_recv"],
            d["ttnn_sdpa_forwarder_scratch"],
            d["device_chunk_size"],
            d["ttnn_attention_block_output"],
            attention_block_semaphores=attn_semaphores,
            # MoE parameters (no gate_mm / routing tensors for dense MLP)
            shared_residual_mcast_src_tensor=d["ttnn_residual_mcast_src"],
            gate_mm_weights_tensor=None,
            gate_bias_tensor=None,
            gate_indices_tensor=None,
            gate_output_scores_tensor=None,
            gate_output_indices_tensor=None,
            gate_proj_weights_tensor=d["gate_proj_weights"],
            up_proj_weights_tensor=d["up_proj_weights"],
            down_proj_weights_tensor=d["down_proj_weights"],
            moe_final_output_tensor=None,
            rmsnorm_gamma_tensor=d["ffn_norm_overlapped"],
            shared_gate_weights_overlapped=d["shared_gate_weights_overlapped"],
            shared_up_weights_overlapped=d["shared_up_weights_overlapped"],
            shared_down_weights_tensor=d["shared_down_weights_tensor"],
            shared_k_parallel=d["shared_k_parallel"],
            shared_n_parallel=d["shared_n_parallel"],
            moe_semaphores=moe_semaphores,
            reduce_intermediate_tensors=d["reduce_intermediate_tensors"],
            reduce_output_tensor=d["reduce_output_tensor"],
            reduce_semaphores=reduce_semaphores,
            reduce_root_coord=ttnn.MeshCoordinate(d["reduce_root_coord"]),
            # Shared parameters
            enable_routing=False,
            bcast_cluster_axis=bcast_cluster_axis,
            bcast_secondary_cluster_axis=bcast_secondary_cluster_axis,
            reduce_cluster_axis=reduce_cluster_axis,
            sdpa_cluster_axis=0,
            num_links=1,
            epsilon=epsilon,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
            use_hardcoded_expert_index=False,
            noc_mode=noc_mode,
            num_iterations=num_internal_iterations,
            upstream_socket=None,
            downstream_socket=None,
            persistent_next_iter_semaphore=persistent_next_iter_semaphore,
            persistent_mode=True,
        )
    ttnn.synchronize_device(submesh)

    # ========================================================================
    # Extract decoder MLP output from reduce root
    # ========================================================================
    root_coord_tuple = d["reduce_root_coord"]
    root_device_idx = root_coord_tuple[0] * mesh_cols + root_coord_tuple[1]
    root_device_tensor = ttnn.get_device_tensors(moe_final_output_tensor)[root_device_idx]
    decoder_mlp_output = ttnn.to_torch(root_device_tensor)
    decoder_mlp_output_valid = extract_routed_expert_output(
        decoder_mlp_output.unsqueeze(0),
        d["num_gate_proj_cores"],
        RoutedExpert.FINAL_OUTPUT_WIDTH_PER_CORE,
        d["per_core_down_proj_N"],
    )

    # ========================================================================
    # Compute golden reference via DecoderBlock.golden (MLA → MLP full pipeline)
    # ========================================================================
    logger.info("Computing golden reference...")

    QNOPE_HEAD_DIM = 128
    QROPE_HEAD_DIM = 64
    KNOPE_DIM = 512
    KROPE_DIM = 64
    HEADS_PER_ROW = 8

    _full_q, _new_kv, _mla_output, _scores, _indices, moe_output = DecoderBlock.golden(
        d["golden_torch_input"],
        d["golden_torch_gamma"],
        d["golden_torch_matmul_weights"],
        d["golden_torch_rmsnorm2_gamma"],
        d["golden_torch_matmul2_weights"],
        d["golden_torch_matmul3_weights"],
        d["golden_torch_sin"],
        d["golden_torch_cos"],
        d["golden_position_ids"],
        d["golden_torch_dkv_matmul_weights"],
        d["golden_torch_dkv_rmsnorm_gamma"],
        d["golden_torch_kv_cache"],
        d["golden_scale"],
        d["golden_torch_kv_b2_proj_weights"],
        d["golden_torch_o_proj_weights"],
        epsilon=epsilon,
        num_qnope_heads=d["golden_total_qnope_heads"],
        num_qrope_heads=d["golden_total_qrope_heads"],
        qnope_head_dim=QNOPE_HEAD_DIM,
        qrope_head_dim=QROPE_HEAD_DIM,
        heads_per_row=HEADS_PER_ROW,
        nope_dim=KNOPE_DIM,
        rope_dim=KROPE_DIM,
        moe_shared_gate_weights=d["golden_moe_shared_gate"],
        moe_shared_up_weights=d["golden_moe_shared_up"],
        moe_shared_down_weights=d["golden_moe_shared_down"],
        moe_gate_proj_weights_dict=d["golden_moe_gate_proj_dict"],
        moe_up_proj_weights_dict=d["golden_moe_up_proj_dict"],
        moe_down_proj_weights_dict=d["golden_moe_down_proj_dict"],
        moe_rmsnorm_gamma=d["golden_moe_rmsnorm_gamma"],
        moe_rmsnorm_epsilon=epsilon,
        moe_enable_routing=False,
    )

    # ========================================================================
    # Validate MLP output vs DecoderBlock golden
    # ========================================================================
    passing, pcc = comp_pcc(moe_output.flatten(), decoder_mlp_output_valid.float().flatten(), 0.975)
    logger.info(f"MLP PCC (decoder vs golden): {pcc}")
    logger.info(f"Golden MLP output: {moe_output.flatten()[:8]}")
    logger.info(f"DecoderBlock MLP output: {decoder_mlp_output_valid.flatten()[:8]}")
    assert passing, f"DecoderBlock MLP Output PCC check failed: {pcc}"

    logger.info("✓ DecoderBlock MLP mesh test passed!")

    ttnn.synchronize_device(submesh)
