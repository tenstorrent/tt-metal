# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Decoder Block compile test: verifies that the fused AttentionBlock + MoE kernel
compiles without errors when all compile-time args from both ops are merged.

The kernel has `#ifdef COMPILE_TEST_ONLY return;` at the top of kernel_main,
so it compiles but does no work. This validates:
  - struct Core role flags resolve for all cores
  - Python-side program descriptor merging produces a valid program
  - No naming conflicts in compile-time args

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_decoder_block.py -v -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights, OverlappedTensor
from models.demos.deepseek_v3_b1.fused_ops.attention_block.op import AttentionBlock
from models.demos.deepseek_v3_b1.fused_ops.decoder_block.op import DecoderBlock
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode
from models.demos.deepseek_v3_b1.prepare_weights import create_gate_indices_tensor, prepare_moe_layer_weights
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import (
    ROUTED_EXPERT_LAYER_IDX,
    RoutedExpert,
    SharedExpert,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.test_post_sdpa import compute_forwarder_scratch_size


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


# ============================================================================
# Unified decoder block tensor setup (single BDW instance for all L1 weights)
# ============================================================================
def create_decoder_block_tensors(
    submesh, mesh_rows, mesh_cols, sender_row, sender_col, position_id, state_dict, layer_idx, num_routed_experts
):
    """Create all tensors required by DecoderBlock.op() using a single BlitzDecodeWeights.

    This avoids the L1 OOM caused by allocating overlapped attention weights twice
    (once for MLA, once for MoE's gate_mm/ffn_norm). A single BDW instance ensures
    the three fused weight groups share L1 without duplication.

    Returns a dict with all attention + MoE + shared expert + reduce tensors.
    """
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
    max_seq_len = 32 * 1024
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

    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # ══════════════════════════════════════════════════════════════════════════
    # All weights via prepare_moe_layer_weights (single BDW)
    # ══════════════════════════════════════════════════════════════════════════
    bdw = BlitzDecodeWeights(submesh)
    layer = prepare_moe_layer_weights(
        bdw,
        state_dict,
        layer_idx,
        num_routed_experts=num_routed_experts,
        move_to_device=True,
    )

    # ── MoE final output config (DRAM streaming matmul output grid) ──
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = submesh.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
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

    # ── MoE input core (same as gather core (12,9) — attention output IS the MoE input) ──
    input_core = ttnn.CoreCoord(device_grid_size.x - 1, RoutedExpert.INPUT_CORE_Y)
    input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(input_core, input_core)])

    # ── Gate indices (via prepare_weights utility) ──
    ttnn_gate_indices = create_gate_indices_tensor(submesh, input_core_grid, mesh_mapper=mesh_mapper)

    # ── Gate output buffers ──
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

    # ══════════════════════════════════════════════════════════════════════════
    # Attention input/intermediate/output mesh tensors
    # ══════════════════════════════════════════════════════════════════════════
    sender_coord = ttnn.MeshCoordinate(sender_row, sender_col)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}), shape, ttnn.ShardOrientation.ROW_MAJOR
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    device_tensors = []
    intermediate_tensors_list = []
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            if row == sender_row and col == sender_col:
                device_tensors.append(torch_input)
            else:
                device_tensors.append(torch.zeros_like(torch_input))
            intermediate_tensors_list.append(torch.zeros_like(torch_input))

    input_tensor_mesh = ttnn.from_torch(
        torch.cat(device_tensors, dim=0),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        torch.cat(intermediate_tensors_list, dim=0),
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
        dtype=ttnn.bfloat16,
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
    torch_kv_cache = torch.full((1, 1, max_seq_len, kvpe_dim), float("-inf"), dtype=torch.bfloat16)
    for i in range(position_id):
        torch_kv_cache[:, :, i, :] = torch.randn(1, 1, 1, kvpe_dim, dtype=torch.bfloat16)
    ttnn_kv_cache = ttnn.from_torch(
        torch_kv_cache,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=kv_mem,
        mesh_mapper=mesh_mapper,
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
    attn_output_overlapped = OverlappedTensor(
        fused_tensor=sdpa_kv_cache_buffer,
        tensor_shape=(M, output_size),
        shard_shape=(M, output_size),
        core_range_set=gather_core_grid,
        dtype=ttnn.bfloat16,
        tile_shape=(1, 32),
        byte_offset=0,
        total_size=num_output_tiles * a_tile_size,
    )

    # ── SDPA worker/forwarder tensors ──
    sdpa_output_cores = FlashMLADecode.ProgramConfig.grid.output_cores(0, NUM_SDPA_WORKERS)
    sdpa_worker_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sdpa_output_cores]
    )
    sdpa_l_per_worker = SDPA_L_WIDTH // NUM_SDPA_WORKERS
    sdpa_ms_per_worker = SDPA_MS_WIDTH // NUM_SDPA_WORKERS

    sdpa_l_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sdpa_worker_grid, (SDPA_L_HEIGHT, sdpa_l_per_worker), ttnn.ShardOrientation.ROW_MAJOR),
    )
    sdpa_ms_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sdpa_worker_grid, (SDPA_L_HEIGHT, sdpa_ms_per_worker), ttnn.ShardOrientation.ROW_MAJOR),
    )

    mesh_sdpa_l = torch.cat([torch.randn((SDPA_L_HEIGHT, SDPA_L_WIDTH), dtype=torch.bfloat16)] * num_devices, dim=0)
    mesh_sdpa_ms = torch.cat([torch.randn((SDPA_L_HEIGHT, SDPA_MS_WIDTH), dtype=torch.bfloat16)] * num_devices, dim=0)

    ttnn_sdpa_input_l = ttnn.from_torch(
        mesh_sdpa_l,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_l_mem,
        tile=sdpa_tile,
        mesh_mapper=shard_mesh_mapper,
    )
    ttnn_sdpa_input_ms = ttnn.from_torch(
        mesh_sdpa_ms,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_ms_mem,
        tile=sdpa_tile,
        mesh_mapper=shard_mesh_mapper,
    )
    ttnn_sdpa_output_l = ttnn.from_torch(
        torch.zeros_like(mesh_sdpa_l),
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_l_mem,
        tile=sdpa_tile,
        mesh_mapper=shard_mesh_mapper,
    )

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
    root_coord = (1, 1)
    reduce_mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh.shape)
    reduce_mesh_mapper = ttnn.create_mesh_mapper(submesh, reduce_mesh_mapper_config)
    tile_1x32 = ttnn.Tile([1, 32])

    reduce_intermediate_tensors = []
    for _ in range(3):
        it = ttnn.from_torch(
            torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=final_output_mem_config,
            tile=tile_1x32,
            mesh_mapper=reduce_mesh_mapper,
        )
        reduce_intermediate_tensors.append(it)

    compute_grid = submesh.compute_with_storage_grid_size()
    reduce_output_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
    reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(reduce_output_core, reduce_output_core)})
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

    # ── Shared expert mcast grid ──
    def _unwrap(t):
        return t.fused_tensor if isinstance(t, OverlappedTensor) else t

    sender_core_from_residual = _unwrap(attn_output_overlapped).memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core_from_residual)])

    return {
        # Attention weights (from prepare_moe_layer_weights via BDW)
        "gamma_overlapped": layer.attn_norm,
        "matmul_weights_overlapped": layer.q_a_proj,
        "rmsnorm2_gamma_overlapped": layer.q_norm,
        "matmul2_weights_overlapped": layer.q_b_proj,
        "matmul3_weights_overlapped": layer.kv_b1_proj,
        "dkv_matmul_weights_overlapped": layer.kv_a_proj,
        "dkv_rmsnorm_gamma_overlapped": layer.kv_norm,
        "kv_b2_overlapped": layer.kv_b2_proj,
        "o_proj_overlapped": layer.o_proj,
        "gate_mm_overlapped": layer.gate_mm,
        "ffn_norm_overlapped": layer.ffn_norm,
        # Attention activation/buffer tensors
        "input_tensor_mesh": input_tensor_mesh,
        "intermediate_tensor_mesh": intermediate_tensor_mesh,
        "ttnn_qrope_sin": ttnn_qrope_sin,
        "ttnn_qrope_cos": ttnn_qrope_cos,
        "ttnn_trans_mat": ttnn_trans_mat,
        "ttnn_krope_cos": ttnn_krope_cos,
        "ttnn_krope_sin": ttnn_krope_sin,
        "ttnn_kv_cache": ttnn_kv_cache,
        "position_id": position_id,
        "ttnn_position_ids": ttnn_position_ids,
        "scale": scale,
        "sdpa_kv_cache_buffer": sdpa_kv_cache_buffer,
        "sdpa_out_interm_buffer": sdpa_out_interm_buffer,
        "sender_coord": sender_coord,
        "ttnn_sdpa_input_l": ttnn_sdpa_input_l,
        "ttnn_sdpa_input_ms": ttnn_sdpa_input_ms,
        "ttnn_sdpa_output_l": ttnn_sdpa_output_l,
        "ttnn_sdpa_intermediate_recv": ttnn_sdpa_intermediate_recv,
        "ttnn_sdpa_forwarder_scratch": ttnn_sdpa_forwarder_scratch,
        "ttnn_attention_block_output": attn_output_overlapped,
        # MoE tensors (attention_block_output IS the MoE residual input — overlapped with kv cache)
        "ttnn_residual_mcast_src": attn_output_overlapped,
        "ttnn_gate_bias": layer.gate_bias,
        "ttnn_gate_indices": ttnn_gate_indices,
        "gate_output_scores_tensor": gate_output_scores_tensor,
        "gate_output_indices_tensor": gate_output_indices_tensor,
        "gate_proj_weights": layer.routed_gate_proj[0],
        "up_proj_weights": layer.routed_up_proj[0],
        "down_proj_weights": layer.routed_down_proj[0],
        "final_output_mem_config": final_output_mem_config,
        "final_output_total_width": final_output_total_width,
        # Shared expert weights
        "shared_gate_weights_overlapped": layer.shared_gate_proj,
        "shared_up_weights_overlapped": layer.shared_up_proj,
        "shared_down_weights_tensor": layer.shared_down_proj,
        "shared_k_parallel": SharedExpert.K_PARALLEL,
        "shared_n_parallel": SharedExpert.N_PARALLEL,
        # Reduce-to-one
        "reduce_intermediate_tensors": reduce_intermediate_tensors,
        "reduce_output_tensor": reduce_output_tensor,
        "reduce_root_coord": root_coord,
        # MoE mcast grid (for shared expert)
        "mcast_grid": mcast_grid,
    }


# ============================================================================
# Test: DecoderBlock compile-only
# ============================================================================
@pytest.mark.parametrize("sender_row, sender_col", [(1, 0)])
@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2)])
@pytest.mark.parametrize("position_id", [127])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
            "worker_l1_size": 1433168,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
def test_decoder_block_compile(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    position_id,
    noc_mode,
    get_reference_model_state_dict,
):
    """Compile-only test: verifies fused kernel compiles with merged args."""
    num_devices = mesh_rows * mesh_cols
    logger.info(f"Number of devices: {num_devices}")
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than available")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    if device_grid_size.x < 12:
        pytest.skip(f"Device grid too small: need 12 columns, have {device_grid_size.x}")

    num_routed_experts = 1
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
        num_routed_experts=num_routed_experts,
    )

    num_cores = device_grid_size.x * device_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, device_grid_size, row_wise=True)
    ttnn.synchronize_device(submesh)
    reduce_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(4)]
    ttnn.synchronize_device(submesh)

    attn_semaphores = AttentionBlock.create_semaphores(submesh)
    moe_semaphores = MoeOp.create_semaphores(submesh)

    logger.info("Calling DecoderBlock.op() with COMPILE_TEST_ONLY...")
    result = DecoderBlock.op(
        # AttentionBlock parameters
        d["input_tensor_mesh"],
        d["intermediate_tensor_mesh"],
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
        d["position_id"],
        d["ttnn_position_ids"],
        d["scale"],
        d["sdpa_kv_cache_buffer"],
        d["sdpa_out_interm_buffer"],
        d["sender_coord"],
        # Post-SDPA
        d["kv_b2_overlapped"],
        d["o_proj_overlapped"],
        d["ttnn_sdpa_input_l"],
        d["ttnn_sdpa_input_ms"],
        d["ttnn_sdpa_output_l"],
        d["ttnn_sdpa_intermediate_recv"],
        d["ttnn_sdpa_forwarder_scratch"],
        0,  # sdpa_per_device_chunk_size
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
        reduce_root_coord=ttnn.MeshCoordinate(d["reduce_root_coord"]),
        # Shared parameters
        enable_routing=True,
        extra_defines={"COMPILE_TEST_ONLY": "1"},
    )

    ttnn.synchronize_device(submesh)
    logger.info("DecoderBlock compile test PASSED - kernel compiled and ran (no-op) successfully")
    assert result is not None, "DecoderBlock.op() returned None"
