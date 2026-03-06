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
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.fused_ops.attention_block.op import AttentionBlock
from models.demos.deepseek_v3_b1.fused_ops.decoder_block.op import DecoderBlock
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import (
    create_routed_expert_tensors,
    create_shared_expert_tensors,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.test_post_sdpa import compute_forwarder_scratch_size


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


# ============================================================================
# Attention Block tensor setup (adapted from test_attention_block.py)
# ============================================================================
def create_attention_block_tensors(submesh, mesh_rows, mesh_cols, sender_row, sender_col, position_id):
    """Create all tensors required by AttentionBlock.get_program_context().

    Returns a dict containing all ttnn tensors and scalars needed to call
    DecoderBlock.op() for the attention portion.
    """
    num_devices = mesh_rows * mesh_cols
    num_tp = mesh_cols
    skip_ccl = False

    device_grid_size = submesh.compute_with_storage_grid_size()

    # Head configuration
    NUM_QNOPE_HEADS = 64
    NUM_QROPE_HEADS = 64
    QNOPE_HEAD_DIM = 128
    QROPE_HEAD_DIM = 64
    QNOPE_OUT_DIM = 512
    KNOPE_DIM = 512
    KROPE_DIM = 64

    M = 1
    K1 = QNOPE_OUT_DIM
    post_sdpa_intermediate = 8192
    K2 = 8192
    output_size = 7168
    shape = (1, 7168)
    matmul_weights_shape = (7168, 1536)
    max_seq_len = 32 * 1024
    scale = (QNOPE_HEAD_DIM + QROPE_HEAD_DIM) ** -0.5
    rmsnorm2_width = 1536

    QNOPE_GRID_COLS = 8
    QROPE_GRID_COLS = 4
    matmul2_grid_y = 8
    qnope_num_cores = QNOPE_GRID_COLS * matmul2_grid_y
    qrope_num_cores = QROPE_GRID_COLS * matmul2_grid_y
    kvpe_dim = KNOPE_DIM + KROPE_DIM

    NUM_SDPA_WORKERS = 8
    SDPA_L_HEIGHT = 8
    SDPA_L_WIDTH = 512 * NUM_SDPA_WORKERS
    SDPA_MS_WIDTH = 32 * NUM_SDPA_WORKERS

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    tile = ttnn.Tile([1, 32])

    # KV cache branch grid
    kv_cache_branch_start_offset = (0, 8)
    kv_cache_branch_rope_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(8 + kv_cache_branch_start_offset[0], kv_cache_branch_start_offset[1]),
                ttnn.CoreCoord(8 + kv_cache_branch_start_offset[0], 1 + kv_cache_branch_start_offset[1]),
            )
        }
    )

    # ── SDPA KV cache buffer (overlaps with pre-SDPA CBs) ──
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
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # ── SDPA output intermediate buffer ──
    sdpa_out_interm_num_cores = device_grid_size.x * device_grid_size.y
    sdpa_out_interm_shard_height = 40
    sdpa_out_interm_shard_width = 544
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
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        tile=ttnn.Tile([8, 32]),
    )

    # ── Torch tensors ──
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape, dtype=torch.bfloat16)
    torch_matmul_weights = torch.randn(matmul_weights_shape, dtype=torch.bfloat16)
    torch_rmsnorm2_gamma = torch.randn((1, rmsnorm2_width), dtype=torch.bfloat16)

    total_qnope_dim = num_tp * NUM_QNOPE_HEADS * QNOPE_HEAD_DIM
    total_qrope_dim = num_tp * NUM_QROPE_HEADS * QROPE_HEAD_DIM
    torch_matmul2_weights_full = torch.randn(
        (matmul_weights_shape[1], total_qnope_dim + total_qrope_dim), dtype=torch.bfloat16
    )
    torch_matmul3_weights = torch.randn((num_tp * NUM_QNOPE_HEADS, QNOPE_HEAD_DIM, QNOPE_OUT_DIM), dtype=torch.bfloat16)
    torch_dkv_matmul_weights = torch.randn((7168, KNOPE_DIM + KROPE_DIM), dtype=torch.bfloat16)
    torch_gate_mm_weights = torch.zeros((7168, 256), dtype=torch.bfloat16)
    torch_ffn_norm = torch.zeros((1, 7168), dtype=torch.bfloat16)
    torch_dkv_rmsnorm_gamma = torch.randn((1, KNOPE_DIM), dtype=torch.bfloat16)

    torch_weights1 = torch.randn((K1, post_sdpa_intermediate), dtype=torch.bfloat16)
    torch_weights2 = torch.randn((K2, output_size), dtype=torch.bfloat16)
    torch_kv_b2_proj_weights = torch.cat([torch_weights1] * num_tp, dim=1) if num_tp > 1 else torch_weights1
    torch_o_proj_weights = torch.cat([torch_weights2] * num_tp, dim=0) if num_tp > 1 else torch_weights2

    # ── RoPE tensors ──
    position_ids = torch.tensor([position_id])
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, QROPE_HEAD_DIM, 2, dtype=torch.float32) / QROPE_HEAD_DIM))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    torch_cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)
    torch_sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)
    torch_trans_mat = get_rot_transformation_mat()

    # ── BlitzDecodeWeights overlapped tensors ──
    bdw = BlitzDecodeWeights(submesh)
    (
        matmul_weights_overlapped,
        matmul2_weights_overlapped,
        dkv_matmul_weights_overlapped,
    ) = bdw.get_tt_q_ab_proj_and_kv_a_proj_weights(
        torch_matmul_weights, torch_matmul2_weights_full, torch_dkv_matmul_weights
    )

    torch_matmul3_weights_flat = torch_matmul3_weights.reshape(num_tp * NUM_QNOPE_HEADS * QNOPE_HEAD_DIM, QNOPE_OUT_DIM)
    matmul3_weights_overlapped, kv_b2_overlapped = bdw.get_tt_kv_b12_proj_weights(
        torch_matmul3_weights_flat, torch_kv_b2_proj_weights
    )

    (
        o_proj_overlapped,
        gate_mm_overlapped,
        gamma_overlapped,
        rmsnorm2_gamma_overlapped,
        dkv_rmsnorm_gamma_overlapped,
        _,
    ) = bdw.get_tt_o_proj_and_gate_mm_weights(
        torch_o_proj_weights,
        torch_gate_mm_weights,
        torch_gamma,
        torch_rmsnorm2_gamma,
        torch_dkv_rmsnorm_gamma,
        torch_ffn_norm,
    )

    # ── Input/intermediate mesh tensors ──
    sender_coord = ttnn.MeshCoordinate(sender_row, sender_col)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}), shape, ttnn.ShardOrientation.ROW_MAJOR
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    device_tensors = []
    intermediate_tensors = []
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            if row == sender_row and col == sender_col:
                device_tensors.append(torch_input)
            else:
                device_tensors.append(torch.zeros_like(torch_input))
            intermediate_tensors.append(torch.zeros_like(torch_input))

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
        torch.cat(intermediate_tensors, dim=0),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )

    # ── RoPE TTNN tensors ──
    qrope_dram_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    ttnn_qrope_cos = ttnn.from_torch(
        torch_cos.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_dram_mem,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_qrope_sin = ttnn.from_torch(
        torch_sin.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_dram_mem,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_krope_cos = ttnn.from_torch(
        torch_cos.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_dram_mem,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_krope_sin = ttnn.from_torch(
        torch_sin.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=qrope_dram_mem,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
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
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
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
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
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
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
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
    ttnn_output = ttnn.from_torch(
        torch.zeros((SDPA_INPUT_NUM_CORES * HEADS_PER_ROW, QNOPE_OUT_DIM), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=sdpa_mem,
        tile=sdpa_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # ── Post-SDPA tensors ──
    a_tile = ttnn.Tile([M, 32])
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    gather_core = ttnn.CoreCoord(12, 9)
    gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

    ttnn_gather1_output = ttnn.from_torch(
        torch.zeros((M, post_sdpa_intermediate), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn.get_device_tensors(input_tensor_mesh)[0].device(),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(gather_core_grid, (M, post_sdpa_intermediate), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        tile=a_tile,
    )

    gather2_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(gather_core_grid, (M, output_size), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_gather2_output = ttnn.from_torch(
        torch.cat([torch.zeros((M, output_size), dtype=torch.bfloat16)] * num_devices, dim=0),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=gather2_mem,
        mesh_mapper=mesh_mapper,
    )

    ccl_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(gather_core_grid, (M, output_size), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_ccl_intermediate = ttnn.from_torch(
        torch.cat([torch.zeros((M, output_size), dtype=torch.bfloat16)] * num_devices, dim=0),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=ccl_mem,
        mesh_mapper=mesh_mapper,
    )

    output_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(gather_core_grid, (M, output_size), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_attention_block_output = ttnn.from_torch(
        torch.cat([torch.zeros((M, output_size), dtype=torch.bfloat16)] * num_devices, dim=0),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=output_mem,
        mesh_mapper=mesh_mapper,
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
        mesh_mapper=mesh_mapper,
    )
    ttnn_sdpa_input_ms = ttnn.from_torch(
        mesh_sdpa_ms,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_ms_mem,
        tile=sdpa_tile,
        mesh_mapper=mesh_mapper,
    )
    ttnn_sdpa_output_l = ttnn.from_torch(
        torch.zeros_like(mesh_sdpa_l),
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_l_mem,
        tile=sdpa_tile,
        mesh_mapper=mesh_mapper,
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
        mesh_mapper=mesh_mapper,
    )

    # SDPA forwarder scratch buffer
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
        mesh_mapper=mesh_mapper,
    )

    return {
        "input_tensor_mesh": input_tensor_mesh,
        "intermediate_tensor_mesh": intermediate_tensor_mesh,
        "gamma_overlapped": gamma_overlapped,
        "matmul_weights_overlapped": matmul_weights_overlapped,
        "rmsnorm2_gamma_overlapped": rmsnorm2_gamma_overlapped,
        "matmul2_weights_overlapped": matmul2_weights_overlapped,
        "matmul3_weights_overlapped": matmul3_weights_overlapped,
        "ttnn_qrope_sin": ttnn_qrope_sin,
        "ttnn_qrope_cos": ttnn_qrope_cos,
        "ttnn_trans_mat": ttnn_trans_mat,
        "ttnn_krope_cos": ttnn_krope_cos,
        "ttnn_krope_sin": ttnn_krope_sin,
        "dkv_matmul_weights_overlapped": dkv_matmul_weights_overlapped,
        "dkv_rmsnorm_gamma_overlapped": dkv_rmsnorm_gamma_overlapped,
        "ttnn_kv_cache": ttnn_kv_cache,
        "position_id": position_id,
        "ttnn_position_ids": ttnn_position_ids,
        "scale": scale,
        "ttnn_output": ttnn_output,
        "sdpa_kv_cache_buffer": sdpa_kv_cache_buffer,
        "sdpa_out_interm_buffer": sdpa_out_interm_buffer,
        "sender_coord": sender_coord,
        "kv_b2_overlapped": kv_b2_overlapped,
        "o_proj_overlapped": o_proj_overlapped,
        "ttnn_gather1_output": ttnn_gather1_output,
        "ttnn_gather2_output": ttnn_gather2_output,
        "ttnn_ccl_intermediate": ttnn_ccl_intermediate,
        "ttnn_sdpa_input_l": ttnn_sdpa_input_l,
        "ttnn_sdpa_input_ms": ttnn_sdpa_input_ms,
        "ttnn_sdpa_output_l": ttnn_sdpa_output_l,
        "ttnn_sdpa_intermediate_recv": ttnn_sdpa_intermediate_recv,
        "ttnn_sdpa_forwarder_scratch": ttnn_sdpa_forwarder_scratch,
        "ttnn_attention_block_output": ttnn_attention_block_output,
        "gate_mm_overlapped": gate_mm_overlapped,
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
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
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
):
    """Compile-only test: verifies fused kernel compiles with merged args."""
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than available")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    if device_grid_size.x < 12:
        pytest.skip(f"Device grid too small: need 12 columns, have {device_grid_size.x}")

    logger.info("Creating attention block tensors...")
    attn = create_attention_block_tensors(submesh, mesh_rows, mesh_cols, sender_row, sender_col, position_id)

    logger.info("Creating MoE tensors...")
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)
    r = create_routed_expert_tensors(
        submesh,
        use_hardcoded_expert_index=False,
        mesh_mapper=mesh_mapper,
        create_final_output=False,
        enable_routing=False,
    )
    M, K = 1, 7168

    # Create routing-specific small tensors manually (gate_mm comes from Group 2 overlap above)
    device_grid_size = submesh.compute_with_storage_grid_size()
    input_core = ttnn.CoreCoord(device_grid_size.x - 1, 9)
    input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(input_core, input_core)])
    tile_16x16 = ttnn.Tile([16, 16])
    tile_1x16 = ttnn.Tile((1, 16))

    gate_input_shard_spec = ttnn.ShardSpec(input_core_grid, (16, 16), ttnn.ShardOrientation.ROW_MAJOR)
    gate_input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gate_input_shard_spec
    )
    torch_bias = torch.randn((1, 8, 32), dtype=torch.bfloat16)
    torch_bias_transposed = torch.transpose(torch_bias.reshape(16, 16), 0, 1).contiguous()
    ttnn_gate_bias = ttnn.from_torch(
        torch_bias_transposed,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
        mesh_mapper=mesh_mapper,
    )
    torch_indices = torch.arange(256, dtype=torch.int32).reshape(16, 16).T.contiguous().to(torch.uint16)
    ttnn_gate_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
        mesh_mapper=mesh_mapper,
    )
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

    sender_core = r["ttnn_residual_mcast_src"].memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(submesh, M, K, mcast_grid, mesh_mapper=mesh_mapper)

    logger.info("Creating reduce tensors...")
    root_coord = (1, 1)
    reduce_mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh.shape)
    reduce_mesh_mapper = ttnn.create_mesh_mapper(submesh, reduce_mesh_mapper_config)

    tile_1x32 = ttnn.Tile([1, 32])
    final_output_total_width = r["final_output_total_width"]
    final_output_mem_config = r["final_output_mem_config"]

    intermediate_tensors = []
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
        intermediate_tensors.append(it)

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

    num_cores = compute_grid.x * compute_grid.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
    ttnn.synchronize_device(submesh)
    reduce_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(4)]
    ttnn.synchronize_device(submesh)

    attn_semaphores = AttentionBlock.create_semaphores(submesh)
    moe_semaphores = MoeOp.create_semaphores(submesh)

    logger.info("Calling DecoderBlock.op() with COMPILE_TEST_ONLY...")
    result = DecoderBlock.op(
        # AttentionBlock parameters
        attn["input_tensor_mesh"],
        attn["intermediate_tensor_mesh"],
        attn["gamma_overlapped"],
        attn["matmul_weights_overlapped"],
        attn["rmsnorm2_gamma_overlapped"],
        attn["matmul2_weights_overlapped"],
        attn["matmul3_weights_overlapped"],
        attn["ttnn_qrope_sin"],
        attn["ttnn_qrope_cos"],
        attn["ttnn_trans_mat"],
        attn["ttnn_krope_cos"],
        attn["ttnn_krope_sin"],
        attn["dkv_matmul_weights_overlapped"],
        attn["dkv_rmsnorm_gamma_overlapped"],
        attn["ttnn_kv_cache"],
        attn["position_id"],
        attn["ttnn_position_ids"],
        attn["scale"],
        attn["ttnn_output"],
        attn["sdpa_kv_cache_buffer"],
        attn["sdpa_out_interm_buffer"],
        attn["sender_coord"],
        # Post-SDPA
        attn["kv_b2_overlapped"],
        attn["o_proj_overlapped"],
        attn["ttnn_gather1_output"],
        attn["ttnn_gather2_output"],
        attn["ttnn_ccl_intermediate"],
        attn["ttnn_sdpa_input_l"],
        attn["ttnn_sdpa_input_ms"],
        attn["ttnn_sdpa_output_l"],
        attn["ttnn_sdpa_intermediate_recv"],
        attn["ttnn_sdpa_forwarder_scratch"],
        0,  # sdpa_per_device_chunk_size
        attn["ttnn_attention_block_output"],
        attention_block_semaphores=attn_semaphores,
        # MoE parameters
        shared_residual_mcast_src_tensor=r["ttnn_residual_mcast_src"],
        gate_mm_weights_tensor=attn["gate_mm_overlapped"],
        gate_bias_tensor=ttnn_gate_bias,
        gate_indices_tensor=ttnn_gate_indices,
        gate_output_scores_tensor=gate_output_scores_tensor,
        gate_output_indices_tensor=gate_output_indices_tensor,
        gate_proj_weights_tensor=r["gate_proj_weights"],
        up_proj_weights_tensor=r["up_proj_weights"],
        down_proj_weights_tensor=r["down_proj_weights"],
        moe_final_output_tensor=r["final_output_tensor"],
        rmsnorm_gamma_tensor=r["ttnn_rmsnorm_gamma"],
        shared_gate_weights_overlapped=s["shared_gate_weights_overlapped"],
        shared_up_weights_overlapped=s["shared_up_weights_overlapped"],
        shared_down_weights_tensor=s["ttnn_down_weights"],
        shared_k_parallel=s["k_parallel"],
        shared_n_parallel=s["n_parallel"],
        moe_semaphores=moe_semaphores,
        reduce_intermediate_tensors=intermediate_tensors,
        reduce_output_tensor=reduce_output_tensor,
        reduce_semaphores=reduce_semaphores,
        reduce_root_coord=ttnn.MeshCoordinate(root_coord),
        # Shared parameters
        enable_routing=True,
        extra_defines={"COMPILE_TEST_ONLY": "1"},
    )

    ttnn.synchronize_device(submesh)
    logger.info("DecoderBlock compile test PASSED - kernel compiled and ran (no-op) successfully")
    assert result is not None, "DecoderBlock.op() returned None"
