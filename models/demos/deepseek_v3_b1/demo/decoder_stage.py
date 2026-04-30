# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decoder and dense block pipeline stages (socket-fed bcast + fused DecoderBlock + reduce-to-one)."""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import yarn_get_mscale
from models.demos.deepseek_v3.tt.rope import get_cos_sin_matrix, get_rot_transformation_mat
from models.demos.deepseek_v3_b1.demo.stage import (
    ACTIVATION_PAGE_SIZE_BYTES,
    ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES,
    DEFAULT_ACTIVATION_FIFO_PAGES,
    PIPELINE_CORE_COORD,
    StageContext,
    StageKind,
    activation_fifo_size_bytes,
)
from models.demos.deepseek_v3_b1.fused_ops.attention_block.op import AttentionBlock
from models.demos.deepseek_v3_b1.fused_ops.decoder_block.op import DecoderBlock
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.metadata.metadata import DeepseekMetadata, create_metadata_tensor
from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.persistent_loop.op import PersistentLoop
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import HostIoPlacement, LoopbackConfig, PipelineBlock
from models.demos.deepseek_v3_b1.micro_ops.sdpa_reduce_to_all.op import compute_forwarder_scratch_size
from models.demos.deepseek_v3_b1.model_dimensions import RoutedExpert, SharedExpert
from models.demos.deepseek_v3_b1.utils import (
    deinterleave_kv_cache,
    get_pinned_optimal_dram_bank_to_logical_worker_assignment,
)
from models.demos.deepseek_v3_b1.weights.prepare import (
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3MoELayerWeights,
    create_gate_indices_tensor,
)


def create_decoder_block_tensors(
    submesh,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    position_id,
    max_seq_len,
    reduce_root_coord=ttnn.MeshCoordinate(1, 1),
    *,
    weights: DeepSeekV3MoELayerWeights | DeepSeekV3DenseLayerWeights,
    metadata: DeepseekMetadata | None = None,
    num_slots: int = 64,
    is_moe: bool = True,
    validate_debug_tensors: bool = False,
    torch_input=None,
    forward_metadata=False,
):
    """Create all tensors required by DecoderBlock.op().

    ``weights`` must be built on ``submesh`` (e.g. via
    ``prepare_moe_layer_weights`` / ``prepare_dense_layer_weights`` from
    ``weights/prepare.py``).

    Returns a dict with all attention + FFN + shared expert + reduce tensors.
    Intermediate torch CPU tensors (torch_input, torch_kv_cache, etc.) are
    included so that callers (e.g. golden-reference builders) can reuse them.
    """
    if is_moe and not isinstance(weights, DeepSeekV3MoELayerWeights):
        raise TypeError(f"is_moe=True requires DeepSeekV3MoELayerWeights, got {type(weights).__name__}")
    if not is_moe and not isinstance(weights, DeepSeekV3DenseLayerWeights):
        raise TypeError(f"is_moe=False requires DeepSeekV3DenseLayerWeights, got {type(weights).__name__}")
    torch.manual_seed(0)
    if metadata is None:
        metadata = DeepseekMetadata(position_id=position_id)
    position_id = metadata.position_id
    num_devices = mesh_rows * mesh_cols
    device_grid_size = submesh.compute_with_storage_grid_size()
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)

    # TODO: Shouldn't hardcode this here
    class _RopeConfig:
        qk_rope_head_dim = 64
        rope_theta = 10000.0
        rope_scaling = {
            "factor": 40,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }

    _RopeConfig.max_seq_len = max_seq_len

    # Constants for runtime tensors
    QNOPE_HEAD_DIM = 128
    QROPE_HEAD_DIM = _RopeConfig.qk_rope_head_dim
    QNOPE_OUT_DIM = 512
    KNOPE_DIM = 512
    KROPE_DIM = 64

    M = 1
    K = 7168
    output_size = 7168
    shape = (1, K)
    q_head_dim = QNOPE_HEAD_DIM + QROPE_HEAD_DIM
    mscale = yarn_get_mscale(
        _RopeConfig.rope_scaling["factor"],
        _RopeConfig.rope_scaling["mscale_all_dim"],
    )
    scale = q_head_dim**-0.5 * mscale * mscale
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

    # SDPA KV cache buffer
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

    # SDPA output intermediate buffer
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

    if torch_input is None:
        torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # FFN final output config (DRAM streaming matmul output grid)
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

    # MoE-only gate indices and output buffers
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
        moe_ref_gate_output_scores = None
        moe_ref_gate_output_indices = None
        if validate_debug_tensors:
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

    if forward_metadata:
        padding = DeepseekMetadata.aligned_size_bytes() // dtype_size(ttnn.bfloat16)
        padded_shape = (1, K + padding)
        padded_input = torch.nn.functional.pad(torch_input, (0, padding), value=0)
    else:
        padded_shape = shape
        padded_input = torch_input
    # Attention input/intermediate/output mesh tensors
    sender_coord = ttnn.MeshCoordinate(sender_row, sender_col)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}), padded_shape, ttnn.ShardOrientation.ROW_MAJOR
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    device_tensors = []
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            if row == sender_row and col == sender_col:
                device_tensors.append(padded_input)
            else:
                device_tensors.append(torch.zeros_like(padded_input))

    input_tensor_mesh = ttnn.from_torch(
        torch.cat(device_tensors, dim=0),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )

    # RoPE TTNN tensors
    qrope_dram_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    position_ids = torch.tensor([position_id])

    cos_sin_4d, sin_sin_4d = get_cos_sin_matrix(_RopeConfig)
    torch_cos = cos_sin_4d.squeeze(0).squeeze(0)  # [max_seq_len, dim]
    torch_sin = sin_sin_4d.squeeze(0).squeeze(0)  # [max_seq_len, dim]
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

    # Rotation transform matrix tensor
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
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=trans_mem,
        tile=trans_tile,
        mesh_mapper=mesh_mapper,
    )

    # Metadata / position IDs
    metadata_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
    )
    ttnn_metadata_tensor = create_metadata_tensor(submesh, metadata_core_grid, metadata)

    # KV cache (ND sharded DRAM)
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
    torch_kv_cache = torch.zeros((num_slots, 1, max_seq_len, kvpe_dim), dtype=torch.bfloat16)
    torch_kv_cache[:, :, :position_id, :] = torch.randn(num_slots, 1, position_id, kvpe_dim, dtype=torch.bfloat16)
    torch_kv_cache_shuffled = deinterleave_kv_cache(torch_kv_cache, dcs, num_sp)
    kv_cache_2d_mesh_mapper = ttnn.ShardTensor2dMesh(submesh, mesh_shape=(mesh_rows, mesh_cols), dims=(2, None))
    if position_id == 0:
        ttnn_kv_cache = DRAMZeroFill.allocate_kv_cache_on_device(
            submesh,
            num_users=torch_kv_cache.shape[0],
            max_seq_len=max_seq_len,
            kvpe_dim=kvpe_dim,
            dtype=ttnn.bfloat8_b,
            mesh_shape=(mesh_rows, mesh_cols),
        )
    else:
        ttnn_kv_cache = ttnn.from_torch(
            torch_kv_cache_shuffled,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=kv_mem,
            mesh_mapper=kv_cache_2d_mesh_mapper,
        )

    # KV cache clone for standalone AttentionBlock validation
    ttnn_kv_cache_attn_ref = None
    if validate_debug_tensors:
        if position_id == 0:
            ttnn_kv_cache_attn_ref = DRAMZeroFill.allocate_kv_cache_on_device(
                submesh,
                num_users=torch_kv_cache.shape[0],
                max_seq_len=max_seq_len,
                kvpe_dim=kvpe_dim,
                dtype=ttnn.bfloat8_b,
                mesh_shape=(mesh_rows, mesh_cols),
            )
        else:
            ttnn_kv_cache_attn_ref = ttnn.from_torch(
                torch_kv_cache_shuffled,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=submesh,
                memory_config=kv_mem,
                mesh_mapper=kv_cache_2d_mesh_mapper,
            )

    # SDPA output tensor
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
    ttnn_sdpa_output = None
    if validate_debug_tensors:
        ttnn_sdpa_output = ttnn.from_torch(
            torch.zeros((SDPA_INPUT_NUM_CORES * HEADS_PER_ROW, QNOPE_OUT_DIM), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=sdpa_mem,
            mesh_mapper=mesh_mapper,
            tile=sdpa_tile,
        )

    # Post-SDPA tensors
    a_tile = ttnn.Tile([M, 32])
    shard_mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    gather_core = ttnn.CoreCoord(12, 9)
    gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

    # Attention block output / MoE residual input (overlapped with sdpa_kv_cache_buffer)
    # These are temporally disjoint: the kv cache on core (12,9) is done after SDPA,
    # so the attention output and MoE residual input can reuse that L1 region.
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
    attn_ref_output = None
    if validate_debug_tensors:
        attn_ref_output = ttnn.from_torch(
            mesh_output_torch,
            device=submesh,
            layout=ttnn.TILE_LAYOUT,
            tile=a_tile,
            dtype=ttnn.bfloat16,
            memory_config=output_mem_config,
            mesh_mapper=shard_mesh_mapper,
        )

    # SDPA worker/forwarder tensors
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
    # THIS BUFFER SIZE IS NOT CORRECT BECAUSE WE'RE INCORRECTLY DIVIDING BY 2
    # TODO: Plan to remove this scratch buffer entirely once we reduce cb memory usage currently being overlapped with this buffer.
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

    # Reduce-to-one tensors
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

    # Standalone MoE reference reduce tensors (MoE only)
    if is_moe:
        moe_ref_reduce_intermediate = None
        moe_ref_reduce_output = None
        if validate_debug_tensors:
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

    # Routed weight tensors differ between MoE (list) and dense (single tensor)
    routed_gate = weights.routed_gate_proj[0] if is_moe else weights.routed_gate_proj
    routed_up = weights.routed_up_proj[0] if is_moe else weights.routed_up_proj
    routed_down = weights.routed_down_proj[0] if is_moe else weights.routed_down_proj

    result = {
        # Attention weights (from prepare_*_layer_weights)
        "gamma_overlapped": weights.attn_norm,
        "matmul_weights_overlapped": weights.q_a_proj,
        "rmsnorm2_gamma_overlapped": weights.q_norm,
        "matmul2_weights_overlapped": weights.q_b_proj,
        "matmul3_weights_overlapped": weights.kv_b1_proj,
        "dkv_matmul_weights_overlapped": weights.kv_a_proj,
        "dkv_rmsnorm_gamma_overlapped": weights.kv_norm,
        "kv_b2_overlapped": weights.kv_b2_proj,
        "o_proj_overlapped": weights.o_proj,
        "ffn_norm_overlapped": weights.ffn_norm,
        # Attention activation/buffer tensors
        "input_tensor_mesh": input_tensor_mesh,
        "ttnn_qrope_sin": ttnn_qrope_sin,
        "ttnn_qrope_cos": ttnn_qrope_cos,
        "ttnn_trans_mat": ttnn_trans_mat,
        "ttnn_krope_cos": ttnn_krope_cos,
        "ttnn_krope_sin": ttnn_krope_sin,
        "ttnn_kv_cache": ttnn_kv_cache,
        "ttnn_kv_cache_attn_ref": ttnn_kv_cache_attn_ref,
        "ttnn_metadata_tensor": ttnn_metadata_tensor,
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
        "shared_gate_weights_overlapped": weights.shared_gate_proj,
        "shared_up_weights_overlapped": weights.shared_up_proj,
        "shared_down_weights_tensor": weights.shared_down_proj,
        "shared_k_parallel": SharedExpert.K_PARALLEL,
        "shared_n_parallel": SharedExpert.N_PARALLEL,
        # Reduce-to-one
        "reduce_intermediate_tensors": intermediate_tensors,
        "reduce_output_tensor": reduce_output_tensor,
        "reduce_root_coord": reduce_root_coord,
        "num_gate_proj_cores": num_gate_proj_cores,
        "per_core_down_proj_N": per_core_down_proj_N,
        "mcast_grid": mcast_grid,
        "forward_metadata": forward_metadata,
        # Intermediate CPU tensors (for golden-reference builders)
        "torch_input": torch_input,
        "torch_kv_cache": torch_kv_cache,
        "torch_sin": torch_sin,
        "torch_cos": torch_cos,
        "torch_position_ids": position_ids,
    }
    # MoE-only keys
    if is_moe:
        result.update(
            {
                "gate_mm_overlapped": weights.gate_mm,
                "ttnn_gate_bias": weights.gate_bias,
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


class DecoderStage(StageKind):
    """Shared implementation for MoE and dense decoder pipeline stages."""

    MOE_SENDER_CORE = ttnn.CoreCoord(12, 9)

    def __init__(
        self,
        *,
        weights: DeepSeekV3MoELayerWeights | DeepSeekV3DenseLayerWeights,
        layer_idx: int,
        metadata: DeepseekMetadata,
        max_seq_len: int,
        num_slots: int,
        persistent_mode: bool,
        is_torus: bool,
        is_moe: bool,
        num_routed_experts: int,
        use_hardcoded_expert_index: bool,
        enable_routing: bool,
        forward_metadata: bool = True,
        upstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        downstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        host_loopback: bool = False,
    ) -> None:
        if not isinstance(weights, (DeepSeekV3MoELayerWeights, DeepSeekV3DenseLayerWeights)):
            raise ValueError(
                f"Invalid weights type: {type(weights)}, expected DeepSeekV3MoELayerWeights or DeepSeekV3DenseLayerWeights"
            )
        if is_moe and not isinstance(weights, DeepSeekV3MoELayerWeights):
            raise ValueError(f"MoE weights must be a DeepSeekV3MoELayerWeights, got {type(weights)}")
        if not is_moe and not isinstance(weights, DeepSeekV3DenseLayerWeights):
            raise ValueError(f"Dense weights must be a DeepSeekV3DenseLayerWeights, got {type(weights)}")

        self._weights = weights
        self._layer_idx = layer_idx
        self._metadata = metadata
        self._max_seq_len = max_seq_len
        self._num_slots = num_slots
        self._persistent_mode = persistent_mode
        self._is_torus = is_torus
        self._is_moe = is_moe
        self._num_routed_experts = num_routed_experts
        self._use_hardcoded_expert_index = use_hardcoded_expert_index
        self._enable_routing = enable_routing
        self._forward_metadata = forward_metadata
        self._upstream_fifo_pages = upstream_fifo_pages
        self._downstream_fifo_pages = downstream_fifo_pages
        self._host_loopback = host_loopback
        self._num_links_bcast = 1
        self._num_links_allreduce = 2
        self._state: dict[str, Any] = {}

    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        mesh_device = ctx.mesh_device
        pipeline_config = ctx.pipeline_config
        my_stage_idx = ctx.my_stage_idx

        gate_proj_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(mesh_device, ttnn.NOC.NOC_0)
        gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
        shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
        aggregator_core = shard_cores_list[0]

        stage_entry_device = pipeline_config[my_stage_idx].entry_node_coord
        reduce_root_coord = pipeline_config[my_stage_idx].exit_node_coord

        exit_upstream_cores = [ttnn.MeshCoreCoord(reduce_root_coord, c) for c in shard_cores_list]
        assert (
            ACTIVATION_PAGE_SIZE_BYTES % len(shard_cores_list) == 0
        ), "ACTIVATION_PAGE_SIZE_BYTES must be divisible by len(shard_cores_list)"

        exit_upstream_page_size = ACTIVATION_PAGE_SIZE_BYTES // len(shard_cores_list)

        if self._forward_metadata:
            page_size = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES
        else:
            page_size = ACTIVATION_PAGE_SIZE_BYTES
        upstream_fifo_size = activation_fifo_size_bytes(page_size, self._upstream_fifo_pages)
        downstream_fifo_size = activation_fifo_size_bytes(page_size, self._downstream_fifo_pages)

        return PipelineBlock(
            mesh_device,
            PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=upstream_fifo_size,
            downstream_d2d_socket_fifo_size=downstream_fifo_size,
            upstream_d2d_socket_page_size=page_size,
            downstream_d2d_socket_page_size=page_size,
            entry_node_downstream=ttnn.MeshCoreCoord(stage_entry_device, self.MOE_SENDER_CORE),
            exit_node_upstream=exit_upstream_cores,
            exit_upstream_page_size=exit_upstream_page_size,
            forward_metadata=self._forward_metadata,
            my_stage_idx=my_stage_idx,
            stages_metadata=ctx.stages_metadata,
            pipeline_config=pipeline_config,
            loopback=LoopbackConfig.host_loopback(HostIoPlacement.default(PIPELINE_CORE_COORD))
            if self._host_loopback
            else LoopbackConfig.fabric_loopback(HostIoPlacement.default(PIPELINE_CORE_COORD)),
        )

    def _build_decoder_program_context(self) -> tuple[Any, Any, Any]:
        """Build DecoderBlock program before pipeline launch; requires ``self._state`` fully populated."""
        d = self._state["d"]
        if self._is_moe:
            gate_mm_weights_tensor = d["gate_mm_overlapped"]
            gate_bias_tensor = d["ttnn_gate_bias"]
            gate_indices_tensor = d["ttnn_gate_indices"]
            gate_output_scores_tensor = d["gate_output_scores_tensor"]
            gate_output_indices_tensor = d["gate_output_indices_tensor"]
            enable_routing = self._enable_routing
            use_hardcoded_expert_index = self._use_hardcoded_expert_index
        else:
            gate_mm_weights_tensor = None
            gate_bias_tensor = None
            gate_indices_tensor = None
            gate_output_scores_tensor = None
            gate_output_indices_tensor = None
            enable_routing = False
            use_hardcoded_expert_index = False

        return DecoderBlock.get_program_context(
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
            d["ttnn_metadata_tensor"],
            d["scale"],
            d["sdpa_kv_cache_buffer"],
            d["sdpa_out_interm_buffer"],
            d["sender_coord"],
            d["kv_b2_overlapped"],
            d["o_proj_overlapped"],
            None,
            None,
            None,
            d["ttnn_sdpa_intermediate_recv"],
            d["ttnn_sdpa_forwarder_scratch"],
            d["device_chunk_size"],
            d["ttnn_attention_block_output"],
            attention_block_semaphores=self._state["attn_semaphores"],
            shared_residual_mcast_src_tensor=d["ttnn_residual_mcast_src"],
            gate_mm_weights_tensor=gate_mm_weights_tensor,
            gate_bias_tensor=gate_bias_tensor,
            gate_indices_tensor=gate_indices_tensor,
            gate_output_scores_tensor=gate_output_scores_tensor,
            gate_output_indices_tensor=gate_output_indices_tensor,
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
            moe_semaphores=self._state["moe_semaphores"],
            reduce_intermediate_tensors=d["reduce_intermediate_tensors"],
            reduce_output_tensor=d["reduce_output_tensor"],
            reduce_semaphores=self._state["reduce_semaphores"],
            reduce_root_coord=self._state["reduce_root_coord"],
            enable_routing=enable_routing,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            reduce_cluster_axis=1,
            sdpa_cluster_axis=0,
            num_links_bcast=self._num_links_bcast,
            num_links_allreduce=self._num_links_allreduce,
            skip_ccl=False,
            upstream_socket=self._state["recv_socket"],
            downstream_sockets=self._state["downstream_sockets"],
            persistent_next_iter_semaphore=self._state.get("persistent_next_iter_semaphore"),
            persistent_mode=self._persistent_mode,
            termination_semaphore=self._state.get("termination_semaphore"),
            is_torus=self._is_torus,
            forward_metadata=self._forward_metadata,
        )

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        mesh_device = ctx.mesh_device
        pipeline_config = ctx.pipeline_config
        my_stage_idx = ctx.my_stage_idx

        sender_coord = pipeline_config[my_stage_idx].entry_node_coord
        reduce_root_coord = pipeline_config[my_stage_idx].exit_node_coord

        num_cores = mesh_device.compute_with_storage_grid_size().x * mesh_device.compute_with_storage_grid_size().y
        available_cores = ttnn.num_cores_to_corerangeset(
            num_cores, mesh_device.compute_with_storage_grid_size(), row_wise=True
        )

        attn_semaphores = AttentionBlock.create_semaphores(
            mesh_device, num_links_bcast=self._num_links_bcast, num_links_allreduce=self._num_links_allreduce
        )
        moe_semaphores = MoeOp.create_semaphores(mesh_device)
        reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(4)]
        self._persistent_loop = PersistentLoop(mesh_device, available_cores, self._persistent_mode)

        if self._is_moe:
            d = create_decoder_block_tensors(
                mesh_device,
                mesh_device.shape[0],
                mesh_device.shape[1],
                sender_coord[0],
                sender_coord[1],
                self._metadata.position_id,
                self._max_seq_len,
                reduce_root_coord=reduce_root_coord,
                weights=self._weights,
                metadata=self._metadata,
                num_slots=self._num_slots,
                forward_metadata=self._forward_metadata,
            )
        else:
            d = create_decoder_block_tensors(
                mesh_device,
                mesh_device.shape[0],
                mesh_device.shape[1],
                sender_coord[0],
                sender_coord[1],
                self._metadata.position_id,
                self._max_seq_len,
                reduce_root_coord=reduce_root_coord,
                weights=self._weights,
                metadata=self._metadata,
                num_slots=self._num_slots,
                is_moe=False,
                forward_metadata=self._forward_metadata,
            )
        ttnn.synchronize_device(mesh_device)

        recv_socket = pipeline_block.get_downstream_socket()
        downstream_sockets = pipeline_block.get_upstream_sockets()

        self._state = {
            "d": d,
            "attn_semaphores": attn_semaphores,
            "moe_semaphores": moe_semaphores,
            "reduce_semaphores": reduce_semaphores,
            "reduce_root_coord": reduce_root_coord,
            "recv_socket": recv_socket,
            "downstream_sockets": downstream_sockets,
        }

        if self._persistent_mode:
            self._state["persistent_next_iter_semaphore"] = self._persistent_loop.next_iter_semaphore
            self._state["termination_semaphore"] = self._persistent_loop.termination_semaphore

        self._state["decoder_program_context"] = self._build_decoder_program_context()

        logger.info(f"[rank={my_stage_idx}] {type(self).__name__} setup complete")

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        DecoderBlock.execute(*self._state["decoder_program_context"])

    def terminate(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        self._persistent_loop.terminate()


class MoEDecoderStage(DecoderStage):
    """Decoder stage: bcast + fused attention + MoE + reduce-to-one.

    Requires ``weights`` as a pre-loaded :class:`DeepSeekV3MoELayerWeights`
    (typically from ``WeightProvider.load_moe_layer``).
    """

    def __init__(
        self,
        *,
        weights: DeepSeekV3MoELayerWeights,
        layer_idx: int = 4,
        num_routed_experts: int = 256,
        metadata: DeepseekMetadata = DeepseekMetadata(),
        max_seq_len: int = 128 * 1024,
        num_slots: int = 64,
        persistent_mode: bool = True,
        use_hardcoded_expert_index: bool = False,
        enable_routing: bool = True,
        is_torus: bool = True,
        forward_metadata: bool = False,
        upstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        downstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        host_loopback: bool = False,
    ) -> None:
        super().__init__(
            weights=weights,
            layer_idx=layer_idx,
            metadata=metadata,
            max_seq_len=max_seq_len,
            num_slots=num_slots,
            persistent_mode=persistent_mode,
            is_torus=is_torus,
            is_moe=True,
            num_routed_experts=num_routed_experts,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
            enable_routing=enable_routing,
            forward_metadata=forward_metadata,
            upstream_fifo_pages=upstream_fifo_pages,
            downstream_fifo_pages=downstream_fifo_pages,
            host_loopback=host_loopback,
        )


class DenseDecoderStage(DecoderStage):
    """Dense decoder stage: bcast + fused attention + dense MLP + reduce-to-one.

    Requires ``weights`` as a pre-loaded :class:`DeepSeekV3DenseLayerWeights`
    (typically from ``WeightProvider.load_dense_layer``).
    """

    def __init__(
        self,
        *,
        weights: DeepSeekV3DenseLayerWeights,
        layer_idx: int = 0,
        metadata: DeepseekMetadata = DeepseekMetadata(),
        max_seq_len: int = 128 * 1024,
        num_slots: int = 64,
        persistent_mode: bool = True,
        is_torus: bool = True,
        forward_metadata: bool = False,
        upstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        downstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        host_loopback: bool = False,
    ) -> None:
        super().__init__(
            weights=weights,
            layer_idx=layer_idx,
            metadata=metadata,
            max_seq_len=max_seq_len,
            num_slots=num_slots,
            persistent_mode=persistent_mode,
            is_torus=is_torus,
            is_moe=False,
            num_routed_experts=0,
            use_hardcoded_expert_index=False,
            enable_routing=False,
            forward_metadata=forward_metadata,
            upstream_fifo_pages=upstream_fifo_pages,
            downstream_fifo_pages=downstream_fifo_pages,
            host_loopback=host_loopback,
        )
