# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN LM Head Sampling CCL Broadcast + Mcast + Matmul Op Test

In multi-device mode: CCL broadcasts input_a [1, 7168] from sender device to all
devices, then on each device the sender core multicasts to 101 matmul cores.
Each matmul core holds a weight shard [7168, N_per_core] and computes
[1, 7168] x [7168, N_per_core] -> [1, N_per_core].
Output stays width-sharded across matmul cores.

In single-device mode (skip_ccl=True): CCL is skipped and the input is used directly.
"""
import os
import struct

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_slow_dispatch, skip_with_llk_assert
from models.demos.deepseek_v3_b1.demo.pipeline import create_single_galaxy_spec_decode_pipeline_configuration
from models.demos.deepseek_v3_b1.demo.stage import TOKEN_META_PAGE_SIZE_BYTES
from models.demos.deepseek_v3_b1.demo.weight_provider import SyntheticWeightProvider, _build_synthetic_mtp_state_dict
from models.demos.deepseek_v3_b1.fused_ops.lm_head_sampling.op import LMHeadSampling
from models.demos.deepseek_v3_b1.metadata.metadata import METADATA_TENSOR_NUM_UINT32
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import MeshWrapper, SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import StageMetadata
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import (
    build_broadcast_test_inputs,
    create_fabric_router_config,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles
from models.demos.deepseek_v3_b1.utils import float_to_uint32
from models.demos.deepseek_v3_b1.weights.prepare import _MTP_LAYER_IDX
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

_LM_HEAD_SAMPLING_REFERENCE_PT_ENV = "DEEPSEEK_V3_LM_HEAD_SAMPLING_REFERENCE_PT"

_VOCAB_SIZE = 129280
_EMBED_HIDDEN = 7168
_LM_HEAD_N_SYNTHETIC = 101 * 160  # 16160
_LM_HEAD_SAMPLING_SEED = 42


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _per_shard_quantize(tensor: torch.Tensor, *, num_shards: int, shard_dim: int, dtype) -> torch.Tensor:
    """Quantize a tensor by splitting into shards first, matching ShardTensorToMesh.

    The pipeline's EphemeralTensorCache uses ``ShardTensorToMesh`` which splits the
    bf16 tensor into ``num_shards`` pieces along ``shard_dim`` and converts each
    shard to the target block-float dtype independently.  A single HOST round-trip
    on the full tensor can produce different shared exponents than per-shard
    conversion, so this helper replicates the per-shard path entirely on the HOST.
    """
    shards = tensor.chunk(num_shards, dim=shard_dim)
    quantized_shards = []
    for shard in shards:
        shard_c = shard.contiguous()
        tt = ttnn.from_torch(shard_c, dtype=dtype, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((32, 32)))
        quantized_shards.append(ttnn.to_torch(tt).to(torch.bfloat16))
    return torch.cat(quantized_shards, dim=shard_dim)


def compute_lm_head_sampling_golden(iterations: int):
    """Compute expected (base_token, spec_token) pairs for the MTP pipeline.

    Golden chain per iteration:
      1. Base LM head sampling → base_token
      2. MTP forward (RMSNorm → concat → per-shard EH projection) → mtp_logits
      3. Spec LM head sampling on mtp_logits → spec_token

    Per-shard quantization in step 2 replicates the hardware path where
    ShardTensorToMesh splits before block-float conversion.

    Weights match SyntheticWeightProvider:
      - Embedding: one-hot (emb[i, i % K] = 1)
      - LM head / norms / eh_proj: seeded randn, block-float quantized
    """
    K = _EMBED_HIDDEN

    # One-hot embedding (same as SyntheticWeightProvider.load_embedding)
    base_embed_w = torch.zeros(_VOCAB_SIZE, K, dtype=torch.bfloat16)
    base_embed_w[torch.arange(_VOCAB_SIZE), torch.arange(_VOCAB_SIZE, dtype=torch.int64) % K] = 1

    # Base LM head weights with folded RMSNorm, per-shard quantized
    g = torch.Generator().manual_seed(_LM_HEAD_SAMPLING_SEED)
    lm_w = torch.randn(_VOCAB_SIZE, K, generator=g, dtype=torch.bfloat16)
    norm_w = torch.randn(K, generator=g, dtype=torch.bfloat16).abs() + 0.1
    base_lm_w = _per_shard_quantize((lm_w * norm_w).T.contiguous(), num_shards=8, shard_dim=1, dtype=ttnn.bfloat8_b)

    indices = torch.arange(_VOCAB_SIZE, dtype=torch.int32).reshape(1, _VOCAB_SIZE)

    # MTP weights: fold e/h norms into eh_proj, per-shard quantize
    mtp_sd = _build_synthetic_mtp_state_dict()
    gamma_eh = torch.cat(
        [mtp_sd[f"model.layers.{_MTP_LAYER_IDX}.enorm.weight"], mtp_sd[f"model.layers.{_MTP_LAYER_IDX}.hnorm.weight"]],
        dim=0,
    ).unsqueeze(1)
    eh_proj_raw = mtp_sd[f"model.layers.{_MTP_LAYER_IDX}.eh_proj.weight"].T.contiguous()
    eh_proj = _per_shard_quantize(eh_proj_raw * gamma_eh, num_shards=8, shard_dim=0, dtype=ttnn.bfloat4_b)

    # Spec LM head weights with folded norm, per-shard quantized
    spec_lm_w_folded = mtp_sd["lm_head.weight"] * mtp_sd[f"model.layers.{_MTP_LAYER_IDX}.shared_head.norm.weight"]
    spec_lm_w = _per_shard_quantize(spec_lm_w_folded.T.contiguous(), num_shards=8, shard_dim=1, dtype=ttnn.bfloat8_b)

    k_slice_size = (K * 2) // 8  # concat dim (e_norm + h_norm) split across 8 devices
    eh_proj_f = eh_proj.float()
    sampling_kwargs = dict(indices=indices, k=1, p=1.0, temperature=0.6)

    results = []
    for iteration in range(iterations):
        row_idx = iteration % K
        hidden = base_embed_w[row_idx : row_idx + 1, :].clone()

        # Stage 1: Base LM head → base_token
        base_token_tensor, _ = LMHeadSampling.golden(
            hidden.float(),
            None,
            base_lm_w.float().unsqueeze(0),
            **sampling_kwargs,
        )
        base_token = base_token_tensor.to(torch.uint32).item()

        # Stage 2: MTP forward — per-shard EH projection to match hardware quantization
        h_input = hidden.float()
        h_norm = h_input * torch.rsqrt(h_input.pow(2).mean(-1, keepdim=True) + 1e-6)
        tok_emb = base_embed_w[iteration, :].unsqueeze(0).float()
        e_norm = tok_emb * torch.rsqrt(tok_emb.pow(2).mean(-1, keepdim=True) + 1e-6)
        concat = torch.cat([e_norm, h_norm], dim=-1).to(torch.bfloat16).to(torch.float32)

        mtp_logits = torch.zeros(1, K, dtype=torch.float32)
        for dev_idx in range(8):
            k_start = dev_idx * k_slice_size if dev_idx < 4 else K + (dev_idx - 4) * k_slice_size
            act_slice = concat[0, k_start : k_start + k_slice_size]
            partial = act_slice.unsqueeze(0) @ eh_proj_f[k_start : k_start + k_slice_size, :]
            mtp_logits += partial.to(torch.bfloat16).to(torch.float32)
        mtp_logits = mtp_logits.to(torch.bfloat16).to(torch.float32)

        # Stage 3: Spec LM head → spec_token
        spec_token_tensor, _ = LMHeadSampling.golden(
            mtp_logits,
            None,
            spec_lm_w.float().unsqueeze(0),
            **sampling_kwargs,
        )
        spec_token = spec_token_tensor.to(torch.uint32).item()

        results.append((base_token, spec_token))
    return results, []


def _is_lm_head_sampling_perf_enabled():
    return os.getenv("RUN_LM_HEAD_SAMPLING_PERF", "0") == "1"


def _is_persistent_mode_enabled():
    return os.getenv("TT_RUN_PERSISTENT_MODE", "0") == "1"


@pytest.mark.skipif(not _is_lm_head_sampling_perf_enabled(), reason="Set RUN_LM_HEAD_SAMPLING_PERF=1 to run perf test")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1), (1, 0), (2, 1), (2, 0)])
@pytest.mark.parametrize("num_iters,num_warmup_iters", [(20, 6)])
@pytest.mark.parametrize("enable_mtp", [False, True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 1600000,
        }
    ],
    indirect=True,
)
@skip_with_llk_assert("Skip perf tests with LLK asserts enabled.")
def test_perf(bh_2d_mesh_device, use_fp32, final_mesh_coord, num_iters, num_warmup_iters, device_params, enable_mtp):
    """Performance test for LM-head sampling with optional MTP fusion.

    When enable_mtp=True, also runs:
    - Embedding lookup from argmax output token
    - h_rmsnorm and e_rmsnorm
    - Concat [h_norm|e_norm]
    - EH projection DRAM streaming matmul
    """
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    seed = 7

    # MTP dimensions
    embedding_dim = 7168
    mtp_output_dim = 7168
    tile_width = 32
    num_dram_banks = 8
    mtp_n_per_core = mtp_output_dim // num_dram_banks
    mtp_padded_dim = num_dram_banks * mtp_n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)

    # MTP tensors (only used when enable_mtp=True)
    # Embedding table must cover all possible token IDs (num_devices * n_total for global indices)
    torch_embedding = torch.randn((num_devices * n_total, embedding_dim), dtype=torch.bfloat16) if enable_mtp else None
    torch_h_gamma = torch.randn((M, K), dtype=torch.bfloat16) if enable_mtp else None
    torch_e_gamma = torch.randn((M, embedding_dim), dtype=torch.bfloat16) if enable_mtp else None
    torch_eh_proj = torch.randn((K + embedding_dim, mtp_output_dim), dtype=torch.bfloat16) if enable_mtp else None
    torch_eh_proj_padded = None
    if enable_mtp:
        torch_eh_proj_padded = torch.zeros((K + embedding_dim, mtp_padded_dim), dtype=torch.bfloat16)
        torch_eh_proj_padded[:, :mtp_output_dim] = torch_eh_proj

    torch_expected_idx, torch_mtp_output = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
        fuse_mtp=enable_mtp,
        embedding_tensor=torch_embedding.float() if enable_mtp else None,
        h_gamma_tensor=torch_h_gamma.float() if enable_mtp else None,
        e_gamma_tensor=torch_e_gamma.float() if enable_mtp else None,
        eh_projection_tensor=torch_eh_proj.float() if enable_mtp else None,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes + (256 + 8 if enable_mtp else 0)) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    # MTP-specific memory configs
    compute_cores = submesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0) if enable_mtp else None
    compute_core_grid = (
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores])
        if enable_mtp
        else None
    )
    eh_shard_grid = (
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(submesh.dram_grid_size().x - 1, submesh.dram_grid_size().y - 1),
                )
            }
        )
        if enable_mtp
        else None
    )
    mtp_output_mem_config = (
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(compute_core_grid, (M, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
        )
        if enable_mtp
        else None
    )
    eh_proj_mem_config = (
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(eh_shard_grid, (K + embedding_dim, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
        )
        if enable_mtp
        else None
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=mcast_core,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
        input_tensor_torch=torch_a,
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    # MTP tensors
    ttnn_embedding = None
    ttnn_h_gamma = None
    ttnn_e_gamma = None
    ttnn_eh_proj = None
    ttnn_mtp_output = None
    if enable_mtp:
        ttnn_embedding = ttnn.from_torch(
            torch_embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=submesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        ttnn_h_gamma = ttnn.from_torch(
            torch_h_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=input_a_mem_config,
            tile=a_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        ttnn_e_gamma = ttnn.from_torch(
            torch_e_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=input_a_mem_config,
            tile=a_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        torch_eh_proj_shuffled = shuffle_tensor_tiles(torch_eh_proj_padded, tile_width, num_dram_banks)
        ttnn_eh_proj = ttnn.from_torch(
            torch_eh_proj_shuffled,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=eh_proj_mem_config,
            tile=b_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        ttnn_mtp_output = ttnn.from_torch(
            torch.zeros((num_devices, M, mtp_padded_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=mtp_output_mem_config,
            tile=out_tile,
            mesh_mapper=mesh_mapper,
        )

    mcast_dst_buf, mcast_eh_dst_buf = _create_mcast_working_bufs(
        submesh,
        mcast_core,
        matmul_core_grid,
        M,
        K,
        a_tile,
        embedding_dim=embedding_dim if enable_mtp else None,
        num_devices=num_devices,
        mesh_mapper=mesh_mapper,
    )

    stage1_semaphores = [ttnn.create_global_semaphore(submesh, final_core_grid, 0) for _ in range(2)]
    stage2_semaphores = [ttnn.create_global_semaphore(submesh, final_core_grid, 0) for _ in range(2)]
    ttnn.synchronize_device(submesh)

    submesh.enable_program_cache()
    profiler = BenchmarkProfiler()

    # Initial run to compile
    _ = LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        eh_mm_fused_buffer=ttnn_mtp_output,
        embedding_tensor=ttnn_embedding,
        h_gamma_tensor=ttnn_h_gamma,
        e_gamma_tensor=ttnn_e_gamma,
        eh_projection_tensor=ttnn_eh_proj,
        mcast_dst_working_buf_tensor=mcast_dst_buf,
        mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        bcast_semaphores=bcast_inputs.semaphores,
        global_semaphore=stage1_semaphores[0],
        global_stage2_semaphore=stage2_semaphores[0],
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        fabric_config=device_params["fabric_config"],
        enable_mtp=enable_mtp,
    )
    ttnn.synchronize_device(submesh)

    trace_id_warmup = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_warmup_iters):
        _ = LMHeadSampling.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            ttnn_gamma,
            ttnn_b,
            ttnn_scores,
            eh_mm_fused_buffer=ttnn_mtp_output,
            embedding_tensor=ttnn_embedding,
            h_gamma_tensor=ttnn_h_gamma,
            e_gamma_tensor=ttnn_e_gamma,
            eh_projection_tensor=ttnn_eh_proj,
            mcast_dst_working_buf_tensor=mcast_dst_buf,
            mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
            sender_coord=sender_coord,
            indices_tensor=ttnn_indices,
            output_index_tensor=ttnn_output_index,
            argmax_final_core_coord=final_core,
            argmax_final_mesh_coord=final_mesh_coord,
            bcast_semaphores=bcast_inputs.semaphores,
            global_semaphore=stage1_semaphores[i % 2],
            global_stage2_semaphore=stage2_semaphores[i % 2],
            fabric_scratch_tensor=ttnn_fabric_scratch,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
            enable_mtp=enable_mtp,
        )
    ttnn.end_trace_capture(submesh, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh)

    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_iters):
        _ = LMHeadSampling.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            ttnn_gamma,
            ttnn_b,
            ttnn_scores,
            eh_mm_fused_buffer=ttnn_mtp_output,
            embedding_tensor=ttnn_embedding,
            h_gamma_tensor=ttnn_h_gamma,
            e_gamma_tensor=ttnn_e_gamma,
            eh_projection_tensor=ttnn_eh_proj,
            mcast_dst_working_buf_tensor=mcast_dst_buf,
            mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
            sender_coord=sender_coord,
            indices_tensor=ttnn_indices,
            output_index_tensor=ttnn_output_index,
            argmax_final_core_coord=final_core,
            argmax_final_mesh_coord=final_mesh_coord,
            bcast_semaphores=bcast_inputs.semaphores,
            global_semaphore=stage1_semaphores[i % 2],
            global_stage2_semaphore=stage2_semaphores[i % 2],
            fabric_scratch_tensor=ttnn_fabric_scratch,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
            enable_mtp=enable_mtp,
        )
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)

    mtp_suffix = "+MTP" if enable_mtp else ""
    profiler.start(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace-warmup")
    ttnn.execute_trace(submesh, trace_id_warmup, blocking=False)
    ttnn.release_trace(submesh, trace_id_warmup)
    ttnn.synchronize_device(submesh)
    profiler.end(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace-warmup")

    signpost("start")
    profiler.start(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace")
    ttnn.execute_trace(submesh, trace_id, blocking=False)
    ttnn.release_trace(submesh, trace_id)
    ttnn.synchronize_device(submesh)
    profiler.end(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace")
    signpost("stop")

    trace_duration_ns = profiler.get_duration(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace")
    warmup_duration_ns = profiler.get_duration(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace-warmup")
    effective_duration_ns = max(0.0, trace_duration_ns - warmup_duration_ns)
    avg_iter_ns = effective_duration_ns / float(max(1, num_iters))
    logger.info(
        f"LMHead+Argmax{mtp_suffix} mesh(4x2) trace perf: final_mesh_coord={final_mesh_coord}, "
        f"iters={num_iters}, total_ns={effective_duration_ns:.2f}, avg_iter_ns={avg_iter_ns:.2f}"
    )

    final_output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_torch = ttnn.to_torch(final_output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    logger.info(f"Final output: {final_output_torch}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        final_output_torch, torch_expected_idx
    ), f"Perf run fused mesh argmax mismatch. expected={torch_expected_idx.item()}, got={int(final_output_torch.item())}"

    # MTP PCC check
    if enable_mtp:
        assert torch_mtp_output is not None, "MTP output cannot be None"
        final_mtp_shards = ttnn.get_device_tensors(ttnn_mtp_output)
        final_mtp_torch = (
            ttnn.to_torch(final_mtp_shards[final_device_idx])
            .to(torch.float32)
            .reshape(1, mtp_padded_dim)[:, :mtp_output_dim]
        )
        mtp_passing_pcc, _ = comp_pcc(final_mtp_torch, torch_mtp_output.float(), 0.99)
        if not mtp_passing_pcc:
            max_diff = (final_mtp_torch - torch_mtp_output.float()).abs().max()
            logger.warning(f"MTP output PCC check failed. Max diff: {max_diff}")
        assert mtp_passing_pcc, "Perf run MTP output PCC check failed"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("seed", [123, 1337, 52098])
@skip_with_llk_assert("Hit LLK_ASSERT for unpacker data format conversion. Issue: #41024")
@pytest.mark.skip(reason="Disabled: broken by PR #42662 (speculative decoding refactor). Tracked in #42964.")
def test_single_device(
    bh_2d_mesh_device,
    use_fp32,
    seed,
    device_params,
):
    """Single-device fused LM-head + argmax sampling with pre-cached width-sharded indices."""
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(), torch_gamma.float(), torch_b.float(), indices=torch_indices, k=1, p=1.0
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)]), (1, 1), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )

    input_tensor_mesh = ttnn.from_torch(
        torch_a,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
    )

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
    )
    ttnn.synchronize_device(submesh)

    output_index_torch = ttnn.to_torch(ttnn_output_index).to(torch.uint32).reshape(1, 1)
    logger.info(f"Output index: {output_index_torch}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        output_index_torch, torch_expected_idx
    ), f"Fused argmax index mismatch. expected={torch_expected_idx.item()}, got={output_index_torch.item()}"


@pytest.mark.skip(reason="Disabled: broken by PR #42662 (speculative decoding refactor). Tracked in #42964.")
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("seed", [1337])
@skip_with_llk_assert("Hit LLK_ASSERT for unpacker data format conversion. Issue: #41024")
@pytest.mark.skip(reason="Skipping test for now, TODO: use new metadata format for test")
def test_single_device_d2h(
    bh_2d_mesh_device,
    use_fp32,
    seed,
    device_params,
):
    """Single-device fused LM-head + argmax with optional D2H token emission enabled."""
    if not is_slow_dispatch():
        pytest.skip("Skipping D2H socket test in fast dispatch mode")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    d2h_page_size_bytes = 256

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(), torch_gamma.float(), torch_b.float(), indices=torch_indices, k=1, p=1.0
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)]), (1, 1), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )

    input_tensor_mesh = ttnn.from_torch(
        torch_a,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        torch.zeros_like(torch_a),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
    )

    d2h_socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), final_core)
    d2h_socket = ttnn.D2HSocket(submesh, d2h_socket_core, d2h_page_size_bytes * 4)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
        socket_output=d2h_socket,
    )

    output_index_torch = ttnn.to_torch(ttnn_output_index).to(torch.uint32).reshape(1, 1)
    assert torch.equal(
        output_index_torch, torch_expected_idx
    ), f"Fused argmax index mismatch. expected={torch_expected_idx.item()}, got={output_index_torch.item()}"

    d2h_page_words = d2h_page_size_bytes // 4
    d2h_read_tensor = ttnn.from_torch(
        torch.zeros((1, d2h_page_words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    d2h_socket.read_tensor(d2h_read_tensor)
    d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
    d2h_socket.barrier()
    ttnn.synchronize_device(submesh)
    logger.info(f"D2H token: {d2h_token}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        d2h_token, torch_expected_idx
    ), f"D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "sender_coord, final_mesh_coord, seed",
    [
        ((1, 1), (0, 0), 7),
        ((0, 0), (1, 1), 1337),
        ((3, 0), (2, 0), 4242),
    ],
)
@skip_with_llk_assert("Hit LLK_ASSERT for unpacker data format conversion. Issue: #41024")
def test_multidevice(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    sender_coord,
    seed,
    device_params,
):
    """4x2 mesh fused LM-head + k=1 sampling (argmax) with CCL enabled."""
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    # Global indices are unique across mesh devices.
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_mesh_coord = ttnn.MeshCoordinate(sender_coord[0], sender_coord[1])
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_mesh_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=mcast_core,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
        input_tensor_torch=torch_a,
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_mesh_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        bcast_semaphores=bcast_inputs.semaphores,
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        fabric_config=device_params["fabric_config"],
    )
    ttnn.synchronize_device(submesh)

    output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_index = ttnn.to_torch(output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    logger.info(f"Final output index: {final_output_index}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        final_output_index, torch_expected_idx
    ), f"Fused mesh argmax index mismatch. expected={torch_expected_idx.item()}, got={final_output_index.item()}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1)])
@pytest.mark.parametrize("seed", [5449])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
@skip_with_llk_assert("Hit LLK_ASSERT for unpacker data format conversion. Issue: #41024")
def test_d2h(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
    device_params,
):
    """4x2 mesh fused LM-head + argmax with optional D2H token emission on final mesh device."""
    if not is_slow_dispatch():
        pytest.skip("Skipping D2H socket test in fast dispatch mode")

    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    d2h_page_size_bytes = 256

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=mcast_core,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
        input_tensor_torch=torch_a,
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)

    d2h_socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(final_mesh_coord[0], final_mesh_coord[1]), final_core)
    d2h_socket = ttnn.D2HSocket(submesh, d2h_socket_core, d2h_page_size_bytes * 4)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        bcast_semaphores=bcast_inputs.semaphores,
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        socket_output=d2h_socket,
        fabric_config=device_params["fabric_config"],
    )
    ttnn.synchronize_device(submesh)

    output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_index = ttnn.to_torch(output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    assert torch.equal(
        final_output_index, torch_expected_idx
    ), f"Fused mesh argmax index mismatch. expected={torch_expected_idx.item()}, got={final_output_index.item()}"

    d2h_page_words = d2h_page_size_bytes // 4
    d2h_read_tensor = ttnn.from_torch(
        torch.zeros((1, d2h_page_words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    d2h_socket.read_tensor(d2h_read_tensor)
    d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
    logger.info(f"D2H token: {d2h_token}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        d2h_token, torch_expected_idx
    ), f"Mesh D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1), (0, 1), (1, 0), (0, 0), (3, 1)])
@pytest.mark.parametrize("seed", [5449])
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
@skip_with_llk_assert("Hit LLK_ASSERT for unpacker data format conversion. Issue: #41024")
def test_d2d_to_d2h_pipeline(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
    device_params,
):
    """4x2 mesh fused LM-head + argmax with D2D output routed through D2D forwarding to D2H."""
    if ttnn.get_num_devices() < 32:
        pytest.skip("Test requires a full galaxy")
    if not is_slow_dispatch():
        pytest.skip("Skipping D2D/D2H pipeline test in fast dispatch mode")

    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    ttnn.enable_asynchronous_slow_dispatch(submesh)

    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    socket_page_size_bytes = 256
    socket_fifo_size = 256

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    lmhead_input_core_x = 10
    lmhead_input_core_y = 9
    lmhead_input_core = ttnn.CoreCoord(lmhead_input_core_x, lmhead_input_core_y)
    lmhead_input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(lmhead_input_core, lmhead_input_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    argmax_final_core = ttnn.CoreCoord(0, 0)
    argmax_final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)])

    mcast_bbox = matmul_core_grid.bounding_box()
    reserved_cores = {(argmax_final_core.x, argmax_final_core.y), (lmhead_input_core.x, lmhead_input_core.y)}
    extra_cores = []
    for y in range(device_grid_size.y):
        for x in range(device_grid_size.x):
            if (x, y) in reserved_cores:
                continue
            if mcast_bbox.contains(ttnn.CoreCoord(x, y)):
                continue
            extra_cores.append(ttnn.CoreCoord(x, y))
    logger.info(f"Extra cores: {extra_cores}")
    if len(extra_cores) < 4:
        pytest.skip("Test requires at least 4 spare cores for D2D/D2H pipeline wiring")
    d2d1_core = ttnn.CoreCoord(11, 0)
    d2d2_core = ttnn.CoreCoord(11, 1)
    d2h_core = ttnn.CoreCoord(11, 2)
    dummy_h2d_core = ttnn.CoreCoord(11, 3)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(lmhead_input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=lmhead_input_core,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
        input_tensor_torch=torch_a,
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)

    final_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        argmax_final_core,
    )

    d2d1_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2d1_core,
    )
    d2d2_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2d2_core,
    )
    d2h_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2h_core,
    )
    dummy_h2d_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        dummy_h2d_core,
    )

    logger.info(f"final_mesh_core: {final_mesh_core}")
    logger.info(f"d2d1_mesh_core: {d2d1_mesh_core}")
    logger.info(f"d2d2_mesh_core: {d2d2_mesh_core}")
    logger.info(f"d2h_mesh_core: {d2h_mesh_core}")
    logger.info(f"dummy_h2d_mesh_core: {dummy_h2d_mesh_core}")

    h2d_socket = ttnn.H2DSocket(
        submesh, dummy_h2d_mesh_core, ttnn.BufferType.L1, socket_fifo_size, ttnn.H2DMode.HOST_PUSH
    )
    d2h_socket = ttnn.D2HSocket(submesh, d2h_mesh_core, socket_fifo_size)
    logger.info("Creating HostInterface")
    host_io = HostInterface(
        h2d_socket,
        d2h_socket,
        socket_page_size_bytes,
        socket_page_size_bytes,
        core_to_core_socket_buffer_size=socket_fifo_size,
        h2d_downstream_core=dummy_h2d_mesh_core,
        d2h_upstream_core=d2d2_mesh_core,
    )
    logger.info("Creating SocketInterface")
    socket_interface = SocketInterface(
        socket_page_size_bytes,
        socket_fifo_size,
        socket_page_size_bytes,
        d2d1_mesh_core,
        d2d2_mesh_core,
        upstream_core_coord=final_mesh_core,
        downstream_socket=host_io.get_upstream_socket(),
        sender_mesh=MeshWrapper(mesh_device=submesh),
        receiver_mesh=MeshWrapper(mesh_device=submesh),
    )

    logger.info("Running HostInterface")
    host_io.run()
    logger.info("Running SocketInterface")
    socket_interface.run()
    logger.info("Running LMHeadSampling")
    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=argmax_final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        bcast_semaphores=bcast_inputs.semaphores,
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        socket_output=socket_interface.get_upstream_socket(),
        fabric_config=device_params["fabric_config"],
    )
    d2h_page_words = socket_page_size_bytes // 4
    d2h_read_tensor = ttnn.from_torch(
        torch.zeros((1, d2h_page_words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    d2h_socket.read_tensor(d2h_read_tensor)
    d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
    logger.info(f"D2D->D2H token: {d2h_token}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        d2h_token, torch_expected_idx
    ), f"Mesh D2D->D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"

    host_io.terminate(False)
    socket_interface.terminate(True)

    ttnn.synchronize_device(submesh)


def parse_output_page(output_tensor: ttnn.Tensor) -> dict:
    """Parse a 256-byte DeepseekMetadata output page into a dict.

    Layout (64 uint32 words = 256 bytes):
      words  0-15 : header (tok0_id, tok0_type, tok0_pos, tok1_id, tok1_type, tok1_pos,
                             slot_id, token_id, position_id, prefill_token_id,
                             temperature, k, probability_mass_threshold, _pad0-2)
      words 16-47 : p_indices[32]  (uint32)
      words 48-63 : p_scores[32]   (bf16 packed as uint16, 2 per uint32)
    """
    raw = ttnn.to_torch(output_tensor).to(torch.int32).flatten()
    assert (
        raw.numel() >= METADATA_TENSOR_NUM_UINT32
    ), f"output tensor has {raw.numel()} words, expected >= {METADATA_TENSOR_NUM_UINT32}"

    def _u32_to_f32(bits: int) -> float:
        return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]

    p_indices = raw[16:48].tolist()
    scores_packed = raw[48:64].contiguous().view(torch.bfloat16)
    p_scores = scores_packed.float().tolist()

    return {
        "tok0_id": int(raw[0].item()),
        "tok0_type": int(raw[1].item()),
        "tok0_pos": int(raw[2].item()),
        "tok1_id": int(raw[3].item()),
        "tok1_type": int(raw[4].item()),
        "tok1_pos": int(raw[5].item()),
        "slot_id": int(raw[6].item()),
        "token_id": int(raw[7].item()),
        "position_id": int(raw[8].item()),
        "prefill_token_id": int(raw[9].item()),
        "temperature": _u32_to_f32(raw[10].item()),
        "k": int(raw[11].item()),
        "probability_mass_threshold": _u32_to_f32(raw[12].item()),
        "p_indices": p_indices,
        "p_scores": p_scores,
    }


def create_input_page(
    token_id: int,
    position_id: int,
    prefill_token_id: int,
    slot_id: int,
    temperature: float = 0.0,
    top_k: int = 0,
    probability_mass_threshold: float = 0.0,
) -> ttnn.Tensor:
    """Build a TOKEN_PAGE_SIZE_BYTES input page matching the DeepseekMetadata input layout.

    Word indices (from model.py InputField):
      [1] token_type  [2] tok0_position_id  [6] slot_id (user_id)
      [7] token_id    [8] position_id       [9] prefill_token_id
      [10] temperature (f32 bits)  [11] top_k  [12] probability_mass_threshold (f32 bits)
    """
    page = torch.zeros(1, METADATA_TENSOR_NUM_UINT32, dtype=torch.int32)
    page[0, 2] = position_id
    page[0, 6] = slot_id
    page[0, 7] = token_id
    page[0, 8] = position_id
    page[0, 9] = prefill_token_id
    page[0, 10] = float_to_uint32(temperature)
    page[0, 11] = top_k
    page[0, 12] = float_to_uint32(probability_mass_threshold)
    return ttnn.from_torch(page, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "worker_l1_size": 1453716,
        }
    ],
    indirect=True,
)
def test_persistent_mode_spec_decode(mesh_device, use_fp32):
    """4-stage 4x2 single-galaxy pipeline with MTP + verification:
    P1(Embedding + SpecLMHead) -> P2(BaseLMHead+EH Matmul) -> P3(Passthrough ACTIVATION_W_TOKEN_META) -> P4(Passthrough ACTIVATION_W_TOKEN_META) -> P1(D2H TOKEN_META).

    The verification stage (P1) receives gathered logits + token metadata, runs its
    own LM head + argmax, then outputs a TOKEN_META page (64 bytes) back to P1.

    TOKEN_META page layout (uint32 words):
      [0] num_tokens  (0=stale, 1=accept, 2=reject)
      [1] tok0_id     [2] tok0_type (0=BASE,1=SPEC)  [3] tok0_pos
      [4] tok1_id     [5] tok1_type                   [6] tok1_pos
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())

    iterations = 50
    run_golden = False

    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs == 4:
        config = create_single_galaxy_spec_decode_pipeline_configuration(
            SyntheticWeightProvider(fold_rmsnorm_weights=True),
            fp32_dest_acc_en=use_fp32,
        )
    elif num_procs == 16:
        config = create_single_pod_spec_decode_no_decoder_pipeline_configuration(
            SyntheticWeightProvider(fold_rmsnorm_weights=True),
            fp32_dest_acc_en=use_fp32,
        )
    else:
        raise ValueError(f"Test does not support {num_procs} distributed processes")

    print(f"[TEST] config created, building pipeline", flush=True)
    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(True)
    stages_metadata = {i: StageMetadata(rank=i, mesh_id=i) for i in range(num_procs)}
    pipeline = config.build_pipeline(
        mesh_device,
        stages_metadata=stages_metadata,
        pipeline_config=pipeline_config,
    )
    pid = pipeline.my_mesh_id
    logger.debug(f"[TEST P{pid}] pipeline built, calling setup_and_run")

    pos_id = 0
    slot_id = 0

    golden = None
    golden_debug = None
    pipeline.setup_and_run()
    logger.debug(f"[TEST P{pid}] setup_and_run complete")

    token_meta_words = TOKEN_META_PAGE_SIZE_BYTES // 4
    raw_indices = []
    raw_scores = []
    if pipeline.my_mesh_id == 0:
        if run_golden:
            logger.debug(f"[TEST] computing golden...")
            golden, golden_debug = compute_lm_head_sampling_golden(iterations)
            logger.debug(f"[TEST] golden computed, creating config")
        else:
            logger.debug(f"[TEST] skipping golden computation")

        output_tensor = ttnn.from_torch(
            torch.zeros(1, token_meta_words, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        for iteration in range(iterations):
            logger.debug(f"[TEST P{pid}] iter {iteration} write_token")
            token_tensor = create_input_page(
                token_id=10,
                position_id=iteration,
                prefill_token_id=10,
                slot_id=slot_id,
                temperature=0.6,
                top_k=32,
                probability_mass_threshold=1.0,
            )
            pipeline.write_token(token_tensor)
            logger.debug(f"[TEST P{pid}] iter {iteration} read_output")
            pipeline.read_output(output_tensor)

            page = parse_output_page(output_tensor)
            type_name = {0: "BASE", 1: "SPEC"}

            if run_golden:
                expected_base, expected_spec = golden[iteration]
            else:
                expected_base = None
                expected_spec = None

            nonzero_p_idx = [(i, v) for i, v in enumerate(page["p_indices"]) if v != 0]
            nonzero_p_sc = [(i, v) for i, v in enumerate(page["p_scores"]) if v != 0.0]

            if iteration < 50:
                raw_dump = ttnn.to_torch(output_tensor).flatten().tolist()
                raw_indices.append(raw_dump[16:48])
                raw_scores.append(raw_dump[48:64])
                hdr = [f"0x{int(v) & 0xFFFFFFFF:08X}" for v in raw_dump[:16]]
                idx = [f"0x{int(v) & 0xFFFFFFFF:08X}" for v in raw_dump[16:48]]
                scr = [f"0x{int(v) & 0xFFFFFFFF:08X}" for v in raw_dump[48:64]]
                print(f"[RAW P{pid}] iter {iteration} HDR={hdr}", flush=True)
                print(f"[RAW P{pid}] iter {iteration} IDX={idx}", flush=True)
                print(f"[RAW P{pid}] iter {iteration} SCR={scr}", flush=True)

            logger.info(
                f"[TEST P{pid}] iter {iteration} | "
                f"t0={page['tok0_id']}/{type_name.get(page['tok0_type'], '?')} pos={page['tok0_pos']} | "
                f"t1={page['tok1_id']}/{type_name.get(page['tok1_type'], '?')} pos={page['tok1_pos']} | "
                f"slot_id={page['slot_id']} token_id={page['token_id']} "
                f"position_id={page['position_id']} prefill_token_id={page['prefill_token_id']} | "
                f"temperature={page['temperature']:.4f} k={page['k']} "
                f"prob_mass_threshold={page['probability_mass_threshold']:.4f} | "
                f"p_indices(nonzero)={nonzero_p_idx} | "
                f"p_scores(nonzero)={nonzero_p_sc} | "
                f"golden base={expected_base} spec={expected_spec}"
            )

    # check if all raw scores and indices are the same – selection may vary based on random seed
    if pipeline.my_mesh_id == 0:
        for i in range(len(raw_indices)):
            assert raw_indices[i] == raw_indices[0], f"Raw indices for iteration {i} are not the same"
            assert raw_scores[i] == raw_scores[0], f"Raw scores for iteration {i} are not the same"
    logger.debug(f"[TEST P{pid}] all iterations done, barrier")
    pipeline.barrier()
    logger.debug(f"[TEST P{pid}] barrier done, terminate")
    pipeline.terminate()
    logger.debug(f"[TEST P{pid}] terminate done, final barrier")
    pipeline.barrier()
    logger.debug(f"[TEST P{pid}] final barrier done")
