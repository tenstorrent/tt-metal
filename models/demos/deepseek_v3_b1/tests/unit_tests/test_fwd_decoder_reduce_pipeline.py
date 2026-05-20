# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Two-stage pipeline integration test for forward + DecoderBlock (attention + MoE/MLP) + reduce-to-all.

Mirrors test_fwd_moe_reduce_pipeline.py but uses the full DecoderBlock (attention + FFN)
instead of standalone MoeOp. Tests the new multi-entry/exit decoder pipeline:

Stage 0:
  Parallel H2D/D2H + HostInterface (one per device) with embedding lookup.
  ParallelSocketInterface connects exit (-> stage 1) and entry (<- loopback).
Stage 1:
  PipelineBlock with pipeline_device_coords for per-device parallel forwarding.
  Entry D2D -> moe_sender_core socket -> forward + fused DecoderBlock + reduce-to-all -> exit D2D.
"""

import sys
import time

import pytest
import torch
from loguru import logger

import ttnn


def _log(msg):
    """Timestamped, immediately-flushed log line."""
    t = time.perf_counter()
    print(f"[{t:.3f}] {msg}", flush=True)
    sys.stdout.flush()


from conftest import requires_hybrid_allocator
from models.common.utility_functions import comp_pcc, is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.decoder_stage import create_decoder_block_tensors
from models.demos.deepseek_v3_b1.demo.stage import ACTIVATION_PAGE_SIZE_BYTES
from models.demos.deepseek_v3_b1.fused_ops.attention_block.op import AttentionBlock
from models.demos.deepseek_v3_b1.fused_ops.decoder_block.op import DecoderBlock
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.metadata.metadata import DeepseekMetadata
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import (
    MeshWrapper,
    ParallelSocketInterface,
    _combine_overlapping_programs,
    _group_by_device,
)
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import HostIoPlacement, LoopbackConfig, PipelineBlock
from models.demos.deepseek_v3_b1.model_dimensions import RoutedExpert
from models.demos.deepseek_v3_b1.tests.unit_tests.test_decoder_block import create_decoder_golden_tensors
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import DENSE_LAYER_IDX, extract_routed_expert_output
from models.demos.deepseek_v3_b1.utils import get_pinned_optimal_dram_bank_to_logical_worker_assignment
from models.demos.deepseek_v3_b1.weights.prepare import prepare_dense_layer_weights


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _dump_program_cores(prog, label=""):
    """Dump core usage and CB info for a ProgramDescriptor."""
    cores = set()
    for k in prog.kernels:
        for c in ttnn.corerange_to_cores(k.core_ranges):
            cores.add((c.x, c.y))
    return cores


def _dispatch_merged_programs(all_entries, mesh_device, io_tensors=None):
    """Merge (device_coord, program) entries by device and dispatch in a single generic_op."""
    _log(f"_dispatch_merged: {len(all_entries)} entries")
    if io_tensors is None:
        dummy_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device
        )
        io_tensors = [dummy_tensor, dummy_tensor]
    groups = _group_by_device(all_entries)
    mesh_program_descriptor = ttnn.MeshProgramDescriptor()
    for device_coord, progs in groups:
        if len(progs) > 1:
            progs = _combine_overlapping_programs(progs)
            merged = ttnn.merge_program_descriptors(progs) if len(progs) > 1 else progs[0]
        else:
            merged = progs[0]
        mesh_program_descriptor[ttnn.MeshCoordinateRange(device_coord, device_coord)] = merged
    _log("_dispatch_merged: calling generic_op...")
    result = ttnn.generic_op(io_tensors, mesh_program_descriptor)
    _log("_dispatch_merged: generic_op returned")
    return result


def build_worker_grid_excluding_cores(device_grid_size, excluded_cores):
    """Build a CoreRangeSet covering the full device grid minus a set of excluded cores."""
    full_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
    )
    excluded_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in excluded_cores])
    return full_grid.subtract(excluded_set)


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_router_config": create_fabric_router_config(15232),
            "worker_l1_size": 1431568,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("vocab_size, embedding_dim", [(64, 7168)])
@pytest.mark.parametrize("token_id", [0])
@pytest.mark.parametrize("position_id", [0])
@pytest.mark.parametrize("max_seq_len", [32 * 1024])
@pytest.mark.timeout(12000)
@requires_hybrid_allocator
def test_fwd_decoder_reduce_pipeline(
    mesh_device,
    vocab_size,
    embedding_dim,
    token_id,
    position_id,
    max_seq_len,
    device_params,
    get_reference_model_state_dict,
):
    """Two-stage pipeline: stage 0 = H2D/D2H, stage 1 = full DecoderBlock (dense MLP) with forward + reduce-to-all."""
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    _log(f"test start, enabling async slow dispatch")
    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    _log(f"rank={my_mesh_id}, num_procs={num_procs}")
    if num_procs < 2:
        pytest.skip(f"Requires at least 2 distributed processes, got {num_procs}")

    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small (need >= 13x10)")

    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(True)
    assert len(pipeline_config) == num_procs + 1

    is_stage0 = my_mesh_id == 0
    is_stage1 = my_mesh_id == 1

    K = RoutedExpert.K  # 7168
    epsilon = 1e-6

    pipeline_core = ttnn.CoreCoord(12, 8)
    core_io = ttnn.CoreCoord(0, 2)
    moe_sender_core = ttnn.CoreCoord(12, 9)

    token_size_bytes = DeepseekMetadata.aligned_size_bytes()
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_factor = 1
    embedding_fifo_size = embedding_size_bytes * embedding_fifo_factor

    torch_embedding = torch.arange(vocab_size * K, dtype=torch.float32).reshape(1, 1, vocab_size, K).to(torch.bfloat16)

    # -- Core setup for reduce aggregation --
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(mesh_device, gate_proj_noc)
    num_gate_proj_cores = len(gate_proj_worker_cores)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
    shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)

    mesh_rows, mesh_cols = mesh_device.shape
    num_devices = int(mesh_rows) * int(mesh_cols)
    device_coords = [ttnn.MeshCoordinate(r, c) for r in range(int(mesh_rows)) for c in range(int(mesh_cols))]

    # Determine forward entry / reduce exit columns from the pipeline config.
    entry_column = 0
    reduce_exit_column = 0
    pipeline_idx = my_mesh_id if my_mesh_id > 0 else 1
    if len(pipeline_config) > 1:
        entry_column = int(pipeline_config[pipeline_idx].entry_node_coord[1])
        reduce_exit_column = int(pipeline_config[pipeline_idx].exit_node_coord[1])

    exit_column = entry_column
    entry_column_coords = [ttnn.MeshCoordinate(r, entry_column) for r in range(int(mesh_rows))]
    exit_column_coords = [ttnn.MeshCoordinate(r, reduce_exit_column) for r in range(int(mesh_rows))]

    logger.info(
        f"entry_column={entry_column}, reduce_exit_column={reduce_exit_column}, "
        f"entry_devices={len(entry_column_coords)}, exit_devices={len(exit_column_coords)}"
    )

    # -- Decoder tensor setup (stage 1 only needs device tensors) --
    layer_idx = DENSE_LAYER_IDX
    sender_row = 1
    sender_col = entry_column

    state_dict = None
    layer_weights = None
    d = None

    if is_stage0 or is_stage1:
        logger.info(f"[rank={my_mesh_id}] preparing dense MLP model state dict")
        state_dict = get_reference_model_state_dict(
            layer_idx=layer_idx,
            is_moe=False,
            seed=RoutedExpert.SEED,
        )

        logger.info(f"[rank={my_mesh_id}] preparing dense layer weights on device")
        layer_weights = prepare_dense_layer_weights(mesh_device, state_dict, layer_idx, move_to_device=True)

        logger.info(f"[rank={my_mesh_id}] creating decoder block tensors")
        torch_input_for_decoder = torch_embedding[0, 0, token_id : token_id + 1, :]
        d = create_decoder_block_tensors(
            mesh_device,
            int(mesh_rows),
            int(mesh_cols),
            sender_row,
            sender_col,
            position_id,
            max_seq_len=max_seq_len,
            weights=layer_weights,
            metadata=DeepseekMetadata(position_id=position_id, slot_id=0),
            num_slots=1,
            is_moe=False,
            entry_column=sender_col,
            torch_input=torch_input_for_decoder,
        )
        logger.info(f"[rank={my_mesh_id}] decoder block tensors created")

    # Pre-compute golden reference while device is idle (before pipeline dispatch)
    golden_flat = None
    if is_stage0:
        logger.info("[rank=0] pre-computing golden reference")
        golden = create_decoder_golden_tensors(
            d,
            mesh_device,
            int(mesh_rows),
            int(mesh_cols),
            sender_row,
            sender_col,
            state_dict,
            layer_idx,
            metadata=DeepseekMetadata(position_id=position_id, slot_id=0),
            max_seq_len=max_seq_len,
            num_slots=1,
            is_moe=False,
        )

        QNOPE_HEAD_DIM = 128
        QROPE_HEAD_DIM = 64
        KNOPE_DIM = 512
        KROPE_DIM = 64
        HEADS_PER_ROW = 8

        _full_q, _golden_new_kv, _mla_output, _scores, _indices, moe_output = DecoderBlock.golden(
            golden["golden_torch_input"],
            golden["golden_torch_gamma"],
            golden["golden_torch_matmul_weights"],
            golden["golden_torch_rmsnorm2_gamma"],
            golden["golden_torch_matmul2_weights"],
            golden["golden_torch_matmul3_weights"],
            golden["golden_torch_sin"],
            golden["golden_torch_cos"],
            golden["golden_metadata"],
            golden["golden_torch_dkv_matmul_weights"],
            golden["golden_torch_dkv_rmsnorm_gamma"],
            golden["golden_torch_kv_cache"],
            golden["golden_scale"],
            golden["golden_torch_kv_b2_proj_weights"],
            golden["golden_torch_o_proj_weights"],
            epsilon=epsilon,
            num_qnope_heads=golden["golden_total_qnope_heads"],
            num_qrope_heads=golden["golden_total_qrope_heads"],
            qnope_head_dim=QNOPE_HEAD_DIM,
            qrope_head_dim=QROPE_HEAD_DIM,
            heads_per_row=HEADS_PER_ROW,
            nope_dim=KNOPE_DIM,
            rope_dim=KROPE_DIM,
            moe_shared_gate_weights=golden["golden_moe_shared_gate"],
            moe_shared_up_weights=golden["golden_moe_shared_up"],
            moe_shared_down_weights=golden["golden_moe_shared_down"],
            moe_gate_proj_weights_dict=golden["golden_moe_gate_proj_dict"],
            moe_up_proj_weights_dict=golden["golden_moe_up_proj_dict"],
            moe_down_proj_weights_dict=golden["golden_moe_down_proj_dict"],
            moe_rmsnorm_gamma=golden["golden_moe_rmsnorm_gamma"],
            moe_rmsnorm_epsilon=epsilon,
            moe_enable_routing=False,
        )

        golden_flat = moe_output.flatten().to(torch.bfloat16)
        logger.info(f"[rank=0] golden pre-computed, first 5 values: {golden_flat[:5].tolist()}")

    # Compute per-shard reduce payload
    reduce_payload_per_shard = ACTIVATION_PAGE_SIZE_BYTES // num_gate_proj_cores

    # -- Pipeline block setup (collective -- all hosts must participate simultaneously) --
    pipeline_block = None
    h2d_sockets = None
    d2h_sockets = None
    host_ios = None
    exit_socket_interface = None
    entry_socket_interface = None
    stage0_program_entries = None

    if is_stage0:
        embedding_tensor = ttnn.from_torch(
            torch_embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # PipelineBlock.__init__ calls generate_blitz_decode_pipeline (collective).
        # Stage 0 must match so all processes participate.
        ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(True)

        stage0_exit_col = int(pipeline_config[0].exit_node_coord[1])
        stage0_exit_dcs = [ttnn.MeshCoordinate(r, stage0_exit_col) for r in range(int(mesh_rows))]
        loopback_entry_col = int(pipeline_config[num_procs].entry_node_coord[1])
        loopback_entry_dcs = [ttnn.MeshCoordinate(r, loopback_entry_col) for r in range(int(mesh_rows))]

        fwd_send_d2d_cores = [ttnn.MeshCoreCoord(dc, pipeline_core) for dc in stage0_exit_dcs]
        fwd_recv_d2d_cores = [ttnn.MeshCoreCoord(dc, pipeline_core) for dc in entry_column_coords]
        bwd_send_d2d_cores = [ttnn.MeshCoreCoord(dc, pipeline_core) for dc in exit_column_coords]
        bwd_recv_d2d_cores = [ttnn.MeshCoreCoord(dc, pipeline_core) for dc in loopback_entry_dcs]

        exit_socket_interface = ParallelSocketInterface(
            embedding_size_bytes,
            embedding_fifo_size,
            send_core_coords=fwd_send_d2d_cores,
            recv_core_coords=fwd_recv_d2d_cores,
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=my_mesh_id + 1),
        )

        entry_socket_interface = ParallelSocketInterface(
            embedding_size_bytes,
            embedding_fifo_size,
            send_core_coords=bwd_send_d2d_cores,
            recv_core_coords=bwd_recv_d2d_cores,
            sender_mesh=MeshWrapper(mesh_id=num_procs - 1),
            receiver_mesh=MeshWrapper(mesh_device),
            receiver_use_reader_config=True,
        )

        h2d_sockets = []
        d2h_sockets = []
        host_ios = []

        for dc_idx in range(len(stage0_exit_dcs)):
            h2d = ttnn.H2DSocket(
                mesh_device,
                ttnn.MeshCoreCoord(stage0_exit_dcs[dc_idx], core_io),
                ttnn.BufferType.L1,
                token_size_bytes * 2,
                ttnn.H2DMode.HOST_PUSH,
            )
            d2h = ttnn.D2HSocket(
                mesh_device,
                ttnn.MeshCoreCoord(loopback_entry_dcs[dc_idx], core_io),
                embedding_fifo_size,
            )
            hio = HostInterface(
                h2d,
                d2h,
                token_size_bytes,
                embedding_size_bytes,
                core_to_core_socket_buffer_size=embedding_fifo_size,
                h2d_downstream_core=ttnn.MeshCoreCoord(stage0_exit_dcs[dc_idx], pipeline_core),
                d2h_upstream_core=ttnn.MeshCoreCoord(loopback_entry_dcs[dc_idx], pipeline_core),
                embedding_tensor=embedding_tensor,
            )
            h2d_sockets.append(h2d)
            d2h_sockets.append(d2h)
            host_ios.append(hio)

        for i, hio in enumerate(host_ios):
            exit_socket_interface._upstream_sockets[i] = hio.get_downstream_socket()
            entry_socket_interface._downstream_sockets[i] = hio.get_upstream_socket()

        all_entries = []
        for hio in host_ios:
            all_entries.extend(hio._build_programs())
        exit_progs = exit_socket_interface.build_programs()
        combined_progs = entry_socket_interface.build_programs(base_programs=exit_progs)

        all_entries.extend(exit_progs)
        all_entries.extend(combined_progs)
        stage0_program_entries = all_entries

        logger.info(f"[rank=0] parallel stage 0 programs built ({len(exit_column_coords)} channels), dispatch deferred")

    elif is_stage1:
        fabric_loopback = LoopbackConfig.fabric_loopback(HostIoPlacement.default(pipeline_core))
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
            pipeline_device_coords=device_coords,
            pipeline_exit_core_coord=pipeline_core,
            entry_downstream_core=moe_sender_core,
            exit_upstream_cores=shard_cores_list,
            exit_upstream_page_size=reduce_payload_per_shard,
            entry_device_coords=entry_column_coords,
            exit_device_coords=exit_column_coords,
            loopback=fabric_loopback,
        )

    else:
        fabric_loopback = LoopbackConfig.fabric_loopback(HostIoPlacement.default(pipeline_core))
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
            pipeline_device_coords=device_coords,
            pipeline_exit_core_coord=pipeline_core,
            entry_device_coords=entry_column_coords,
            exit_device_coords=exit_column_coords,
            loopback=fabric_loopback,
        )

    _log(f"[rank={my_mesh_id}] pipeline block created")

    # Diagnostic: verify exit socket interface configuration
    if is_stage1 and hasattr(pipeline_block, "exit_socket_interface"):
        for si_idx, si in enumerate(pipeline_block.exit_socket_interface):
            has_upstream_list = si.upstream_sockets_list is not None
            list_len = len(si.upstream_sockets_list) if has_upstream_list else 0
            _log(
                f"[rank=1] exit_si[{si_idx}] multi_upstream={si.multi_upstream} "
                f"has_upstream_sockets_list={has_upstream_list} list_len={list_len} "
                f"send_core={si.send_core_coord}"
            )

    # -- Get per-device sockets for forward (input) and reduce (output) --
    forward_sockets = None
    downstream_sockets = None
    if is_stage1:
        _log(f"[rank=1] getting downstream/upstream sockets from pipeline_block")
        raw_fwd = pipeline_block.get_downstream_sockets()
        _log(f"[rank=1] got {len(raw_fwd)} downstream sockets")
        raw_ds = pipeline_block.get_upstream_sockets()
        _log(f"[rank=1] got {len(raw_ds)} upstream sockets")
        forward_sockets = [None] * num_devices
        downstream_sockets = [None] * num_devices
        for idx, row_idx in enumerate(range(int(mesh_rows))):
            fwd_chip_id = row_idx * int(mesh_cols) + entry_column
            ds_chip_id = row_idx * int(mesh_cols) + reduce_exit_column
            forward_sockets[fwd_chip_id] = raw_fwd[idx]
            downstream_sockets[ds_chip_id] = raw_ds[idx]

    # -- Stage 1: create decoder semaphores and program context --
    if is_stage1:
        use_fp32 = False
        noc_mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC
        reduce_cluster_axis = 1
        num_links_bcast = 1
        num_links_allreduce = 2

        num_cores = device_grid.x * device_grid.y
        available_cores = ttnn.num_cores_to_corerangeset(num_cores, device_grid, row_wise=True)
        _log(f"[rank=1] synchronize_device before semaphore creation")
        ttnn.synchronize_device(mesh_device)
        _log(f"[rank=1] creating reduce_semaphores")
        reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(4)]
        _log(f"[rank=1] creating persistent_next_iter_semaphore")
        persistent_next_iter_semaphore = ttnn.create_global_semaphore(mesh_device, available_cores, 1)
        _log(f"[rank=1] synchronize_device after semaphore creation")
        ttnn.synchronize_device(mesh_device)

        _log(f"[rank=1] creating attn_semaphores")
        attn_semaphores = AttentionBlock.create_semaphores(
            mesh_device, num_links_bcast=num_links_bcast, num_links_allreduce=num_links_allreduce
        )
        _log(f"[rank=1] creating moe_semaphores")
        moe_semaphores = MoeOp.create_semaphores(mesh_device)
        _log(f"[rank=1] all semaphores created")

        # Allocate forward staging tensor before pipeline_block.run()
        _log(f"[rank=1] allocating forward_staging_tensor")
        forward_staging_tensor = ttnn.from_torch(
            d["torch_input"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(device_grid.x - 1, 9), ttnn.CoreCoord(device_grid.x - 1, 9))]
                    ),
                    (1, K),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            tile=ttnn.Tile([1, 32]),
        )
        _log(f"[rank=1] forward_staging_tensor allocated, synchronize_device")
        ttnn.synchronize_device(mesh_device)
        _log(f"[rank={my_mesh_id}] decoder semaphores and staging tensor created")

        # Build decoder program context
        _log(f"[rank={my_mesh_id}] calling DecoderBlock.get_program_context...")
        decoder_program_context = DecoderBlock.get_program_context(
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
            # MoE parameters (dense MLP: no gate_mm / routing)
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
            reduce_root_coord=d["reduce_root_coord"],
            # Shared parameters
            enable_routing=False,
            reduce_cluster_axis=reduce_cluster_axis,
            sdpa_cluster_axis=0,
            num_links_bcast=num_links_bcast,
            num_links_allreduce=num_links_allreduce,
            epsilon=epsilon,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
            use_hardcoded_expert_index=False,
            noc_mode=noc_mode,
            num_iterations=1,
            upstream_socket=None,
            downstream_sockets=downstream_sockets,
            fabric_config=device_params["fabric_config"],
            persistent_next_iter_semaphore=persistent_next_iter_semaphore,
            persistent_mode=False,
            # Forward + multi-exit
            forward_sockets=forward_sockets,
            forward_staging_tensor=forward_staging_tensor,
            exit_column=exit_column,
            reduce_exit_column=reduce_exit_column,
            forward_metadata_size_bytes=0,
        )
        _log(f"[rank={my_mesh_id}] DecoderBlock.get_program_context returned")
        logger.info(f"[rank={my_mesh_id}] decoder program context built")

    # -- Launch pipeline programs --
    # Two-phase dispatch mirroring the working MOE test (test_fwd_moe_reduce_pipeline.py):
    #   Phase 1: ALL stages dispatch D2D programs (non-blocking with dummy tensors)
    #   Phase 2: Stage 1 dispatches decoder compute ONLY (blocking with real IO tensors)
    # This ensures D2D entry/exit kernels are already running and waiting for data
    # before the compute kernel starts reading from forward_socket.
    if is_stage0:
        _dispatch_merged_programs(stage0_program_entries, mesh_device)
    else:
        pipeline_block.run()
    logger.info(f"[rank={my_mesh_id}] pipeline programs launched")

    ttnn.distributed_context_barrier()

    # -- Stage 0: inject token --
    if is_stage0:
        _log(f"[rank=0] injecting token {token_id}")
        token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
        torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
        torch_token[0, 0] = token_id
        token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for h2d in h2d_sockets:
            h2d.write_tensor(token_tensor)
        logger.info(f"[rank=0] token {token_id} injected to {len(h2d_sockets)} channels")

    # Second barrier: ensures token has been injected and is flowing through the D2D
    # pipeline before compute dispatch (matches MOE test pattern exactly).
    ttnn.distributed_context_barrier()

    # -- Stage 1: dispatch decoder compute ONLY (like MoeOp.op() in MOE test) --
    # D2D programs are already running from pipeline_block.run() above.
    # The decoder reads from forward_socket (data arrives via entry D2D from stage 0),
    # computes attention + FFN, reduce writes to exit D2D's upstream sockets.
    if is_stage1:
        io_tensors, _mpd, attention_block_output_tensor, decoder_entries = decoder_program_context
        _log(f"[rank=1] dispatching decoder compute ({len(decoder_entries)} entries)")
        _dispatch_merged_programs(decoder_entries, mesh_device, io_tensors=io_tensors)
        _log("[rank=1] decoder compute dispatch returned")

    # -- Stage 0: D2H loopback read + golden validation --
    # D2H read_tensor blocks until data flows back through the pipeline
    # (decoder processes → writes to downstream sockets → exit D2D sends → D2H receives).
    if is_stage0:
        _log("[rank=0] starting D2H reads")
        num_elements = embedding_size_bytes // 2
        d2h_results = []
        for sock_idx, d2h_sock in enumerate(d2h_sockets):
            buf = torch.zeros(1, num_elements, dtype=torch.bfloat16)
            buf_tensor = ttnn.from_torch(buf, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            _log(f"[rank=0] reading D2H socket[{sock_idx}]...")
            d2h_sock.read_tensor(buf_tensor)
            _log(f"[rank=0] D2H socket[{sock_idx}] read done")
            result = ttnn.to_torch(buf_tensor)
            nz = torch.count_nonzero(result)
            _log(f"[rank=0] D2H socket[{sock_idx}]: non-zero={nz}/{result.numel()} first5={result[0, :5]}")
            d2h_results.append(result)

        d2h_result_torch = None
        for sock_idx, result in enumerate(d2h_results):
            if torch.count_nonzero(result) > 0:
                d2h_result_torch = result
                _log(f"[rank=0] using D2H socket[{sock_idx}] for golden comparison")
                break
        assert d2h_result_torch is not None, "All D2H sockets returned zeros -- reduce or pipeline failed"

        logger.info("[rank=0] validating D2H output against pre-computed golden")
        assert golden_flat is not None, "Golden reference was not pre-computed"

        reduce_shard_width = reduce_payload_per_shard // dtype_size(ttnn.bfloat16)
        logger.info(
            f"[rank=0] golden: shape={tuple(golden_flat.shape)} nonzero={int(torch.count_nonzero(golden_flat).item())} "
            f"first5={golden_flat[:5].tolist()} last5={golden_flat[-5:].tolist()}"
        )
        logger.info(
            f"[rank=0] golden stats: min={golden_flat.float().min().item():.4f} max={golden_flat.float().max().item():.4f} "
            f"mean={golden_flat.float().mean().item():.4f}"
        )
        logger.info(
            f"[rank=0] reduce layout: num_gate_proj_cores={num_gate_proj_cores} "
            f"reduce_shard_width={reduce_shard_width} per_core_down_proj_N={d['per_core_down_proj_N']}"
        )

        all_passing = True
        per_socket_pccs = []
        for sock_idx, result in enumerate(d2h_results):
            nz = int(torch.count_nonzero(result).item())
            result_flat = result.flatten()
            logger.info(
                f"[rank=0] D2H socket[{sock_idx}] raw: shape={tuple(result.shape)} nonzero={nz} "
                f"first5={result_flat[:5].tolist()} last5={result_flat[-5:].tolist()}"
            )
            logger.info(
                f"[rank=0] D2H socket[{sock_idx}] raw stats: min={result_flat.float().min().item():.4f} "
                f"max={result_flat.float().max().item():.4f} mean={result_flat.float().mean().item():.4f}"
            )
            if nz == 0:
                logger.error(f"[rank=0] D2H socket[{sock_idx}] produced ALL ZEROS")
                all_passing = False
                per_socket_pccs.append((sock_idx, False, "all zeros"))
                continue

            d2h_extracted = extract_routed_expert_output(
                result, num_gate_proj_cores, reduce_shard_width, d["per_core_down_proj_N"]
            )
            logger.info(
                f"[rank=0] D2H socket[{sock_idx}] extracted: shape={tuple(d2h_extracted.shape)} "
                f"first5={d2h_extracted.flatten()[:5].tolist()} last5={d2h_extracted.flatten()[-5:].tolist()}"
            )
            logger.info(
                f"[rank=0] D2H socket[{sock_idx}] extracted stats: "
                f"min={d2h_extracted.float().min().item():.4f} max={d2h_extracted.float().max().item():.4f} "
                f"mean={d2h_extracted.float().mean().item():.4f}"
            )

            raw_pcc_passing, raw_pcc_msg = comp_pcc(golden_flat, result_flat, 0.97)
            logger.info(f"[rank=0] D2H socket[{sock_idx}] raw PCC vs golden: {raw_pcc_msg}")

            passing, pcc_msg = comp_pcc(golden_flat, d2h_extracted.flatten(), 0.97)
            per_socket_pccs.append((sock_idx, passing, pcc_msg))
            logger.info(f"[rank=0] D2H socket[{sock_idx}] extracted PCC vs golden: {pcc_msg}")
            if not passing:
                all_passing = False

        for sock_idx, p, msg in per_socket_pccs:
            logger.info(f"[rank=0] summary socket[{sock_idx}]: passing={p} pcc={msg}")
    ttnn.distributed_context_barrier()

    if is_stage0:
        assert all_passing, f"Stage 0 D2H golden PCC check failed: {per_socket_pccs}"

    # -- Pipeline teardown --
    _log(f"[rank={my_mesh_id}] starting pipeline termination")
    if is_stage0:
        ttnn.distributed_context_barrier()
        for i, hio in enumerate(host_ios):
            _log(f"[rank=0] terminating hio[{i}]")
            hio.terminate(False)
        _log("[rank=0] terminating entry_socket_interface")
        entry_socket_interface.terminate(False)
        _log("[rank=0] terminating exit_socket_interface")
        exit_socket_interface.terminate(True)
        _log("[rank=0] all interfaces terminated")
    else:
        _log(f"[rank={my_mesh_id}] calling pipeline_block.terminate()")
        pipeline_block.terminate()
        _log(f"[rank={my_mesh_id}] pipeline_block.terminate() returned")
    logger.info(f"[rank={my_mesh_id}] programs terminated")

    logger.info(f"[rank={my_mesh_id}] test PASSED")
