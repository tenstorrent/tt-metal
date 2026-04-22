# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Two-stage pipeline integration test for forward + MoE fused op + reduce-to-all.

Uses parallel sockets across all 8 devices in the mesh. Each device independently
receives its input via a per-device forward operation (socket read -> residual tensor)
and produces its output via reduce-to-all (all devices hold the full reduced result).

Stage 0:
  8 parallel H2D/D2H + HostInterface (one per device) with embedding lookup.
  ParallelSocketInterface connects exit (-> stage 1) and entry (<- last stage loopback).
Stage 1:
  PipelineBlock with pipeline_device_coords for per-device parallel forwarding.
  Entry D2D -> moe_sender_core socket -> forward + fused MoE + reduce-to-all -> exit D2D.
Stage 2+ (if applicable):
  passive forwarding, no downstream op
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_slow_dispatch
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import (
    MeshWrapper,
    ParallelSocketInterface,
    _combine_overlapping_programs,
    _group_by_device,
)
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.model_dimensions import RoutedExpert
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import (
    ROUTED_EXPERT_LAYER_IDX,
    create_routed_expert_tensors,
    create_shared_expert_tensors,
    extract_routed_expert_output,
)
from models.demos.deepseek_v3_b1.utils import get_pinned_optimal_dram_bank_to_logical_worker_assignment


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _dispatch_merged_programs(all_entries, mesh_device):
    """Merge (device_coord, program) entries by device and dispatch in a single generic_op."""
    dummy_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device
    )
    groups = _group_by_device(all_entries)
    mesh_program_descriptor = ttnn.MeshProgramDescriptor()
    for device_coord, progs in groups:
        if len(progs) > 1:
            progs = _combine_overlapping_programs(progs)
            merged = ttnn.merge_program_descriptors(progs) if len(progs) > 1 else progs[0]
        else:
            merged = progs[0]
        mesh_program_descriptor[ttnn.MeshCoordinateRange(device_coord, device_coord)] = merged
    return ttnn.generic_op([dummy_tensor, dummy_tensor], mesh_program_descriptor)


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
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("vocab_size, embedding_dim", [(64, 7168)])
@pytest.mark.parametrize("token_id", [0])
@pytest.mark.timeout(12000)
def test_bcast_moe_reduce_pipeline(
    mesh_device, vocab_size, embedding_dim, token_id, device_params, get_reference_model_state_dict
):
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs < 2:
        pytest.skip(f"Requires at least 2 distributed processes, got {num_procs}")

    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for MoE (need >= 13x10)")

    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline()
    assert len(pipeline_config) == num_procs + 1

    is_torus = device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_2D_TORUS_Y

    is_stage0 = my_mesh_id == 0
    is_stage1 = my_mesh_id == 1

    M = RoutedExpert.M
    K = RoutedExpert.K

    pipeline_core = ttnn.CoreCoord(12, 8)
    core_io = ttnn.CoreCoord(0, 2)
    moe_sender_core = ttnn.CoreCoord(12, 9)
    moe_worker_core_grid = build_worker_grid_excluding_cores(device_grid, [pipeline_core])

    token_size_bytes = 64
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_factor = 1
    embedding_fifo_size = embedding_size_bytes * embedding_fifo_factor

    torch_embedding = torch.arange(vocab_size * K, dtype=torch.float32).reshape(1, 1, vocab_size, K).to(torch.bfloat16)

    reduce_root_coord = ttnn.MeshCoordinate(1, 0)

    # -- Core setup for reduce aggregation --
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(mesh_device, gate_proj_noc)
    num_gate_proj_cores = len(gate_proj_worker_cores)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
    shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
    aggregator_core = shard_cores_list[0]

    mesh_rows, mesh_cols = mesh_device.shape
    device_coords = [ttnn.MeshCoordinate(r, c) for r in range(int(mesh_rows)) for c in range(int(mesh_cols))]

    # Determine exit/entry column from pipeline config.
    exit_column = None
    if len(pipeline_config) > 1:
        entry_node = pipeline_config[0].exit_node_coord
        try:
            exit_column = int(entry_node[1])
        except Exception:
            exit_column = 0
    if exit_column is None:
        exit_column = 0
    pipeline_column_coords = [ttnn.MeshCoordinate(r, exit_column) for r in range(int(mesh_rows))]
    logger.info(f"exit_column={exit_column}, pipeline_column_coords={len(pipeline_column_coords)} devices")

    # -- MoE tensor setup (needed before PipelineBlock to get correct reduce shard size) --
    result_scores = None
    result_indices = None
    result_output = None
    r = None
    s = None
    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=256,
        include_global=False,
    )

    if is_stage0 or is_stage1:
        logger.info(f"[rank={my_mesh_id}] creating routed expert tensors")
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        logger.info(f"[rank={my_mesh_id}] mesh_mapper created")
        needs_device_tensors = is_stage1
        r = create_routed_expert_tensors(
            mesh_device,
            use_hardcoded_expert_index=True,
            mesh_mapper=mesh_mapper,
            state_dict=state_dict,
            is_moe=True,
            layer_idx=ROUTED_EXPERT_LAYER_IDX,
            skip_attention_weights=True,
            create_device_tensors=needs_device_tensors,
        )
        logger.info(f"[rank={my_mesh_id}] routed expert tensors created")
        mcast_grid = moe_worker_core_grid
        s = create_shared_expert_tensors(
            mesh_device,
            M,
            K,
            mcast_grid,
            mesh_mapper=mesh_mapper,
            state_dict=state_dict,
            is_moe=True,
            layer_idx=ROUTED_EXPERT_LAYER_IDX,
            create_device_tensors=needs_device_tensors,
        )
        logger.info(f"[rank={my_mesh_id}] MoE tensors created")

    # Compute per-shard reduce payload from actual shard spec (not from embedding dim)
    if r is not None:
        reduce_payload_per_shard = r.per_core_down_proj_N * 2  # bfloat16
        logger.info(
            f"[rank={my_mesh_id}] reduce_payload_per_shard={reduce_payload_per_shard} "
            f"(per_core_down_proj_N={r.per_core_down_proj_N})"
        )
    else:
        reduce_payload_per_shard = embedding_size_bytes // num_gate_proj_cores
    # Diagnostic: verify socket page sizes match expectations
    if r is not None:
        logger.info(
            f"[rank={my_mesh_id}] DIAG page sizes: "
            f"embedding_size_bytes={embedding_size_bytes} "
            f"reduce_payload_per_shard={reduce_payload_per_shard} "
            f"num_gate_proj_cores(pinned)={num_gate_proj_cores} "
            f"r.num_gate_proj_cores(device)={r.num_gate_proj_cores} "
            f"r.final_output_total_width={r.final_output_total_width} "
            f"r.final_output_width_per_core={r.final_output_width_per_core} "
            f"r.per_core_down_proj_N={r.per_core_down_proj_N} "
            f"len(shard_cores_list)={len(shard_cores_list)} "
            f"total_upstream_bytes={len(shard_cores_list)}*{reduce_payload_per_shard}={len(shard_cores_list)*reduce_payload_per_shard} "
            f"downstream_page_size(=embedding_size_bytes)={embedding_size_bytes}"
        )

    # -- Pipeline block setup (collective -- all hosts must participate simultaneously) --
    pipeline_block = None
    h2d_sockets = None
    d2h_sockets = None
    host_ios = None
    exit_socket_interface = None
    entry_socket_interface = None
    stage0_program_entries = None

    if is_stage0:
        import time as _time

        _t0_total = _time.time()
        print(f"[TEST] stage0: starting setup, my_mesh_id={my_mesh_id} num_procs={num_procs}", flush=True)

        embedding_tensor = ttnn.from_torch(
            torch_embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"[TEST] stage0: embedding on device, elapsed={_time.time()-_t0_total:.3f}s", flush=True)

        # Stages 1-15 call generate_blitz_decode_pipeline inside PipelineBlock.__init__,
        # which is a collective requiring all processes. Stage 0 must participate too.
        print(f"[TEST] stage0: calling generate_blitz_decode_pipeline (collective)...", flush=True)
        ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline()
        print(f"[TEST] stage0: generate_blitz_decode_pipeline done", flush=True)

        d2d_cores = [ttnn.MeshCoreCoord(dc, pipeline_core) for dc in pipeline_column_coords]
        print(f"[TEST] stage0: d2d_cores={[(str(c.device_coord),str(c.core_coord)) for c in d2d_cores]}", flush=True)

        print(f"[TEST] stage0: creating exit_socket_interface (PSI)...", flush=True)
        _t0 = _time.time()
        exit_socket_interface = ParallelSocketInterface(
            embedding_size_bytes,
            embedding_fifo_size,
            send_core_coords=d2d_cores,
            recv_core_coords=d2d_cores,
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=my_mesh_id + 1),
        )
        print(f"[TEST] stage0: exit_socket_interface done in {_time.time()-_t0:.3f}s", flush=True)

        print(f"[TEST] stage0: creating entry_socket_interface (PSI)...", flush=True)
        _t0 = _time.time()
        entry_socket_interface = ParallelSocketInterface(
            embedding_size_bytes,
            embedding_fifo_size,
            send_core_coords=d2d_cores,
            recv_core_coords=d2d_cores,
            sender_mesh=MeshWrapper(mesh_id=num_procs - 1),
            receiver_mesh=MeshWrapper(mesh_device),
            receiver_use_reader_config=True,
        )
        print(f"[TEST] stage0: entry_socket_interface done in {_time.time()-_t0:.3f}s", flush=True)

        print(f"[TEST] stage0: creating H2D/D2H/HostInterface for {len(pipeline_column_coords)} devices...")
        _t0 = _time.time()
        h2d_sockets = []
        d2h_sockets = []
        host_ios = []

        for dc_idx, dc in enumerate(pipeline_column_coords):
            _t_dc = _time.time()
            h2d = ttnn.H2DSocket(
                mesh_device,
                ttnn.MeshCoreCoord(dc, core_io),
                ttnn.BufferType.L1,
                token_size_bytes * 2,
                ttnn.H2DMode.HOST_PUSH,
            )
            d2h = ttnn.D2HSocket(
                mesh_device,
                ttnn.MeshCoreCoord(dc, core_io),
                embedding_fifo_size,
            )
            hio = HostInterface(
                h2d,
                d2h,
                token_size_bytes,
                embedding_size_bytes,
                core_to_core_socket_buffer_size=embedding_fifo_size,
                h2d_downstream_core=ttnn.MeshCoreCoord(dc, pipeline_core),
                d2h_upstream_core=ttnn.MeshCoreCoord(dc, pipeline_core),
                embedding_tensor=embedding_tensor,
            )
            h2d_sockets.append(h2d)
            d2h_sockets.append(d2h)
            host_ios.append(hio)
            print(f"[TEST] stage0:   device[{dc_idx}] dc={dc} done in {_time.time()-_t_dc:.3f}s", flush=True)

        print(f"[TEST] stage0: all H2D/D2H/HostInterface done in {_time.time()-_t0:.3f}s", flush=True)

        print(f"[TEST] stage0: wiring upstream/downstream sockets...", flush=True)
        for i, hio in enumerate(host_ios):
            exit_socket_interface._upstream_sockets[i] = hio.get_downstream_socket()
            entry_socket_interface._downstream_sockets[i] = hio.get_upstream_socket()
        print(f"[TEST] stage0: wiring done", flush=True)

        all_entries = []
        for hio in host_ios:
            all_entries.extend(hio._build_programs())
        exit_progs = exit_socket_interface.build_programs()
        combined_progs = entry_socket_interface.build_programs(base_programs=exit_progs)
        all_entries.extend(combined_progs)
        stage0_program_entries = all_entries
        print(
            f"[TEST] stage0: programs built (dispatch deferred until after tensor alloc), elapsed={_time.time()-_t0_total:.3f}s",
            flush=True,
        )
        logger.info(
            f"[rank=0] parallel stage 0 programs built ({len(pipeline_column_coords)} channels), dispatch deferred"
        )
    elif is_stage1:
        import time as _time

        _t0_total = _time.time()
        print(f"[TEST] stage1: creating PipelineBlock, my_mesh_id={my_mesh_id} num_procs={num_procs}", flush=True)
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
            entry_device_coords=pipeline_column_coords,
            exit_device_coords=pipeline_column_coords,
        )
        print(f"[TEST] stage1: PipelineBlock done in {_time.time()-_t0_total:.3f}s", flush=True)
        logger.info(
            f"[rank=1] DIAG PipelineBlock config: "
            f"upstream_d2d_socket_page_size={embedding_size_bytes} "
            f"downstream_d2d_socket_page_size={embedding_size_bytes} "
            f"exit_upstream_page_size={reduce_payload_per_shard} "
            f"num_exit_upstream_cores={len(shard_cores_list)} "
            f"total_exit_upstream_bytes={len(shard_cores_list)*reduce_payload_per_shard}"
        )
    else:
        import time as _time

        _t0_total = _time.time()
        print(f"[TEST] stage{my_mesh_id}: creating PipelineBlock", flush=True)
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
            pipeline_device_coords=device_coords,
            pipeline_exit_core_coord=pipeline_core,
            entry_device_coords=pipeline_column_coords,
            exit_device_coords=pipeline_column_coords,
        )
        print(f"[TEST] stage{my_mesh_id}: PipelineBlock done in {_time.time()-_t0_total:.3f}s", flush=True)

    logger.info(f"[rank={my_mesh_id}] pipeline block created")

    # -- Get per-device sockets for forward (input) and reduce (output) --
    # Only pipeline column devices have sockets; expand to 8-element lists indexed by chip_id.
    forward_sockets = None
    downstream_sockets = None
    if is_stage1:
        raw_fwd = pipeline_block.get_downstream_sockets()
        raw_ds = pipeline_block.get_upstream_sockets()
        num_devices = int(mesh_rows) * int(mesh_cols)
        forward_sockets = [None] * num_devices
        downstream_sockets = [None] * num_devices
        for idx, row_idx in enumerate(range(int(mesh_rows))):
            chip_id = row_idx * int(mesh_cols) + exit_column
            forward_sockets[chip_id] = raw_fwd[idx]
            downstream_sockets[chip_id] = raw_ds[idx]
        logger.info(
            f"[rank={my_mesh_id}] {len(raw_fwd)} forward sockets mapped to "
            f"{sum(1 for s in forward_sockets if s is not None)}/{num_devices} devices, "
            f"{len(raw_ds)} downstream socket groups"
        )
        for idx, row_idx in enumerate(range(int(mesh_rows))):
            chip_id = row_idx * int(mesh_cols) + exit_column
            ds = downstream_sockets[chip_id]
            if ds is not None:
                if isinstance(ds, list):
                    logger.info(
                        f"[rank=1] DIAG downstream_sockets[chip_id={chip_id}]: "
                        f"{len(ds)} sockets (one per upstream worker)"
                    )
                else:
                    logger.info(f"[rank=1] DIAG downstream_sockets[chip_id={chip_id}]: single socket")
    if is_stage1:
        kv_cache_shard_height = 256
        kvpe_dim = 576
        num_mcast_cores = mcast_grid.num_cores()
        kv_cache_shard_spec = ttnn.ShardSpec(
            mcast_grid, (kv_cache_shard_height, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR
        )
        sdpa_kv_cache_buffer = ttnn.from_torch(
            torch.zeros((kv_cache_shard_height * num_mcast_cores, kvpe_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
            ),
        )

        sdpa_out_interm_shard_height = 48
        sdpa_out_interm_shard_width = 1024
        num_worker_cores = moe_worker_core_grid.num_cores()
        sdpa_out_interm_shard_spec = ttnn.ShardSpec(
            moe_worker_core_grid,
            (sdpa_out_interm_shard_height, sdpa_out_interm_shard_width),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sdpa_out_interm_buffer = ttnn.from_torch(
            torch.zeros(
                (sdpa_out_interm_shard_height * num_worker_cores, sdpa_out_interm_shard_width),
                dtype=torch.bfloat16,
            ),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                sdpa_out_interm_shard_spec,
            ),
            tile=ttnn.Tile([8, 32]),
        )
        logger.info(f"[rank={my_mesh_id}] SDPA buffers created")

        tile_1x32 = ttnn.Tile([1, 32])
        final_output_total_width = r.final_output_total_width

        # -- Reduce-to-all tensors --
        reduce_mesh_mapper_config = ttnn.MeshMapperConfig(
            [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape
        )
        reduce_mesh_mapper = ttnn.create_mesh_mapper(mesh_device, reduce_mesh_mapper_config)

        final_output_mem_config = r.final_output_mem_config
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
        reduce_intermediate_tensors = ttnn.from_torch(
            torch.zeros([4, 2, final_output_total_width * 3], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=intermediate_mem_config,
            tile=tile_1x32,
            mesh_mapper=reduce_mesh_mapper,
        )
        logger.info(f"[rank={my_mesh_id}] reduce intermediate tensors created")

        compute_grid = mesh_device.compute_with_storage_grid_size()
        reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(aggregator_core, aggregator_core)})
        reduce_output_shard_spec = ttnn.ShardSpec(
            reduce_output_shard_grid,
            (1, final_output_total_width),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        reduce_output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, reduce_output_shard_spec
        )
        reduce_output_data = torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16)
        reduce_output_tensor = ttnn.from_torch(
            reduce_output_data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=reduce_output_mem_config,
            tile=tile_1x32,
            mesh_mapper=reduce_mesh_mapper,
        )
        logger.info(f"[rank={my_mesh_id}] reduce output tensor created")

        num_cores = compute_grid.x * compute_grid.y
        reduce_available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
        reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, reduce_available_cores, 0) for _ in range(4)]

        moe_semaphores = MoeOp.create_semaphores(mesh_device)
        logger.info(f"[rank={my_mesh_id}] semaphores created")

        # Allocate the dedicated cb46 forward staging buffer NOW (before
        # pipeline_block.run launches persistent kernels). Otherwise the
        # host-to-device upload performed by ttnn.from_torch deadlocks
        # against the in-flight dispatch queue.
        moe_K_single = r.ttnn_residual_mcast_src.shape[-1]
        moe_forward_staging_tensor_single = MoeOp.create_forward_staging_tensor(
            mesh_device,
            moe_sender_core,
            moe_K_single,
            dtype=r.ttnn_residual_mcast_src.dtype,
        )
        logger.info(
            f"[rank={my_mesh_id}] forward_staging_tensor allocated "
            f"(sender_core={moe_sender_core}, K={moe_K_single})"
        )

    # -- Launch pipeline programs --
    if is_stage0:
        _dispatch_merged_programs(stage0_program_entries, mesh_device)
        logger.info(f"[rank=0] stage0 programs dispatched (deferred)")
    else:
        pipeline_block.run()
    logger.info(f"[rank={my_mesh_id}] pipeline programs launched")

    logger.info(f"[rank={my_mesh_id}] waiting at barrier after pipeline launch")
    ttnn.distributed_context_barrier()
    logger.info(f"[rank={my_mesh_id}] passed barrier after pipeline launch")

    if is_stage0:
        logger.info(f"[rank=0] launching token injection")
        token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
        torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
        torch_token[0, 0] = token_id
        token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for h2d in h2d_sockets:
            h2d.write_tensor(token_tensor)
        logger.info(f"[rank=0] token {token_id} injected to {len(h2d_sockets)} channels")

    ttnn.distributed_context_barrier()

    if is_stage1:
        logger.info(f"[rank=1] launching MoE forward + reduce-to-all (num_iterations=1)")
        result_scores, result_indices, result_output = MoeOp.op(
            r.ttnn_residual_mcast_src,
            gate_mm_weights_tensor=r.ttnn_gate_mm_weights,
            gate_bias_tensor=r.ttnn_gate_bias,
            gate_indices_tensor=r.ttnn_gate_indices,
            gate_output_scores_tensor=r.gate_output_scores_tensor,
            gate_output_indices_tensor=r.gate_output_indices_tensor,
            gate_proj_weights_tensor=r.gate_proj_weights,
            up_proj_weights_tensor=r.up_proj_weights,
            down_proj_weights_tensor=r.down_proj_weights,
            final_output_tensor=r.final_output_tensor,
            rmsnorm_gamma_tensor=r.ttnn_rmsnorm_gamma,
            shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
            shared_up_weights_overlapped=s.shared_up_weights_overlapped,
            shared_down_weights_tensor=s.ttnn_down_weights,
            shared_k_parallel=s.k_parallel,
            shared_n_parallel=s.n_parallel,
            use_hardcoded_expert_index=True,
            sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer=sdpa_out_interm_buffer,
            num_iterations=1,
            reduce_intermediate_tensors=reduce_intermediate_tensors,
            reduce_output_tensor=reduce_output_tensor,
            reduce_semaphores=reduce_semaphores,
            reduce_root_coord=reduce_root_coord,
            forward_sockets=forward_sockets,
            forward_staging_tensor=moe_forward_staging_tensor_single,
            semaphores=moe_semaphores,
            worker_core_grid=moe_worker_core_grid,
            is_torus=is_torus,
            downstream_sockets=downstream_sockets,
            exit_column=exit_column,
        )
        logger.info("[rank=1] MoE + reduce-to-all completed")

    # -- Stage 0: D2H loopback read + golden validation --
    d2h_passing = True
    d2h_pcc_msg = ""
    if is_stage0:
        logger.info("[rank=0] waiting for D2H result from pipeline loopback")
        num_elements = embedding_size_bytes // 2
        # Read ALL D2H sockets to find which channel(s) carry valid data
        d2h_results = []
        for sock_idx, d2h_sock in enumerate(d2h_sockets):
            buf = torch.zeros(1, num_elements, dtype=torch.bfloat16)
            buf_tensor = ttnn.from_torch(buf, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            d2h_sock.read_tensor(buf_tensor)
            result = ttnn.to_torch(buf_tensor)
            nz = torch.count_nonzero(result)
            logger.info(f"[rank=0] D2H socket[{sock_idx}]: non-zero={nz}/{result.numel()} " f"first5={result[0, :5]}")
            d2h_results.append(result)

        # Pick the first socket with non-zero data for validation
        d2h_result_torch = None
        for sock_idx, result in enumerate(d2h_results):
            if torch.count_nonzero(result) > 0:
                d2h_result_torch = result
                logger.info(f"[rank=0] using D2H socket[{sock_idx}] for golden comparison")
                break
        assert d2h_result_torch is not None, "All D2H sockets returned zeros -- reduce or pipeline failed"

        from models.demos.deepseek_v3_b1.micro_ops.deepseek_moe_gate.op import DeepseekMoeGateSingleCore

        K_down = s.K_down
        torch_input_row = torch_embedding[0, 0, token_id : token_id + 1, :]

        x_raw = torch_input_row.float()
        variance = x_raw.pow(2).mean(-1, keepdim=True)
        normalized_input = ((x_raw * torch.rsqrt(variance + 1e-6)) * r.torch_rmsnorm_gamma.float()).bfloat16().float()
        logits = normalized_input.float() @ r.torch_gate_mm_weights.float()
        scores = torch.sigmoid(logits)
        gate_input = scores.reshape(1, 8, 32)
        cpu_gate_scores, _ = DeepseekMoeGateSingleCore.golden(
            gate_input, r.torch_bias.float(), r.gate_eps, r.gate_scaling_factor, enable_sigmoid=False
        )

        expected_final_outputs = []
        for dev_idx in range(mesh_rows * mesh_cols):
            actual_expert_scale = cpu_gate_scores.flatten()[dev_idx].float()

            shared_gate_shard = s.torch_gate_weights[:, dev_idx * K_down : (dev_idx + 1) * K_down]
            shared_up_shard = s.torch_up_weights[:, dev_idx * K_down : (dev_idx + 1) * K_down]
            shared_down_shard = s.torch_down_weights[dev_idx * K_down : (dev_idx + 1) * K_down, :]

            _, _, expected_final = MoeOp.golden_single_device(
                torch_input_row,
                shared_gate_weights=shared_gate_shard,
                shared_up_weights=shared_up_shard,
                shared_down_weights=shared_down_shard,
                gate_proj_weights_dict=r.expert_weights_dict,
                up_proj_weights_dict=r.up_proj_weights_dict,
                down_proj_weights_dict=r.down_proj_weights_dict,
                routing_weights_tensor=r.torch_gate_mm_weights,
                bias_tensor=r.torch_bias,
                eps=r.gate_eps,
                scaling_factor=r.gate_scaling_factor,
                use_hardcoded_expert_index=True,
                hardcoded_expert_index=dev_idx,
                explicit_expert_scale=actual_expert_scale,
                rmsnorm_gamma=r.torch_rmsnorm_gamma,
                rmsnorm_epsilon=1e-6,
                include_residual=(dev_idx == 0),
            )
            expected_final_outputs.append(expected_final)

        expected_reduce_output = sum(expected_final_outputs)

        reduce_shard_width = reduce_payload_per_shard // dtype_size(ttnn.bfloat16)
        d2h_valid = extract_routed_expert_output(
            d2h_result_torch, r.num_gate_proj_cores, reduce_shard_width, r.per_core_down_proj_N
        )

        passing, pcc_msg = comp_pcc(expected_reduce_output.flatten(), d2h_valid.flatten(), 0.9)
        logger.info(f"Pipeline Stage 0 D2H Reduce PCC: {pcc_msg}")
        assert passing, f"Pipeline Stage 0 D2H PCC check failed: {pcc_msg}"

    ttnn.distributed_context_barrier()

    # -- Pipeline teardown --
    logger.info(f"[rank={my_mesh_id}] waiting for pipeline termination")
    if is_stage0:
        for hio in host_ios:
            hio.terminate(False)
        entry_socket_interface.terminate(False)
        exit_socket_interface.terminate(True)
    else:
        pipeline_block.terminate()
    logger.info(f"[rank={my_mesh_id}] programs terminated")

    # -- Stage 1: validate MoE golden --
    stage1_reduce_passing = True
    if is_stage1:
        logger.info("[rank=1] validating MoE golden output")
        K_down = s.K_down

        torch_input_row = torch_embedding[0, 0, token_id : token_id + 1, :]

        device_gate_scores = ttnn.to_torch(result_scores, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

        reduce_output_torch = ttnn.to_torch(
            result_output,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )

        expected_final_outputs = []
        for dev_idx in range(mesh_rows * mesh_cols):
            actual_expert_idx = dev_idx
            actual_expert_scale = device_gate_scores[0].flatten()[dev_idx].float()

            shared_gate_shard = s.torch_gate_weights[:, dev_idx * K_down : (dev_idx + 1) * K_down]
            shared_up_shard = s.torch_up_weights[:, dev_idx * K_down : (dev_idx + 1) * K_down]
            shared_down_shard = s.torch_down_weights[dev_idx * K_down : (dev_idx + 1) * K_down, :]
            logger.info(f"[rank=1] validating device {dev_idx}: actual_expert_scale={actual_expert_scale}")
            _, _, expected_final = MoeOp.golden_single_device(
                torch_input_row,
                shared_gate_weights=shared_gate_shard,
                shared_up_weights=shared_up_shard,
                shared_down_weights=shared_down_shard,
                gate_proj_weights_dict=r.expert_weights_dict,
                up_proj_weights_dict=r.up_proj_weights_dict,
                down_proj_weights_dict=r.down_proj_weights_dict,
                routing_weights_tensor=r.torch_gate_mm_weights,
                bias_tensor=r.torch_bias,
                eps=r.gate_eps,
                scaling_factor=r.gate_scaling_factor,
                use_hardcoded_expert_index=True,
                hardcoded_expert_index=actual_expert_idx,
                explicit_expert_scale=actual_expert_scale,
                rmsnorm_gamma=r.torch_rmsnorm_gamma,
                rmsnorm_epsilon=1e-6,
                include_residual=(dev_idx == 0),
            )
            expected_final_outputs.append(expected_final)

        expected_reduce_output = sum(expected_final_outputs)
        logger.info(f"[rank=1] expected_reduce_output computed")

        print("intermediate reduce tensors:")
        for idx, tensor in enumerate(reduce_intermediate_tensors):
            print(f"  reduce_intermediate_tensors[{idx}]: {tensor}")
            t = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
            logger.info(f"  tensor[{idx}]: shape={t.shape} nonzero={torch.count_nonzero(t)}")
            logger.info(f"  tensor[{idx}] first 50 values: {t.flatten()[:50]}")
        print("output tensor:")
        out_reduce_tensor = ttnn.to_torch(
            reduce_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        )
        logger.info(
            f"  reduce_output_tensor: shape={out_reduce_tensor.shape} nonzero={torch.count_nonzero(out_reduce_tensor)}"
        )
        logger.info(f"  reduce_output_tensor first 50 values: {out_reduce_tensor.flatten()[:50]}")

        # Only exit-column devices hold the reduced output; non-exit devices
        # are all zeros by design. Compare the golden against each exit-column
        # device's output.
        exit_chip_ids = [row * int(mesh_cols) + exit_column for row in range(int(mesh_rows))]
        logger.info(f"[rank=1] validating output on exit-column devices: chip_ids={exit_chip_ids}")

        all_passing = True
        per_chip_pccs = []
        for chip_id in exit_chip_ids:
            per_chip_output = reduce_output_torch[chip_id].unsqueeze(0)
            nz = int(torch.count_nonzero(per_chip_output).item())
            logger.info(
                f"[rank=1] chip {chip_id} output: shape={tuple(per_chip_output.shape)} nonzero={nz} "
                f"first5={per_chip_output.flatten()[:5].tolist()}"
            )
            if nz == 0:
                logger.error(f"[rank=1] chip {chip_id} (exit column) produced ALL ZEROS")
                all_passing = False
                per_chip_pccs.append((chip_id, 0.0, "all zeros"))
                continue

            reduce_output_valid = extract_routed_expert_output(
                per_chip_output,
                r.num_gate_proj_cores,
                r.final_output_width_per_core,
                r.per_core_down_proj_N,
            )
            passing, pcc_msg = comp_pcc(expected_reduce_output.flatten(), reduce_output_valid.flatten(), 0.97)
            per_chip_pccs.append((chip_id, passing, pcc_msg))
            logger.info(f"[rank=1] chip {chip_id} Reduce PCC: {pcc_msg}")
            if not passing:
                all_passing = False

        logger.info(f"[rank=1] expected first 5 values: {expected_reduce_output.flatten()[:5]}")
        for chip_id, p, msg in per_chip_pccs:
            logger.info(f"[rank=1] summary chip {chip_id}: passing={p} pcc={msg}")
        assert all_passing, f"Pipeline Stage 1 Reduce PCC check failed: {per_chip_pccs}"

    logger.info(f"[rank={my_mesh_id}] test PASSED")


# ---------------------------------------------------------------------------
# Persistent-mode pipeline test
# ---------------------------------------------------------------------------


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
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("vocab_size, embedding_dim", [(64, 7168)])
@pytest.mark.parametrize("token_id", [0])
@pytest.mark.parametrize("iterations", [2])
@pytest.mark.timeout(12000)
def test_persistent_reduce_pipeline_multi_exit_nodes(
    mesh_device, vocab_size, embedding_dim, iterations, token_id, device_params, get_reference_model_state_dict
):
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs < 2:
        pytest.skip(f"Requires at least 2 distributed processes, got {num_procs}")

    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for MoE (need >= 13x10)")

    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline()
    assert len(pipeline_config) == num_procs + 1

    is_torus = device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_2D_TORUS_Y

    is_stage0 = my_mesh_id == 0
    is_stage1 = my_mesh_id == 1

    M = RoutedExpert.M
    K = RoutedExpert.K

    pipeline_core = ttnn.CoreCoord(12, 8)
    core_io = ttnn.CoreCoord(0, 2)
    moe_sender_core = ttnn.CoreCoord(12, 9)
    moe_worker_core_grid = build_worker_grid_excluding_cores(device_grid, [pipeline_core])

    token_size_bytes = 64
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_factor = 1
    embedding_fifo_size = embedding_size_bytes * embedding_fifo_factor

    torch_embedding = torch.arange(vocab_size * K, dtype=torch.float32).reshape(1, 1, vocab_size, K).to(torch.bfloat16)

    reduce_root_coord = ttnn.MeshCoordinate(1, 0)

    # -- Core setup for reduce aggregation --
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(mesh_device, gate_proj_noc)
    num_gate_proj_cores = len(gate_proj_worker_cores)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
    shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
    aggregator_core = shard_cores_list[0]

    mesh_rows, mesh_cols = mesh_device.shape
    device_coords = [ttnn.MeshCoordinate(r, c) for r in range(int(mesh_rows)) for c in range(int(mesh_cols))]

    # Determine exit/entry column from pipeline config.
    exit_column = None
    if len(pipeline_config) > 1:
        entry_node = pipeline_config[0].exit_node_coord
        try:
            exit_column = int(entry_node[1])
        except Exception:
            exit_column = 0
    if exit_column is None:
        exit_column = 0
    pipeline_column_coords = [ttnn.MeshCoordinate(r, exit_column) for r in range(int(mesh_rows))]
    logger.info(f"exit_column={exit_column}, pipeline_column_coords={len(pipeline_column_coords)} devices")

    # -- MoE tensor setup (needed before PipelineBlock to get correct reduce shard size) --
    result_scores = None
    result_indices = None
    result_output = None
    r = None
    s = None
    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=256,
        include_global=False,
    )

    if is_stage0 or is_stage1:
        logger.info(f"[rank={my_mesh_id}] creating routed expert tensors")
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        logger.info(f"[rank={my_mesh_id}] mesh_mapper created")
        needs_device_tensors = is_stage1
        r = create_routed_expert_tensors(
            mesh_device,
            use_hardcoded_expert_index=True,
            mesh_mapper=mesh_mapper,
            state_dict=state_dict,
            is_moe=True,
            layer_idx=ROUTED_EXPERT_LAYER_IDX,
            skip_attention_weights=True,
            create_device_tensors=needs_device_tensors,
        )
        logger.info(f"[rank={my_mesh_id}] routed expert tensors created")
        mcast_grid = moe_worker_core_grid
        s = create_shared_expert_tensors(
            mesh_device,
            M,
            K,
            mcast_grid,
            mesh_mapper=mesh_mapper,
            state_dict=state_dict,
            is_moe=True,
            layer_idx=ROUTED_EXPERT_LAYER_IDX,
            create_device_tensors=needs_device_tensors,
        )
        logger.info(f"[rank={my_mesh_id}] MoE tensors created")

    # Compute per-shard reduce payload from actual shard spec (not from embedding dim)
    if r is not None:
        reduce_payload_per_shard = r.per_core_down_proj_N * 2  # bfloat16
        logger.info(
            f"[rank={my_mesh_id}] reduce_payload_per_shard={reduce_payload_per_shard} "
            f"(per_core_down_proj_N={r.per_core_down_proj_N})"
        )
    else:
        reduce_payload_per_shard = embedding_size_bytes // num_gate_proj_cores
    # Diagnostic: verify socket page sizes match expectations
    if r is not None:
        logger.info(
            f"[rank={my_mesh_id}] DIAG page sizes: "
            f"embedding_size_bytes={embedding_size_bytes} "
            f"reduce_payload_per_shard={reduce_payload_per_shard} "
            f"num_gate_proj_cores(pinned)={num_gate_proj_cores} "
            f"r.num_gate_proj_cores(device)={r.num_gate_proj_cores} "
            f"r.final_output_total_width={r.final_output_total_width} "
            f"r.final_output_width_per_core={r.final_output_width_per_core} "
            f"r.per_core_down_proj_N={r.per_core_down_proj_N} "
            f"len(shard_cores_list)={len(shard_cores_list)} "
            f"total_upstream_bytes={len(shard_cores_list)}*{reduce_payload_per_shard}={len(shard_cores_list)*reduce_payload_per_shard} "
            f"downstream_page_size(=embedding_size_bytes)={embedding_size_bytes}"
        )

    # -- Pipeline block setup (collective -- all hosts must participate simultaneously) --
    pipeline_block = None
    h2d_sockets = None
    d2h_sockets = None
    host_ios = None
    exit_socket_interface = None
    entry_socket_interface = None
    stage0_program_entries = None

    if is_stage0:
        import time as _time

        _t0_total = _time.time()
        print(f"[TEST] stage0: starting setup, my_mesh_id={my_mesh_id} num_procs={num_procs}", flush=True)

        embedding_tensor = ttnn.from_torch(
            torch_embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"[TEST] stage0: embedding on device, elapsed={_time.time()-_t0_total:.3f}s", flush=True)

        # Stages 1-15 call generate_blitz_decode_pipeline inside PipelineBlock.__init__,
        # which is a collective requiring all processes. Stage 0 must participate too.
        print(f"[TEST] stage0: calling generate_blitz_decode_pipeline (collective)...", flush=True)
        ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline()
        print(f"[TEST] stage0: generate_blitz_decode_pipeline done", flush=True)

        d2d_cores = [ttnn.MeshCoreCoord(dc, pipeline_core) for dc in pipeline_column_coords]
        print(f"[TEST] stage0: d2d_cores={[(str(c.device_coord),str(c.core_coord)) for c in d2d_cores]}", flush=True)

        print(f"[TEST] stage0: creating exit_socket_interface (PSI)...", flush=True)
        _t0 = _time.time()
        exit_socket_interface = ParallelSocketInterface(
            embedding_size_bytes,
            embedding_fifo_size,
            send_core_coords=d2d_cores,
            recv_core_coords=d2d_cores,
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=my_mesh_id + 1),
        )
        print(f"[TEST] stage0: exit_socket_interface done in {_time.time()-_t0:.3f}s", flush=True)

        print(f"[TEST] stage0: creating entry_socket_interface (PSI)...", flush=True)
        _t0 = _time.time()
        entry_socket_interface = ParallelSocketInterface(
            embedding_size_bytes,
            embedding_fifo_size,
            send_core_coords=d2d_cores,
            recv_core_coords=d2d_cores,
            sender_mesh=MeshWrapper(mesh_id=num_procs - 1),
            receiver_mesh=MeshWrapper(mesh_device),
            receiver_use_reader_config=True,
        )
        print(f"[TEST] stage0: entry_socket_interface done in {_time.time()-_t0:.3f}s", flush=True)

        print(f"[TEST] stage0: creating H2D/D2H/HostInterface for {len(pipeline_column_coords)} devices...")
        _t0 = _time.time()
        h2d_sockets = []
        d2h_sockets = []
        host_ios = []

        for dc_idx, dc in enumerate(pipeline_column_coords):
            _t_dc = _time.time()
            h2d = ttnn.H2DSocket(
                mesh_device,
                ttnn.MeshCoreCoord(dc, core_io),
                ttnn.BufferType.L1,
                token_size_bytes * 2,
                ttnn.H2DMode.HOST_PUSH,
            )
            d2h = ttnn.D2HSocket(
                mesh_device,
                ttnn.MeshCoreCoord(dc, core_io),
                embedding_fifo_size,
            )
            hio = HostInterface(
                h2d,
                d2h,
                token_size_bytes,
                embedding_size_bytes,
                core_to_core_socket_buffer_size=embedding_fifo_size,
                h2d_downstream_core=ttnn.MeshCoreCoord(dc, pipeline_core),
                d2h_upstream_core=ttnn.MeshCoreCoord(dc, pipeline_core),
                embedding_tensor=embedding_tensor,
            )
            h2d_sockets.append(h2d)
            d2h_sockets.append(d2h)
            host_ios.append(hio)
            print(f"[TEST] stage0:   device[{dc_idx}] dc={dc} done in {_time.time()-_t_dc:.3f}s", flush=True)

        print(f"[TEST] stage0: all H2D/D2H/HostInterface done in {_time.time()-_t0:.3f}s", flush=True)

        print(f"[TEST] stage0: wiring upstream/downstream sockets...", flush=True)
        for i, hio in enumerate(host_ios):
            exit_socket_interface._upstream_sockets[i] = hio.get_downstream_socket()
            entry_socket_interface._downstream_sockets[i] = hio.get_upstream_socket()
        print(f"[TEST] stage0: wiring done", flush=True)

        all_entries = []
        for hio in host_ios:
            all_entries.extend(hio._build_programs())
        exit_progs = exit_socket_interface.build_programs()
        combined_progs = entry_socket_interface.build_programs(base_programs=exit_progs)
        all_entries.extend(combined_progs)
        stage0_program_entries = all_entries
        print(
            f"[TEST] stage0: programs built (dispatch deferred until after tensor alloc), elapsed={_time.time()-_t0_total:.3f}s",
            flush=True,
        )
        logger.info(
            f"[rank=0] parallel stage 0 programs built ({len(pipeline_column_coords)} channels), dispatch deferred"
        )
    elif is_stage1:
        import time as _time

        _t0_total = _time.time()
        print(f"[TEST] stage1: creating PipelineBlock, my_mesh_id={my_mesh_id} num_procs={num_procs}", flush=True)
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
            entry_device_coords=pipeline_column_coords,
            exit_device_coords=pipeline_column_coords,
        )
        print(f"[TEST] stage1: PipelineBlock done in {_time.time()-_t0_total:.3f}s", flush=True)
        logger.info(
            f"[rank=1] DIAG PipelineBlock config: "
            f"upstream_d2d_socket_page_size={embedding_size_bytes} "
            f"downstream_d2d_socket_page_size={embedding_size_bytes} "
            f"exit_upstream_page_size={reduce_payload_per_shard} "
            f"num_exit_upstream_cores={len(shard_cores_list)} "
            f"total_exit_upstream_bytes={len(shard_cores_list)*reduce_payload_per_shard}"
        )
    else:
        import time as _time

        _t0_total = _time.time()
        print(f"[TEST] stage{my_mesh_id}: creating PipelineBlock", flush=True)
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
            pipeline_device_coords=device_coords,
            pipeline_exit_core_coord=pipeline_core,
            entry_device_coords=pipeline_column_coords,
            exit_device_coords=pipeline_column_coords,
        )
        print(f"[TEST] stage{my_mesh_id}: PipelineBlock done in {_time.time()-_t0_total:.3f}s", flush=True)

    logger.info(f"[rank={my_mesh_id}] pipeline block created")

    # -- Get per-device sockets for forward (input) and reduce (output) --
    # Only pipeline column devices have sockets; expand to 8-element lists indexed by chip_id.
    forward_sockets = None
    downstream_sockets = None
    if is_stage1:
        raw_fwd = pipeline_block.get_downstream_sockets()
        raw_ds = pipeline_block.get_upstream_sockets()
        num_devices = int(mesh_rows) * int(mesh_cols)
        forward_sockets = [None] * num_devices
        downstream_sockets = [None] * num_devices
        for idx, row_idx in enumerate(range(int(mesh_rows))):
            chip_id = row_idx * int(mesh_cols) + exit_column
            forward_sockets[chip_id] = raw_fwd[idx]
            downstream_sockets[chip_id] = raw_ds[idx]
        logger.info(
            f"[rank={my_mesh_id}] {len(raw_fwd)} forward sockets mapped to "
            f"{sum(1 for s in forward_sockets if s is not None)}/{num_devices} devices, "
            f"{len(raw_ds)} downstream socket groups"
        )
        for idx, row_idx in enumerate(range(int(mesh_rows))):
            chip_id = row_idx * int(mesh_cols) + exit_column
            ds = downstream_sockets[chip_id]
            if ds is not None:
                if isinstance(ds, list):
                    logger.info(
                        f"[rank=1] DIAG downstream_sockets[chip_id={chip_id}]: "
                        f"{len(ds)} sockets (one per upstream worker)"
                    )
                else:
                    logger.info(f"[rank=1] DIAG downstream_sockets[chip_id={chip_id}]: single socket")
    if is_stage1:
        kv_cache_shard_height = 256
        kvpe_dim = 576
        num_mcast_cores = mcast_grid.num_cores()
        kv_cache_shard_spec = ttnn.ShardSpec(
            mcast_grid, (kv_cache_shard_height, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR
        )
        sdpa_kv_cache_buffer = ttnn.from_torch(
            torch.zeros((kv_cache_shard_height * num_mcast_cores, kvpe_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
            ),
        )

        sdpa_out_interm_shard_height = 48
        sdpa_out_interm_shard_width = 1024
        num_worker_cores = moe_worker_core_grid.num_cores()
        sdpa_out_interm_shard_spec = ttnn.ShardSpec(
            moe_worker_core_grid,
            (sdpa_out_interm_shard_height, sdpa_out_interm_shard_width),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sdpa_out_interm_buffer = ttnn.from_torch(
            torch.zeros(
                (sdpa_out_interm_shard_height * num_worker_cores, sdpa_out_interm_shard_width),
                dtype=torch.bfloat16,
            ),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                sdpa_out_interm_shard_spec,
            ),
            tile=ttnn.Tile([8, 32]),
        )
        logger.info(f"[rank={my_mesh_id}] SDPA buffers created")

        tile_1x32 = ttnn.Tile([1, 32])
        final_output_total_width = r.final_output_total_width

        # -- Reduce-to-all tensors --
        reduce_mesh_mapper_config = ttnn.MeshMapperConfig(
            [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape
        )
        reduce_mesh_mapper = ttnn.create_mesh_mapper(mesh_device, reduce_mesh_mapper_config)

        final_output_mem_config = r.final_output_mem_config
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
        reduce_intermediate_tensors = ttnn.from_torch(
            torch.zeros([4, 2, final_output_total_width * 3], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=intermediate_mem_config,
            tile=tile_1x32,
            mesh_mapper=reduce_mesh_mapper,
        )
        logger.info(f"[rank={my_mesh_id}] reduce intermediate tensors created")

        compute_grid = mesh_device.compute_with_storage_grid_size()
        reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(aggregator_core, aggregator_core)})
        reduce_output_shard_spec = ttnn.ShardSpec(
            reduce_output_shard_grid,
            (1, final_output_total_width),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        reduce_output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, reduce_output_shard_spec
        )
        reduce_output_data = torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16)
        reduce_output_tensor = ttnn.from_torch(
            reduce_output_data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=reduce_output_mem_config,
            tile=tile_1x32,
            mesh_mapper=reduce_mesh_mapper,
        )
        logger.info(f"[rank={my_mesh_id}] reduce output tensor created")

        num_cores = compute_grid.x * compute_grid.y
        reduce_available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
        reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, reduce_available_cores, 0) for _ in range(4)]

        moe_semaphores = MoeOp.create_semaphores(mesh_device)
        logger.info(f"[rank={my_mesh_id}] semaphores created")

        # Allocate the dedicated cb46 forward staging buffer NOW (before
        # pipeline_block.run launches persistent kernels). Otherwise the
        # host-to-device upload performed by ttnn.from_torch deadlocks
        # against the in-flight dispatch queue.
        moe_K_single = r.ttnn_residual_mcast_src.shape[-1]
        moe_forward_staging_tensor_single = MoeOp.create_forward_staging_tensor(
            mesh_device,
            moe_sender_core,
            moe_K_single,
            dtype=r.ttnn_residual_mcast_src.dtype,
        )
        logger.info(
            f"[rank={my_mesh_id}] forward_staging_tensor allocated "
            f"(sender_core={moe_sender_core}, K={moe_K_single})"
        )

    # -- Launch pipeline programs --
    if is_stage0:
        _dispatch_merged_programs(stage0_program_entries, mesh_device)
        logger.info(f"[rank=0] stage0 programs dispatched (deferred)")
    else:
        pipeline_block.run()
    logger.info(f"[rank={my_mesh_id}] pipeline programs launched")

    logger.info(f"[rank={my_mesh_id}] waiting at barrier after pipeline launch")
    ttnn.distributed_context_barrier()
    logger.info(f"[rank={my_mesh_id}] passed barrier after pipeline launch")

    if is_stage1:
        logger.info(f"[rank=1] launching MoE forward + reduce-to-all (num_iterations=1)")
        result_scores, result_indices, result_output = MoeOp.op(
            r.ttnn_residual_mcast_src,
            gate_mm_weights_tensor=r.ttnn_gate_mm_weights,
            gate_bias_tensor=r.ttnn_gate_bias,
            gate_indices_tensor=r.ttnn_gate_indices,
            gate_output_scores_tensor=r.gate_output_scores_tensor,
            gate_output_indices_tensor=r.gate_output_indices_tensor,
            gate_proj_weights_tensor=r.gate_proj_weights,
            up_proj_weights_tensor=r.up_proj_weights,
            down_proj_weights_tensor=r.down_proj_weights,
            final_output_tensor=r.final_output_tensor,
            rmsnorm_gamma_tensor=r.ttnn_rmsnorm_gamma,
            shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
            shared_up_weights_overlapped=s.shared_up_weights_overlapped,
            shared_down_weights_tensor=s.ttnn_down_weights,
            shared_k_parallel=s.k_parallel,
            shared_n_parallel=s.n_parallel,
            use_hardcoded_expert_index=True,
            sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer=sdpa_out_interm_buffer,
            num_iterations=1,
            persistent_mode=True,
            reduce_intermediate_tensors=reduce_intermediate_tensors,
            reduce_output_tensor=reduce_output_tensor,
            reduce_semaphores=reduce_semaphores,
            reduce_root_coord=reduce_root_coord,
            forward_sockets=forward_sockets,
            forward_staging_tensor=moe_forward_staging_tensor_single,
            semaphores=moe_semaphores,
            worker_core_grid=moe_worker_core_grid,
            is_torus=is_torus,
            downstream_sockets=downstream_sockets,
            exit_column=exit_column,
        )
        logger.info("[rank=1] MoE + reduce-to-all completed")

    # -- Stage 0: D2H loopback read + golden validation --
    d2h_passing = True
    d2h_pcc_msg = ""
    if is_stage0:
        logger.info("[rank=0] waiting for D2H result from pipeline loopback")
        num_elements = embedding_size_bytes // 2

        from models.demos.deepseek_v3_b1.micro_ops.deepseek_moe_gate.op import DeepseekMoeGateSingleCore

        K_down = s.K_down
        torch_input_row = torch_embedding[0, 0, token_id : token_id + 1, :]

        x_raw = torch_input_row.float()
        variance = x_raw.pow(2).mean(-1, keepdim=True)
        normalized_input = ((x_raw * torch.rsqrt(variance + 1e-6)) * r.torch_rmsnorm_gamma.float()).bfloat16().float()
        logits = normalized_input.float() @ r.torch_gate_mm_weights.float()
        scores = torch.sigmoid(logits)
        gate_input = scores.reshape(1, 8, 32)
        cpu_gate_scores, _ = DeepseekMoeGateSingleCore.golden(
            gate_input, r.torch_bias.float(), r.gate_eps, r.gate_scaling_factor, enable_sigmoid=False
        )

        expected_final_outputs = []
        for dev_idx in range(mesh_rows * mesh_cols):
            actual_expert_scale = cpu_gate_scores.flatten()[dev_idx].float()
            shared_gate_shard = s.torch_gate_weights[:, dev_idx * K_down : (dev_idx + 1) * K_down]
            shared_up_shard = s.torch_up_weights[:, dev_idx * K_down : (dev_idx + 1) * K_down]
            shared_down_shard = s.torch_down_weights[dev_idx * K_down : (dev_idx + 1) * K_down, :]

            _, _, expected_final = MoeOp.golden_single_device(
                torch_input_row,
                shared_gate_weights=shared_gate_shard,
                shared_up_weights=shared_up_shard,
                shared_down_weights=shared_down_shard,
                gate_proj_weights_dict=r.expert_weights_dict,
                up_proj_weights_dict=r.up_proj_weights_dict,
                down_proj_weights_dict=r.down_proj_weights_dict,
                routing_weights_tensor=r.torch_gate_mm_weights,
                bias_tensor=r.torch_bias,
                eps=r.gate_eps,
                scaling_factor=r.gate_scaling_factor,
                use_hardcoded_expert_index=True,
                hardcoded_expert_index=dev_idx,
                explicit_expert_scale=actual_expert_scale,
                rmsnorm_gamma=r.torch_rmsnorm_gamma,
                rmsnorm_epsilon=1e-6,
                include_residual=(dev_idx == 0),
            )
            expected_final_outputs.append(expected_final)

        expected_reduce_output = sum(expected_final_outputs)

        for iteration in range(iterations):
            print(f"[rank={my_mesh_id}] iteration {iteration} start")
            logger.info(f"[rank=0] launching token injection")
            token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
            torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
            torch_token[0, 0] = token_id
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            for h2d in h2d_sockets:
                h2d.write_tensor(token_tensor)
            logger.info(f"[rank=0] token {token_id} injected to {len(h2d_sockets)} channels")

            # Read ALL D2H sockets to find which channel(s) carry valid data
            d2h_results = []
            for sock_idx, d2h_sock in enumerate(d2h_sockets):
                buf = torch.zeros(1, num_elements, dtype=torch.bfloat16)
                buf_tensor = ttnn.from_torch(buf, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                d2h_sock.read_tensor(buf_tensor)
                result = ttnn.to_torch(buf_tensor)
                nz = torch.count_nonzero(result)
                logger.info(
                    f"[rank=0] D2H socket[{sock_idx}]: non-zero={nz}/{result.numel()} " f"first5={result[0, :5]}"
                )
                d2h_results.append(result)

            # Pick the first socket with non-zero data for validation
            d2h_result_torch = None
            for sock_idx, result in enumerate(d2h_results):
                if torch.count_nonzero(result) > 0:
                    d2h_result_torch = result
                    logger.info(f"[rank=0] using D2H socket[{sock_idx}] for golden comparison")
                    break
            assert d2h_result_torch is not None, "All D2H sockets returned zeros -- reduce or pipeline failed"

            reduce_shard_width = reduce_payload_per_shard // dtype_size(ttnn.bfloat16)
            d2h_valid = extract_routed_expert_output(
                d2h_result_torch, r.num_gate_proj_cores, reduce_shard_width, r.per_core_down_proj_N
            )

            passing, pcc_msg = comp_pcc(expected_reduce_output.flatten(), d2h_valid.flatten(), 0.9)
            logger.info(f"Pipeline Stage 0 D2H Reduce PCC: {pcc_msg}")
            assert passing, f"Pipeline Stage 0 D2H PCC check failed: {pcc_msg}"

    ttnn.distributed_context_barrier()

    # -- Pipeline teardown --
    logger.info(f"[rank={my_mesh_id}] waiting for pipeline termination")
    if is_stage0:
        for hio in host_ios:
            hio.terminate(False)
        entry_socket_interface.terminate(False)
        exit_socket_interface.terminate(True)
    else:
        pipeline_block.terminate()
    logger.info(f"[rank={my_mesh_id}] programs terminated")
