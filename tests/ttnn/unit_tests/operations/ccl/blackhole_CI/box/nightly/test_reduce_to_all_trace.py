# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from tracy import signpost
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


def compute_reduction(l1, s1, m1, l2, s2, m2, scale_value, l_width=128):
    """
    Compute the online softmax reduction of two partial results.
    Returns (l_new, s_new, m_new)
    """
    m_new = torch.maximum(m1, m2)
    exp_m1 = torch.exp((m1 - m_new) * scale_value)[:, :1].expand(-1, l_width)
    exp_m2 = torch.exp((m2 - m_new) * scale_value)[:, :1].expand(-1, l_width)
    s_new = s1 * torch.exp((m1 - m_new) * scale_value) + s2 * torch.exp((m2 - m_new) * scale_value)
    l_new = l1 * exp_m1 + l2 * exp_m2
    return l_new, s_new, m_new


def compute_reference_reduce_to_all(
    l_data_per_device, s_data_per_device, m_data_per_device, root_device_idx=1, num_cores=8, scale_value=1.0
):
    """
    Compute the reference output for reduce_to_all operation.

    Algorithm:
    Round 1: Neighbor exchange D0<->D1 and D2<->D3
      - All 4 devices compute their local partial reduction
      - D0 and D1 both compute reduction(D0, D1)
      - D2 and D3 both compute reduction(D2, D3)

    Round 2: Neighbor exchange D0<->D3 and D1<->D2
      - All 4 devices exchange their Round 1 results with the other pair
      - All 4 devices compute the final reduction(D0, D1, D2, D3)
    """
    num_devices = len(l_data_per_device)

    def split_by_cores(tensor_list, num_cores):
        result = []
        for device_tensor in tensor_list:
            cores = torch.chunk(device_tensor, num_cores, dim=1)
            result.append(cores)
        return result

    l_per_device_per_core = split_by_cores(l_data_per_device, num_cores)
    s_per_device_per_core = split_by_cores(s_data_per_device, num_cores)
    m_per_device_per_core = split_by_cores(m_data_per_device, num_cores)

    l_final_cores = []
    s_final_cores = []
    m_final_cores = []

    for core_idx in range(num_cores):
        l_dev = [l_per_device_per_core[d][core_idx] for d in range(num_devices)]
        s_dev = [s_per_device_per_core[d][core_idx] for d in range(num_devices)]
        m_dev = [m_per_device_per_core[d][core_idx] for d in range(num_devices)]

        # Round 1: D0<->D1 and D2<->D3 exchanges
        # D0 and D1 both compute reduction(D0, D1)
        l_r1_01, s_r1_01, m_r1_01 = compute_reduction(
            l_dev[0], s_dev[0], m_dev[0], l_dev[1], s_dev[1], m_dev[1], scale_value
        )

        # D2 and D3 both compute reduction(D2, D3)
        l_r1_23, s_r1_23, m_r1_23 = compute_reduction(
            l_dev[2], s_dev[2], m_dev[2], l_dev[3], s_dev[3], m_dev[3], scale_value
        )

        # Round 2: D0<->D3 and D1<->D2 exchanges
        # All devices compute final reduction of (D0+D1) with (D2+D3)
        l_final, s_final, m_final = compute_reduction(l_r1_01, s_r1_01, m_r1_01, l_r1_23, s_r1_23, m_r1_23, scale_value)

        # Final division: l_out = l_final / s_final
        l_out = l_final / s_final[:, :1].expand(-1, l_final.shape[1])

        l_final_cores.append(l_out)
        s_final_cores.append(s_final)
        m_final_cores.append(m_final)

    return torch.cat(l_final_cores, dim=1), torch.cat(s_final_cores, dim=1), torch.cat(m_final_cores, dim=1)


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 548880}),
    ],
    indirect=["device_params"],
    ids=["fabric_1d_ring_trace"],
)
@pytest.mark.parametrize("use_barrier", [False, True], ids=["no_barrier", "with_barrier"])
def test_reduce_to_all_with_trace(bh_1d_mesh_device, use_barrier):
    """Test reduce_to_all operation with trace capture and replay for performance testing."""

    print("\n=== Testing reduce_to_all with TRACE ===")

    # Setup
    num_devices = 4
    root_coord = (1, 0)
    root_device_idx = root_coord[0]
    num_cores = 8
    l_width = 128
    s_m_width = 32

    batch_size = 8
    l_shape = [batch_size, l_width * num_cores]
    s_shape = [batch_size, s_m_width * num_cores]
    m_shape = [batch_size, s_m_width * num_cores]
    intermediate_shape = [batch_size, 192 * num_cores]

    scale_value = 1.0
    topology = ttnn.Topology.Ring

    # Create submesh device
    validate_test(num_devices, topology, bh_1d_mesh_device.shape, 0)
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    # Tensor config
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((8, 32))

    # mux cores (aggregator uses only 2 cores - 1 per link)
    mux_cores = [ttnn.CoreCoord(6, 8), ttnn.CoreCoord(6, 9)]

    # results in better perf compared to automating the generation of worker cores (other than data cores)
    extra_worker_cores = [
        ttnn.CoreCoord(0, 4),
        ttnn.CoreCoord(0, 5),
        ttnn.CoreCoord(0, 6),
        ttnn.CoreCoord(0, 7),
        ttnn.CoreCoord(1, 4),
        ttnn.CoreCoord(1, 5),
        ttnn.CoreCoord(1, 6),
        ttnn.CoreCoord(1, 7),
    ]

    # Shard config
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(2, 8), ttnn.CoreCoord(5, 8)),
            ttnn.CoreRange(ttnn.CoreCoord(2, 9), ttnn.CoreCoord(5, 9)),
        }
    )
    shard_spec_l = ttnn.ShardSpec(shard_grid, [8, 128], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_s = ttnn.ShardSpec(shard_grid, [8, 32], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_int = ttnn.ShardSpec(shard_grid, [8, 192], ttnn.ShardOrientation.ROW_MAJOR)

    mem_config_l = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_l
    )
    mem_config_s = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_s
    )
    mem_config_int = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_int
    )

    mesh_mapper_config = ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    mesh_mapper_config2 = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper2 = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config2)

    # Create random input tensors
    torch.manual_seed(42)
    l_data_per_device = [torch.randn(l_shape, dtype=torch.float32).to(torch.bfloat16) for _ in range(num_devices)]
    s_data_per_device = [torch.rand(s_shape, dtype=torch.float32).to(torch.bfloat16) + 0.5 for _ in range(num_devices)]
    m_data_per_device = [torch.randn(m_shape, dtype=torch.float32).to(torch.bfloat16) for _ in range(num_devices)]

    # Compute reference (convert to float32 for accuracy)
    l_data_f32 = [t.float() for t in l_data_per_device]
    s_data_f32 = [t.float() for t in s_data_per_device]
    m_data_f32 = [t.float() for t in m_data_per_device]

    ref_l, ref_s, ref_m = compute_reference_reduce_to_all(
        l_data_f32, s_data_f32, m_data_f32, root_device_idx, num_cores, scale_value
    )
    ref_l = ref_l.to(torch.bfloat16)

    # Stack data for mesh tensor
    l_data_all = torch.stack(l_data_per_device, dim=0)
    s_data_all = torch.stack(s_data_per_device, dim=0)
    m_data_all = torch.stack(m_data_per_device, dim=0)

    # Create mesh tensors
    l_tensor = ttnn.from_torch(
        l_data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_l,
        mesh_mapper=mesh_mapper,
    )
    s_tensor = ttnn.from_torch(
        s_data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_s,
        mesh_mapper=mesh_mapper,
    )
    m_tensor = ttnn.from_torch(
        m_data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_s,
        mesh_mapper=mesh_mapper,
    )

    # Create intermediate tensors
    fw_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )
    bw_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )
    coord_intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )

    # Create aggregator scratch tensor for aggregator cores
    # The aggregator needs 8 slots per core (4 FWD + 4 BWD), each slot = header + L + S + M packed
    # Slot layout: [header (256B reserved)] [L payload] [S payload] [M payload]
    # - Header: 256B reserved (actual header is smaller, but generous padding avoids overflow)
    # - L: 4 tiles * 256B = 1024B (128 elements width / 32 tile_width = 4 tiles)
    # - S: 1 tile * 256B = 256B
    # - M: 1 tile * 256B = 256B
    # Total per slot: 256 + 1024 + 256 + 256 = 1792B
    # 8 slots * 1792B = 14336B per core
    #
    # In bfloat16 elements per core:
    # - Payload: 192 * 4 = 768 columns (L+S+M per slot * 4 slots per round direction * 2 rounds)
    # - Header padding: 256B * 8 slots / 2 bytes = 1024 elements = 128 columns (at 8 rows)
    # Total shard width: 768 + 128 = 896 columns
    aggregator_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(mux_cores[0], mux_cores[1])})
    aggregator_shard_spec = ttnn.ShardSpec(aggregator_shard_grid, [8, 192 * 4 + 128], ttnn.ShardOrientation.ROW_MAJOR)
    aggregator_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, aggregator_shard_spec
    )
    aggregator_scratch_shape = [8, (192 * 4 + 128) * 2]  # shard_width * 2 cores
    aggregator_scratch_tensor = ttnn.from_torch(
        torch.zeros(aggregator_scratch_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=aggregator_mem_config,
        mesh_mapper=mesh_mapper2,
    )

    # Create barrier resources for device synchronization (only used when use_barrier=True)
    # Uses all_gather_async which requires global semaphores but NOT SubDevice setup.
    # We need 1 set of 2 semaphores - all_gather is inherently synchronizing so no double-buffering needed.
    barrier_tensor = None
    barrier_semaphores = None
    if use_barrier:
        # Create global semaphores for all_gather_async (2 required per the implementation)
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        submesh_device.compute_with_storage_grid_size().x - 1,
                        submesh_device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            }
        )
        barrier_semaphores = [ttnn.create_global_semaphore(submesh_device, ccl_sub_device_crs, 0) for _ in range(2)]

        # Barrier tensor: 1 full tile per device (32x32 = 1024 elements minimum)
        # Shape: [1, 1, 32, 32] per device -> gathered on dim=0
        barrier_shape = [1, 1, 32, 32]  # Minimal: 1 tile (must be 32x32 for TILE_LAYOUT)
        barrier_tensor = ttnn.from_torch(
            torch.zeros(barrier_shape, dtype=torch.bfloat16),
            device=submesh_device,
            layout=layout,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Use DRAM to avoid L1 conflicts
            mesh_mapper=mesh_mapper2,  # Replicated on all devices
        )
        print(f"Barrier created with all_gather_async (2 semaphores)")

    profiler = BenchmarkProfiler()

    # Run once to compile
    print("Running reduce_to_all (compiling)...")
    ttnn.reduce_to_all(
        l_tensor,
        s_tensor,
        m_tensor,
        root_coord=ttnn.MeshCoordinate(root_coord),
        scale_fp32=scale_value,
        fw_intermediate_tensor=fw_intermediate,
        bw_intermediate_tensor=bw_intermediate,
        coord_intermediate_tensor=coord_intermediate,
        topology=topology,
        input_mux_cores=mux_cores,
        extra_worker_cores=extra_worker_cores,
        aggregator_scratch_tensor=aggregator_scratch_tensor,
    )
    # Also compile barrier if used
    if use_barrier:
        _ = ttnn.experimental.all_gather_async(
            barrier_tensor,
            dim=0,
            multi_device_global_semaphore=barrier_semaphores,
            topology=topology,
        )
    ttnn.synchronize_device(submesh_device)

    # Warmup iterations with trace
    logger.info("Capturing warmup trace")
    print("Warmup iterations...")
    num_warmup_iters = 15
    trace_id_warmup = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    for i in range(num_warmup_iters):
        # Optional barrier: force all devices to sync BEFORE reduce_to_all
        # This ensures all devices start the op at the same time, making
        # zone measurements reflect actual latency, not skew from previous iteration
        if use_barrier:
            _ = ttnn.experimental.all_gather_async(
                barrier_tensor,
                dim=0,
                multi_device_global_semaphore=barrier_semaphores,
                topology=topology,
            )
        out_l_trace, out_s_trace, out_m_trace = ttnn.reduce_to_all(
            l_tensor,
            s_tensor,
            m_tensor,
            root_coord=ttnn.MeshCoordinate(root_coord),
            scale_fp32=scale_value,
            fw_intermediate_tensor=fw_intermediate,
            bw_intermediate_tensor=bw_intermediate,
            coord_intermediate_tensor=coord_intermediate,
            topology=topology,
            input_mux_cores=mux_cores,
            extra_worker_cores=extra_worker_cores,
            aggregator_scratch_tensor=aggregator_scratch_tensor,
        )
        if use_barrier:
            _ = ttnn.experimental.all_gather_async(
                barrier_tensor,
                dim=0,
                multi_device_global_semaphore=barrier_semaphores,
                topology=topology,
            )
    ttnn.end_trace_capture(submesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh_device)

    # Capture main trace for perf measurement
    logger.info("Capturing main trace")
    print("Capturing trace for perf measurement...")
    num_perf_iters = 50
    trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    for i in range(num_perf_iters):
        # Optional barrier: force all devices to sync BEFORE reduce_to_all
        if use_barrier:
            _ = ttnn.experimental.all_gather_async(
                barrier_tensor,
                dim=0,
                multi_device_global_semaphore=barrier_semaphores,
                topology=topology,
            )
        out_l_trace, out_s_trace, out_m_trace = ttnn.reduce_to_all(
            l_tensor,
            s_tensor,
            m_tensor,
            root_coord=ttnn.MeshCoordinate(root_coord),
            scale_fp32=scale_value,
            fw_intermediate_tensor=fw_intermediate,
            bw_intermediate_tensor=bw_intermediate,
            coord_intermediate_tensor=coord_intermediate,
            topology=topology,
            input_mux_cores=mux_cores,
            extra_worker_cores=extra_worker_cores,
            aggregator_scratch_tensor=aggregator_scratch_tensor,
        )
        if use_barrier:
            _ = ttnn.experimental.all_gather_async(
                barrier_tensor,
                dim=0,
                multi_device_global_semaphore=barrier_semaphores,
                topology=topology,
            )
    ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh_device)

    logger.info("Starting Trace perf test...")

    # Execute warmup trace
    profiler.start("reduce-to-root-warmup")
    ttnn.execute_trace(submesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(submesh_device, trace_id_warmup)
    ttnn.synchronize_device(submesh_device)
    profiler.end("reduce-to-root-warmup")

    # Execute main trace with signposting for tracy
    signpost("start")
    profiler.start("reduce-to-root-trace")

    ttnn.execute_trace(submesh_device, trace_id, blocking=False)
    ttnn.release_trace(submesh_device, trace_id)
    ttnn.synchronize_device(submesh_device)

    profiler.end("reduce-to-root-trace")
    signpost("stop")

    # Verify the output from the last trace execution
    print("\nVerifying trace output...")
    output_l_torch = ttnn.to_torch(out_l_trace, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))

    # Get root device output
    out_l_root = output_l_torch[root_device_idx]

    # Compare with reference
    out_flat = out_l_root.flatten().float()
    ref_flat = ref_l.flatten().float()
    max_diff = torch.max(torch.abs(out_flat - ref_flat)).item()
    match = max_diff < 0.07  # Allow tolerance for bfloat16

    print(f"L tensor match: {match}, max_diff: {max_diff:.4f}")

    if match:
        print("\n=== TRACE TEST PASSED ===")
    else:
        print("\n=== TRACE TEST FAILED ===")
        # Find location of max diff
        diff_tensor = torch.abs(out_flat - ref_flat)
        max_idx = torch.argmax(diff_tensor).item()
        print(f"Max diff at index={max_idx}")
        print(f"Output value: {out_flat[max_idx].item():.4f}")
        print(f"Reference value: {ref_flat[max_idx].item():.4f}")

    assert match, f"L tensor mismatch! Max diff: {max_diff}"
