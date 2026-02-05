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


def compute_reduction(l1, s1, m1, l2, s2, m2, scale_value):
    """
    Compute the online softmax reduction of two partial results.
    Returns (l_new, s_new, m_new)

    Note: m1, m2, s1, s2 should be 1-column tensors [height, 1] containing the
    max and sum values. These are broadcast across L width.
    """
    m_new = torch.maximum(m1, m2)
    exp_m1 = torch.exp((m1 - m_new) * scale_value)
    exp_m2 = torch.exp((m2 - m_new) * scale_value)
    s_new = s1 * exp_m1 + s2 * exp_m2
    l_new = l1 * exp_m1 + l2 * exp_m2
    return l_new, s_new, m_new


def compute_reference_reduce_to_all(
    l_data_per_device, s_data_per_device, m_data_per_device, num_cores=8, scale_value=1.0
):
    """
    Compute the reference output for reduce_to_all operation.

    Args:
        l_data_per_device: List of L tensors per device, shape [batch, l_width * num_cores]
        s_data_per_device: List of S tensors per device, shape [batch, num_cores] (one value per core)
        m_data_per_device: List of M tensors per device, shape [batch, num_cores] (one value per core)

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
        """Split tensors into per-core chunks"""
        result = []
        for device_tensor in tensor_list:
            cores = torch.chunk(device_tensor, num_cores, dim=1)
            result.append(cores)
        return result

    # Split L into per-core chunks of shape [batch, l_width]
    l_per_device_per_core = split_by_cores(l_data_per_device, num_cores)

    # m and s are already [batch, num_cores], extract column slices as [batch, 1]
    # These are broadcast across l_width during reduction
    l_final_cores = []
    s_final_cores = []
    m_final_cores = []

    for core_idx in range(num_cores):
        l_dev = [l_per_device_per_core[d][core_idx] for d in range(num_devices)]
        # Extract single column for m and s, shape [batch, 1]
        s_dev = [s_data_per_device[d][:, core_idx : core_idx + 1] for d in range(num_devices)]
        m_dev = [m_data_per_device[d][:, core_idx : core_idx + 1] for d in range(num_devices)]

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

        # Final division: l_out = l_final / s_final (s_final is [batch, 1], broadcast)
        l_out = l_final / s_final.expand(-1, l_final.shape[1])

        l_final_cores.append(l_out)
        s_final_cores.append(s_final)
        m_final_cores.append(m_final)

    return torch.cat(l_final_cores, dim=1), torch.cat(s_final_cores, dim=1), torch.cat(m_final_cores, dim=1)


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def compute_forwarder_scratch_size(
    l_width: int,
    num_cores: int,
    tile_height: int = 8,
    tile_width: int = 32,
    bytes_per_element: int = 2,
) -> int:
    """
    Compute the required forwarder scratch buffer size in bytes.

    The forwarder needs slots for packets sent by workers:
    - Each worker sends: 1 MS packet + num_l_chunks L chunk packets
    - For both R1 and R2 rounds
    - For both BRISC (FWD direction) and NCRISC (BWD direction)

    Args:
        l_width: Width of L tensor per core in elements
        num_cores: Number of worker cores
        tile_height: Tile height (default 8)
        tile_width: Tile width (default 32)
        bytes_per_element: Bytes per element (default 2 for bfloat16)

    Returns:
        Total buffer size in bytes (conservative estimate)
    """
    # Hardware constraint: max tiles per SDPA block (DST register limit)
    MAX_TILES_PER_CHUNK = 8

    # Conservative header estimate - no API to query actual size (~32 bytes)
    # Using 256 bytes to be safe
    PACKET_HEADER_SIZE_BYTES_ESTIMATE = 256

    # L1 alignment requirement
    L1_ALIGNMENT = 16

    # Calculate tiles per core for L data
    tiles_per_core_l = l_width // tile_width

    # Conservative chunk estimate: use hardware max constraint
    # Actual chunking determined by program factory, we use worst-case
    num_l_chunks = (tiles_per_core_l + MAX_TILES_PER_CHUNK - 1) // MAX_TILES_PER_CHUNK

    # Slots per round: each worker sends MS (1 slot) + L chunks
    slots_per_worker = 1 + num_l_chunks
    slots_per_round = num_cores * slots_per_worker

    # Slot size: header + largest payload (L chunk)
    # L chunk is larger than MS, so use L chunk size
    tiles_per_chunk = (tiles_per_core_l + num_l_chunks - 1) // num_l_chunks  # ceil
    l_chunk_size_bytes = tiles_per_chunk * tile_height * tile_width * bytes_per_element
    slot_size_bytes = PACKET_HEADER_SIZE_BYTES_ESTIMATE + l_chunk_size_bytes

    # Align slot size to L1 alignment
    slot_size_bytes = ((slot_size_bytes + L1_ALIGNMENT - 1) // L1_ALIGNMENT) * L1_ALIGNMENT

    # Total buffer: R1 + R2 regions for both BRISC and NCRISC
    # Layout: [BRISC R1][BRISC R2][NCRISC R1][NCRISC R2]
    forwarder_buffer_size_bytes = 2 * slots_per_round * slot_size_bytes * 2

    return forwarder_buffer_size_bytes


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "trace_region_size": 548880,
                # "fabric_router_config": create_fabric_router_config(9600),
            }
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_2d_trace"],
)
def test_reduce_to_all_with_trace(bh_1d_mesh_device):
    """Test reduce_to_all operation with trace capture and replay for performance testing."""

    print("\n=== Testing reduce_to_all with TRACE ===")

    # Setup
    num_devices = 4
    num_cores = 8
    l_width = 512
    ms_width = 32  # Combined MS tile: col 0 = max, col 1 = sum

    batch_size = 8
    l_shape = [batch_size, l_width * num_cores]
    ms_shape = [batch_size, ms_width * num_cores]  # Combined MS shape
    intermediate_shape = [batch_size, (l_width + ms_width) * num_cores]

    scale_value = 1.0
    topology = ttnn.Topology.Torus

    # Create submesh device
    # Mesh topology parameter uses 2D fabric routing (configured by FABRIC_2D)
    # but logical device arrangement can be 1D [4,1] for reduce_to_all's neighbor exchange pattern
    validate_test(num_devices, topology, bh_1d_mesh_device.shape, 0)
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    # Tensor config
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((8, 32))

    # forwarder uses only 2 cores - 1 per link
    forwarder_cores = [ttnn.CoreCoord(6, 8), ttnn.CoreCoord(6, 9)]

    # Shard config
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(2, 8), ttnn.CoreCoord(5, 8)),
            ttnn.CoreRange(ttnn.CoreCoord(2, 9), ttnn.CoreCoord(5, 9)),
        }
    )
    shard_spec_l = ttnn.ShardSpec(shard_grid, [batch_size, l_width], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_ms = ttnn.ShardSpec(shard_grid, [batch_size, ms_width], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_int = ttnn.ShardSpec(
        shard_grid, [batch_size, (l_width + ms_width)], ttnn.ShardOrientation.ROW_MAJOR
    )  # L + MS

    mem_config_l = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_l
    )
    mem_config_ms = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_ms
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
    ms_data_per_device = [torch.randn(ms_shape, dtype=torch.float32).to(torch.bfloat16) for _ in range(num_devices)]

    m_data_per_device = []
    s_data_per_device = []

    for d in range(num_devices):
        ms_device = ms_data_per_device[d]
        m_device = torch.zeros((ms_shape[0], num_cores), dtype=torch.bfloat16)
        s_device = torch.zeros((ms_shape[0], num_cores), dtype=torch.bfloat16)
        for core_idx in range(num_cores):
            # Set some meaningful values for m and s
            m_device[:, core_idx] = torch.randn(ms_shape[0], dtype=torch.bfloat16) * 0.5 - 1.0  # max values
            s_device[:, core_idx] = (
                torch.abs(torch.randn(ms_shape[0], dtype=torch.bfloat16)) + 0.1
            )  # sum values (positive)

            # Overwrite the random MS data with our generated m and s values
            ms_device[:, core_idx * ms_width + 0] = m_device[:, core_idx]  # max
            ms_device[:, core_idx * ms_width + 1] = s_device[:, core_idx]  # sum
        m_data_per_device.append(m_device)
        s_data_per_device.append(s_device)

    # Compute reference (convert to float32 for accuracy)
    l_data_f32 = [t.float() for t in l_data_per_device]
    s_data_f32 = [t.float() for t in s_data_per_device]
    m_data_f32 = [t.float() for t in m_data_per_device]

    ref_l, ref_s, ref_m = compute_reference_reduce_to_all(l_data_f32, s_data_f32, m_data_f32, num_cores, scale_value)
    ref_l = ref_l.to(torch.bfloat16)

    # Stack data for mesh tensor
    l_data_all = torch.stack(l_data_per_device, dim=0)
    ms_data_all = torch.stack(ms_data_per_device, dim=0)

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
    ms_tensor = ttnn.from_torch(
        ms_data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_ms,
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

    # Create forwarder scratch tensor for forwarder cores
    # Uses helper function to compute conservative buffer size
    tile_height, tile_width = 8, 32
    bytes_per_element = 2  # bfloat16

    forwarder_buffer_size_bytes = compute_forwarder_scratch_size(
        l_width=l_width,
        num_cores=num_cores,
        tile_height=tile_height,
        tile_width=tile_width,
        bytes_per_element=bytes_per_element,
    )

    # Convert to tensor dimensions
    # Buffer is split across 2 forwarder cores (BRISC region on each)
    # Shard shape is [tile_height, width], so: shard_bytes = tile_height * width * bytes_per_element
    num_forwarder_cores = 2
    forwarder_shard_bytes = forwarder_buffer_size_bytes // num_forwarder_cores
    forwarder_shard_width_elements = forwarder_shard_bytes // (tile_height * bytes_per_element)

    forwarder_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(forwarder_cores[0], forwarder_cores[1])})
    forwarder_shard_spec = ttnn.ShardSpec(
        forwarder_shard_grid, [tile_height, forwarder_shard_width_elements], ttnn.ShardOrientation.ROW_MAJOR
    )
    forwarder_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, forwarder_shard_spec
    )
    forwarder_scratch_shape = [tile_height, forwarder_shard_width_elements * num_forwarder_cores]
    forwarder_scratch_tensor = ttnn.from_torch(
        torch.zeros(forwarder_scratch_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=forwarder_mem_config,
        mesh_mapper=mesh_mapper2,
    )

    profiler = BenchmarkProfiler()

    # Run once to compile
    print("Running reduce_to_all (compiling)...")
    out_l_compile = ttnn.reduce_to_all(
        l_tensor,
        ms_tensor,
        scale_fp32=scale_value,
        fw_intermediate_tensor=fw_intermediate,
        bw_intermediate_tensor=bw_intermediate,
        topology=topology,
        input_forwarder_cores=forwarder_cores,
        forwarder_scratch_tensor=forwarder_scratch_tensor,
    )
    ttnn.synchronize_device(submesh_device)

    # Warmup iterations with trace
    logger.info("Capturing warmup trace")
    num_warmup_iters = 15
    trace_id_warmup = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    for i in range(num_warmup_iters):
        out_l_trace = ttnn.reduce_to_all(
            l_tensor,
            ms_tensor,
            scale_fp32=scale_value,
            fw_intermediate_tensor=fw_intermediate,
            bw_intermediate_tensor=bw_intermediate,
            topology=topology,
            input_forwarder_cores=forwarder_cores,
            forwarder_scratch_tensor=forwarder_scratch_tensor,
        )
    ttnn.end_trace_capture(submesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh_device)

    # Capture main trace for perf measurement
    logger.info("Capturing main trace")
    num_perf_iters = 50
    trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    for i in range(num_perf_iters):
        out_l_trace = ttnn.reduce_to_all(
            l_tensor,
            ms_tensor,
            scale_fp32=scale_value,
            fw_intermediate_tensor=fw_intermediate,
            bw_intermediate_tensor=bw_intermediate,
            topology=topology,
            input_forwarder_cores=forwarder_cores,
            forwarder_scratch_tensor=forwarder_scratch_tensor,
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

    logger.info("Warmup trace execution completed.")

    # Execute main trace with signposting for tracy
    signpost("start")
    profiler.start("reduce-to-root-trace")

    ttnn.execute_trace(submesh_device, trace_id, blocking=False)
    ttnn.release_trace(submesh_device, trace_id)
    ttnn.synchronize_device(submesh_device)

    profiler.end("reduce-to-root-trace")
    signpost("stop")

    logger.info("Main trace execution completed.")

    # Verify the output from the last trace execution
    print("\nVerifying trace output...")
    output_l_torch = ttnn.to_torch(out_l_trace, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))

    # Retrieve output from any device (same on all devices)
    out_l_root = output_l_torch[0]

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
