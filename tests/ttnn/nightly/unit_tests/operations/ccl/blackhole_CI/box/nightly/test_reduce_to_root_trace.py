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
from tests.ttnn.nightly.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


def compute_reference_reduce_to_root(
    l_data_per_device, s_data_per_device, m_data_per_device, root_device_idx=1, num_cores=8, scale_value=1.0
):
    """
    Compute the reference output for reduce_to_root operation.
    """
    num_devices = len(l_data_per_device)

    def split_by_cores(tensor_list, core_width):
        result = []
        for device_tensor in tensor_list:
            cores = torch.chunk(device_tensor, num_cores, dim=1)
            result.append(cores)
        return result

    l_per_device_per_core = split_by_cores(l_data_per_device, 128)
    s_per_device_per_core = split_by_cores(s_data_per_device, 32)
    m_per_device_per_core = split_by_cores(m_data_per_device, 32)

    l_final_cores = []
    s_final_cores = []
    m_final_cores = []

    for core_idx in range(num_cores):
        l_dev = [l_per_device_per_core[d][core_idx] for d in range(num_devices)]
        s_dev = [s_per_device_per_core[d][core_idx] for d in range(num_devices)]
        m_dev = [m_per_device_per_core[d][core_idx] for d in range(num_devices)]

        # Round 1: device 0 -> device 1, device 3 -> device 2
        l1_r1, s1_r1, m1_r1 = l_dev[0], s_dev[0], m_dev[0]
        l2_r1, s2_r1, m2_r1 = l_dev[1], s_dev[1], m_dev[1]

        m_new_dev1 = torch.maximum(m1_r1, m2_r1)
        # P1 = exp((m1 - m) * scale), P2 = exp((m2 - m) * scale)
        exp_m1_dev1 = torch.exp((m1_r1 - m_new_dev1) * scale_value)[:, :1].expand(-1, 128)
        exp_m2_dev1 = torch.exp((m2_r1 - m_new_dev1) * scale_value)[:, :1].expand(-1, 128)
        s_new_dev1 = s1_r1 * torch.exp((m1_r1 - m_new_dev1) * scale_value) + s2_r1 * torch.exp(
            (m2_r1 - m_new_dev1) * scale_value
        )
        l_new_dev1 = l1_r1 * exp_m1_dev1 + l2_r1 * exp_m2_dev1

        # device 3 -> device 2
        l1_r2, s1_r2, m1_r2 = l_dev[3], s_dev[3], m_dev[3]
        l2_r2, s2_r2, m2_r2 = l_dev[2], s_dev[2], m_dev[2]

        m_new_dev2 = torch.maximum(m1_r2, m2_r2)
        # P1 = exp((m1 - m) * scale), P2 = exp((m2 - m) * scale)
        exp_m1_dev2 = torch.exp((m1_r2 - m_new_dev2) * scale_value)[:, :1].expand(-1, 128)
        exp_m2_dev2 = torch.exp((m2_r2 - m_new_dev2) * scale_value)[:, :1].expand(-1, 128)
        s_new_dev2 = s1_r2 * torch.exp((m1_r2 - m_new_dev2) * scale_value) + s2_r2 * torch.exp(
            (m2_r2 - m_new_dev2) * scale_value
        )
        l_new_dev2 = l1_r2 * exp_m1_dev2 + l2_r2 * exp_m2_dev2

        # Round 2: device 2 -> device 1 (final)
        l1_final, s1_final, m1_final = l_new_dev2, s_new_dev2, m_new_dev2
        l2_final, s2_final, m2_final = l_new_dev1, s_new_dev1, m_new_dev1

        m_final = torch.maximum(m1_final, m2_final)
        # P1 = exp((m1 - m) * scale), P2 = exp((m2 - m) * scale)
        exp_m1_final = torch.exp((m1_final - m_final) * scale_value)[:, :1].expand(-1, 128)
        exp_m2_final = torch.exp((m2_final - m_final) * scale_value)[:, :1].expand(-1, 128)
        s_final = s1_final * torch.exp((m1_final - m_final) * scale_value) + s2_final * torch.exp(
            (m2_final - m_final) * scale_value
        )
        l_intermediate = l1_final * exp_m1_final + l2_final * exp_m2_final
        l_final = l_intermediate / s_final[:, :1].expand(-1, 128)

        l_final_cores.append(l_final)
        s_final_cores.append(s_final)
        m_final_cores.append(m_final)

    return torch.cat(l_final_cores, dim=1), torch.cat(s_final_cores, dim=1), torch.cat(m_final_cores, dim=1)


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 217872}),
    ],
    indirect=["device_params"],
    ids=["fabric_1d_linear_trace"],
)
def test_reduce_to_root_with_trace(bh_2d_mesh_device):
    """Test reduce_to_root operation with trace capture and replay."""

    # Setup
    num_devices = 4
    root_coord = (1, 0)
    root_device_idx = root_coord[0]
    num_cores = 8

    topology = ttnn.Topology.Linear
    validate_test(num_devices, topology, bh_2d_mesh_device.shape, 0)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    # Tensor shapes
    l_shape = [8, 128 * num_cores]
    s_shape = [8, 32 * num_cores]
    m_shape = [8, 32 * num_cores]
    intermediate_shapes = [[8, 192 * num_cores]]
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((8, 32))

    # mux cores
    mux_cores = [ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 1), ttnn.CoreCoord(2, 2), ttnn.CoreCoord(2, 3)]

    # Shard config
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 3)),
        }
    )
    shard_spec_l = ttnn.ShardSpec(
        shard_grid,
        [8, 128],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    shard_spec_s = ttnn.ShardSpec(
        shard_grid,
        [8, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    shard_spec_int_l = ttnn.ShardSpec(
        shard_grid,
        [8, 192],
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_l = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_l
    )
    mem_config_s = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_s
    )
    mesh_config_int_l = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_int_l
    )

    mesh_mapper_config = ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    mesh_mapper_config2 = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper2 = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config2)

    # Create intermediate tensors
    intermediate = ttnn.from_torch(
        torch.zeros(intermediate_shapes[0], dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mesh_config_int_l,
        mesh_mapper=mesh_mapper2,
    )

    print("\n=== Testing reduce_to_root with trace ===")

    # Generate test data
    torch.manual_seed(42)
    l_data_per_device = []
    s_data_per_device = []
    m_data_per_device = []

    for device_idx in range(num_devices):
        l_data = torch.randn(l_shape, dtype=torch.bfloat16) * 0.5 + device_idx
        s_data = torch.rand(s_shape, dtype=torch.bfloat16) * 0.5 + 1.0 + device_idx * 0.1
        m_data = torch.randn(m_shape, dtype=torch.bfloat16) * 0.5 + device_idx
        l_data_per_device.append(l_data)
        s_data_per_device.append(s_data)
        m_data_per_device.append(m_data)

    # Create input tensors
    l_data_all = torch.stack(l_data_per_device, dim=0)
    s_data_all = torch.stack(s_data_per_device, dim=0)
    m_data_all = torch.stack(m_data_per_device, dim=0)

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

    scale_value = float(1)
    # Compute reference output
    l_ref, s_ref, m_ref = compute_reference_reduce_to_root(
        l_data_per_device, s_data_per_device, m_data_per_device, root_device_idx, num_cores, scale_value
    )

    profiler = BenchmarkProfiler()
    # Run once to compile
    print("Running reduce_to_root (compiling)...")
    ttnn.reduce_to_root(
        l_tensor,
        s_tensor,
        m_tensor,
        root_coord=ttnn.MeshCoordinate(root_coord),
        scale_fp32=scale_value,
        intermediate_tensor=intermediate,
        topology=topology,
        input_mux_cores=mux_cores,
    )
    ttnn.synchronize_device(submesh_device)

    logger.info("Capturing trace")
    print("Warmup iterations...")
    trace_id_warmup = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    for i in range(15):
        out_l_trace, out_s_trace, out_m_trace = ttnn.reduce_to_root(
            l_tensor,
            s_tensor,
            m_tensor,
            root_coord=ttnn.MeshCoordinate(root_coord),
            scale_fp32=scale_value,
            intermediate_tensor=intermediate,
            topology=topology,
            input_mux_cores=mux_cores,
        )
    ttnn.end_trace_capture(submesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh_device)

    # Capture trace
    print("Capturing trace...")
    trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    for i in range(30):
        out_l_trace, out_s_trace, out_m_trace = ttnn.reduce_to_root(
            l_tensor,
            s_tensor,
            m_tensor,
            root_coord=ttnn.MeshCoordinate(root_coord),
            scale_fp32=scale_value,
            intermediate_tensor=intermediate,
            topology=topology,
            input_mux_cores=mux_cores,
        )

    logger.info("Starting Trace perf test...")
    ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh_device)

    # for warmup
    profiler.start("reduce-to-root-warmup")
    ttnn.execute_trace(submesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(submesh_device, trace_id_warmup)
    ttnn.synchronize_device(submesh_device)
    profiler.end("reduce-to-root-warmup")

    signpost("start")
    profiler.start("reduce-to-root-trace")

    ttnn.execute_trace(submesh_device, trace_id, blocking=False)
    ttnn.release_trace(submesh_device, trace_id)
    ttnn.synchronize_device(submesh_device)

    profiler.end("reduce-to-root-trace")
    signpost("stop")

    # Verify the output from the last trace execution
    print("\nVerifying trace output...")
    out_l_torch = ttnn.to_torch(out_l_trace, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    out_s_torch = ttnn.to_torch(out_s_trace, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    out_m_torch = ttnn.to_torch(out_m_trace, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))

    out_l_root = out_l_torch[root_device_idx]
    out_s_root = out_s_torch[root_device_idx]
    out_m_root = out_m_torch[root_device_idx]

    # Tolerances for bfloat16 with exponentials
    rtol = 0.01
    atol = 0.06

    # Check L tensor
    l_match = torch.allclose(out_l_root, l_ref, rtol=rtol, atol=atol)
    assert l_match, "L tensor output does not match reference after trace execution"

    # Check S tensor (only column 0)
    s_cols_to_check = [i * 32 for i in range(8)]
    s_output_col0 = out_s_root[:, s_cols_to_check]
    s_ref_col0 = s_ref[:, s_cols_to_check]
    s_match = torch.allclose(s_output_col0, s_ref_col0, rtol=rtol, atol=atol)
    assert s_match, "S tensor output does not match reference after trace execution"

    # Check M tensor (only column 0)
    m_cols_to_check = [i * 32 for i in range(8)]
    m_output_col0 = out_m_root[:, m_cols_to_check]
    m_ref_col0 = m_ref[:, m_cols_to_check]
    m_match = torch.allclose(m_output_col0, m_ref_col0, rtol=rtol, atol=atol)
    assert m_match, "M tensor output does not match reference after trace execution"
