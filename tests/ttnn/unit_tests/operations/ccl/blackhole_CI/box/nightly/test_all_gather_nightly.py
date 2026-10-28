# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.nightly.t3000.ccl.test_all_gather import run_all_gather_impl
from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0, skip_for_n_or_less_dev


def ti_cond_skip(condition, reason):
    if condition:
        pytest.skip("Skipping unsupported case: " + reason)


def validate_test(num_devices, topology, shape, cluster_axis):
    if topology is None:
        fabric_config = ttnn.get_fabric_config()
        topology = ttnn.Topology.Ring if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING else ttnn.Topology.Linear
    ti_cond_skip((1 == num_devices), "Can't run a CCL test on 1 device")
    ti_cond_skip(
        ((shape[cluster_axis] != num_devices) and (topology == ttnn.Topology.Ring)),
        "Ring configuration requires the entire row or column so it loops around",
    )
    ti_cond_skip((shape[cluster_axis] < num_devices), "Test requires more devices than are available on this platform")


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (2, [1, 1, 256, 256], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
        ttnn.uint32,
        ttnn.bfloat8_b,
    ],
    ids=[
        "float_16",
        "uint_32",
        "bfloat_8",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
        (False, 3),
    ],
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
)
@pytest.mark.parametrize("cluster_axis", [0])
def test_all_gather_linear_2D_nightly(
    bh_1d_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
    cluster_axis,
):
    validate_test(num_devices, None, bh_1d_mesh_device.shape, cluster_axis)
    if cluster_axis == 0:
        submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


# Multicast with pages that do not divide the packet: fp32 tiles (4096 B) against the 4352 B fabric payload, so packets
# cut tiles, and fp32 row-major rows (8192 B) larger than a packet. The small volume keeps the op on multicast.
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(3)
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, ag_input_dtype, layout",
    [
        (4, [1, 1, 32, 512], 3, ttnn.float32, ttnn.TILE_LAYOUT),
        (4, [1, 1, 32, 8192], 3, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ],
    ids=["fp32_tile_straddles_packets", "fp32_rm_page_larger_than_packet"],
)
@pytest.mark.parametrize("mem_config_input, mem_config_ag", [(ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)])
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
)
@pytest.mark.parametrize("cluster_axis", [0])
def test_all_gather_multicast_non_scatter_pages(
    bh_1d_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    cluster_axis,
):
    validate_test(num_devices, None, bh_1d_mesh_device.shape, cluster_axis)
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=False,
        num_iters=3,
        cluster_axis=cluster_axis,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(3)
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (4, [1, 1, 128, 2048], 3, ttnn.TILE_LAYOUT),
        (4, [1, 1, 256, 768], 3, ttnn.TILE_LAYOUT),
    ],
    ids=["4_device_test", "4_device_large_test"],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.uint32,
        ttnn.bfloat8_b,
    ],
    ids=[
        "uint_32",
        "bfloat_8",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
        (False, 3),
    ],
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
)
@pytest.mark.parametrize("cluster_axis", [0])
def test_all_gather_linear_4D_nightly(
    bh_1d_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
    cluster_axis,
):
    validate_test(num_devices, None, bh_1d_mesh_device.shape, cluster_axis)
    if cluster_axis == 0:
        submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(2)
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout",
    [
        ([1, 1, 128, 2048], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 256, 512], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 256, 768], 3, ttnn.TILE_LAYOUT),
    ],
    ids=["4_device_test", "2_device_test", "4_device_large_test"],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "float_16",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
        (False, 3),
    ],
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
    ],
    indirect=True,
)
@pytest.mark.parametrize("cluster_axis", [0])
def test_all_gather_ring_nightly(
    bh_1d_mesh_device,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
    cluster_axis,
):
    num_devices = bh_1d_mesh_device.shape[0]
    validate_test(num_devices, None, bh_1d_mesh_device.shape, cluster_axis)
    if cluster_axis == 0:
        submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim",
    [
        (2, [1, 1, 256, 256], 3),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype, layout",
    [
        (ttnn.float32, ttnn.TILE_LAYOUT),
        # (ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT),  # insufficient infra support (Issue #43909)
        (ttnn.bfloat16, ttnn.TILE_LAYOUT),
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),  # sanity RM test
        (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        (ttnn.bfloat4_b, ttnn.TILE_LAYOUT),
        (ttnn.uint32, ttnn.TILE_LAYOUT),
        (ttnn.uint16, ttnn.TILE_LAYOUT),
        (ttnn.uint8, ttnn.TILE_LAYOUT),
        (ttnn.int32, ttnn.TILE_LAYOUT),
    ],
    ids=[
        "float32_tile",
        # "fp8_e4m3_rm",
        "bfloat16_tile",
        "bfloat16_rm",
        "bfloat8_b_tile",
        "bfloat4_b_tile",
        "uint32_tile",
        "uint16_tile",
        "uint8_tile",
        "int32_tile",
    ],
)
@pytest.mark.parametrize("mem_config_input, mem_config_ag", [(ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)])
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
)
@pytest.mark.parametrize("cluster_axis", [0])
def test_all_gather_dtype(
    bh_1d_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    cluster_axis,
):
    validate_test(num_devices, None, bh_1d_mesh_device.shape, cluster_axis)
    if cluster_axis == 0:
        submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=False,
        num_iters=3,
        cluster_axis=cluster_axis,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(7)
@pytest.mark.parametrize("ag_output_shape", [[1, 1, 128, 128]])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("ag_input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
    ],
)
@pytest.mark.parametrize("enable_trace", [False])
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112},
    ],
    indirect=True,
)
def test_all_gather_training_shapes(
    bh_2d_mesh_device,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
):
    num_devices = bh_2d_mesh_device.shape[0]
    cluster_axis = 0

    validate_test(num_devices, ttnn.Topology.Linear, bh_2d_mesh_device.shape, cluster_axis)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
    )


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(3)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1_link", "2_links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (4, [1, 1, 32, 128], 3, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(0, 0),
                            ),
                        }
                    ),
                    [32, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(0, 0),
                            ),
                        }
                    ),
                    [32, 128],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (False, 3),
    ],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("cluster_axis", [0])
def test_all_gather_broken(
    bh_2d_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
    cluster_axis,
):
    validate_test(num_devices, all_gather_topology, bh_2d_mesh_device.shape, cluster_axis)
    if cluster_axis == 0:
        submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        num_links=num_links,
        use_persistent_buffers=False,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        allowed_pcc=0.9999,
        all_gather_function=ttnn.experimental.all_gather_async,
        use_barrier=True,
        all_gather_topology=all_gather_topology,
        chunks_per_sync=20,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


def _nd_dram_sharded(shard_shape):
    # A DRAM bank id is the core's x alone, so the grid is one row of 8 banks.
    cores = ttnn.num_cores_to_corerangeset(8, ttnn.CoreCoord(8, 1), row_wise=True)
    spec = ttnn.NdShardSpec(
        ttnn.Shape(shard_shape), cores, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
    )
    return ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec)


def _l1_sharded(memory_layout, shard_height, shard_width, num_cores):
    cores = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(8, 8), row_wise=True)
    return ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
    )


# 8 devices on a line and a ring; small shapes run multicast, large ones unicast. Slices that start mid-row (11 tiles x 5
# rows per device), 256 B row-major rows, a row-major gather on dim 0, an ND-sharded DRAM output with each device's
# stripe in one bank, a height-sharded L1 input, and 192 B row-major rows resharded into 256 B output pages.
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(7)
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, mem_config_input, mem_config_ag",
    [
        ([1, 1, 160, 2816], 3, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 1792, 2816], 3, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 2048, 1024], 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        ([8, 1, 256, 4096], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (
            [1, 1, 512, 8192],
            3,
            ttnn.TILE_LAYOUT,
            _nd_dram_sharded([1, 1, 64, 1024]),
            _nd_dram_sharded([1, 1, 64, 1024]),
        ),
        (
            [1, 1, 2048, 1024],
            3,
            ttnn.TILE_LAYOUT,
            _l1_sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 32, 128, 64),
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        (
            [1, 1, 64, 768],
            3,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
            _l1_sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, 64, 128, 6),
        ),
        (
            [1, 1, 2048, 768],
            3,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
            _l1_sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, 2048, 128, 6),
        ),
    ],
    ids=[
        "mid_row_small",
        "mid_row_large",
        "rm_256B_rows",
        "rm_dim0",
        "nd_dram_stripe_per_bank",
        "l1_height_sharded_in",
        "rm_reshard_3_to_4_small",
        "rm_reshard_3_to_4_large",
    ],
)
@pytest.mark.parametrize("enable_trace", [True, False], ids=["trace", "non-trace"])
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
    ],
    indirect=True,
    ids=["line", "ring"],
)
def test_all_gather_page_walk(
    bh_1d_mesh_device, ag_output_shape, dim, layout, mem_config_input, mem_config_ag, enable_trace
):
    num_devices = bh_1d_mesh_device.shape[0]
    validate_test(num_devices, None, bh_1d_mesh_device.shape, 0)
    run_all_gather_impl(
        bh_1d_mesh_device,
        ag_output_shape,
        dim,
        ttnn.bfloat16,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=2,
        cluster_axis=0,
    )


# Back-to-back gathers into one persistent output with no sync between them, every device but the first stalled after
# each launch: a fast device's next gather must not write into a neighbor's output before that neighbor has relayed or
# copied out the previous one. Small runs multicast, large unicast.
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(7)
@pytest.mark.parametrize("ag_output_shape", [[1, 1, 128, 1024], [1, 1, 1024, 2048]], ids=["small", "large"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}, {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
    ids=["line", "ring"],
)
def test_all_gather_persistent_output_overlap(bh_1d_mesh_device, ag_output_shape):
    mesh_device = bh_1d_mesh_device
    num_iters = 10
    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=tuple(mesh_device.shape))
    goldens = [torch.randn(ag_output_shape, dtype=torch.bfloat16) for _ in range(num_iters)]
    inputs = [
        ttnn.from_torch(golden, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper)
        for golden in goldens
    ]

    def replicated_zeros():
        return ttnn.from_torch(
            torch.zeros(ag_output_shape, dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    output = replicated_zeros()
    captures = [replicated_zeros() for _ in range(num_iters)]
    ttnn.all_gather(inputs[0], dim=3, cluster_axis=0, output_tensor=output)  # compile: every launch below is cached
    ttnn.synchronize_device(mesh_device)

    rows, cols = tuple(mesh_device.shape)
    delays = [[0 if (r, c) == (0, 0) else 2_000_000 for c in range(cols)] for r in range(rows)]
    for i in range(num_iters):
        result = ttnn.all_gather(inputs[i], dim=3, cluster_axis=0, output_tensor=output)
        ttnn.apply_device_delay(mesh_device, delays)
        ttnn.copy(result, captures[i])
    ttnn.synchronize_device(mesh_device)

    for i, capture in enumerate(captures):
        for d, device_tensor in enumerate(ttnn.get_device_tensors(capture)):
            assert torch.equal(ttnn.to_torch(device_tensor), goldens[i]), f"iteration {i}: device {d} mismatch"


# A persistent output in another memory than the memory_config (unset here, so the input's L1): the op builds for the
# output it writes. 16 B L1 rows into 128 B DRAM rows only multicast can carry (the unicast relay would re-read its DRAM
# output at the 16 B L1 alignment), at a size that would otherwise pick unicast.
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(7)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}, {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
    ids=["line", "ring"],
)
def test_all_gather_persistent_output_memory(bh_1d_mesh_device):
    mesh_device = bh_1d_mesh_device
    shape = [1, 1, 65536, 64]
    golden = torch.randn(shape, dtype=torch.bfloat16)
    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=tuple(mesh_device.shape))
    tt_input = ttnn.from_torch(
        golden,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    output = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.all_gather(tt_input, dim=3, cluster_axis=0, output_tensor=output)
    ttnn.synchronize_device(mesh_device)
    for d, device_tensor in enumerate(ttnn.get_device_tensors(output)):
        assert torch.equal(ttnn.to_torch(device_tensor), golden), f"device {d} mismatch"
