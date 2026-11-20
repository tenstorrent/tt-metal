# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_py_test_utils_common.all_gather_validate import ti_cond_skip, validate_test

import math

import pytest
import torch
import ttnn

from tests.nightly.t3000.ccl.test_all_gather import run_all_gather_impl
from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0, skip_for_n_or_less_dev


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


# Blackhole's ring crossover in all_gather's factory selection: unicast once the busiest link-direction carries this
# many bytes, i.e. each device's input bytes * (N - 1) / (2 * links).
BH_RING_UNICAST_LINK_BYTES = 233_000
# Each ring row sits this far below (multicast) or above (unicast) the crossover on every device and link count.
RING_FACTORY_MARGIN = 4


def ring_input_rows(num_devices, num_links, row_bytes, unicast):
    """Rows of each device's input, a multiple of 32, about RING_FACTORY_MARGIN times below or above the crossover."""
    link_bytes = (
        BH_RING_UNICAST_LINK_BYTES * RING_FACTORY_MARGIN
        if unicast
        else BH_RING_UNICAST_LINK_BYTES / RING_FACTORY_MARGIN
    )
    tile_rows = link_bytes * 2 * num_links / ((num_devices - 1) * row_bytes * 32)
    return 32 * max(1, math.ceil(tile_rows) if unicast else math.floor(tile_rows))


# A ring of every device in the box: multicast, then unicast with short row-major rows (256 B per device, so packets of
# 4 KB or less and 5 workers) and with 1 tile per device per output row. Each device's input width is fixed and its
# height is set from the ring's device and link counts, so each row stays on its factory on any box.
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(2)
@pytest.mark.parametrize(
    "input_width, layout, unicast",
    [
        (64, ttnn.TILE_LAYOUT, False),
        (128, ttnn.ROW_MAJOR_LAYOUT, True),
        (32, ttnn.TILE_LAYOUT, True),
    ],
    ids=["multicast", "unicast_rm_short_rows", "unicast_narrow_tile_rows"],
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
    input_width,
    layout,
    unicast,
    ag_input_dtype,
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
    num_links = ttnn.get_num_links(submesh_device, cluster_axis=cluster_axis)
    rows = ring_input_rows(num_devices, num_links, input_width * 2, unicast)  # 2 B per bfloat16
    run_all_gather_impl(
        submesh_device,
        [1, 1, rows, input_width * num_devices],
        3,
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


# Back-to-back gathers into one persistent output with no sync, every device but (0, 0) stalled after each launch and
# each result copied out in queue order: a fast device's next gather must not write a neighbor's output before that
# neighbor has relayed (unicast) or copied out (multicast) the previous one, so the init handshake must stay on with a
# persistent output. The multicast scenario gathers L1 inputs into a DRAM persistent output with no memory_config: the
# factory is chosen for the output it writes (16 B rows the unicast relay cannot re-read from DRAM), at a size the
# input's L1 spec would send to unicast.
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(7)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
def test_all_gather_persistent_output(bh_1d_mesh_device):
    mesh_device = bh_1d_mesh_device
    num_iters = 10
    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=tuple(mesh_device.shape))
    rows, cols = tuple(mesh_device.shape)
    delays = [[0 if (r, c) == (0, 0) else 2_000_000 for c in range(cols)] for r in range(rows)]
    scenarios = {
        "multicast_l1_in_dram_out": ([1, 1, 16384, 64], ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG),
        "unicast": ([1, 1, 1024, 2048], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
    }
    for scenario, (shape, layout, input_memory_config) in scenarios.items():
        goldens = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(num_iters)]
        inputs = [
            ttnn.from_torch(
                golden, device=mesh_device, layout=layout, memory_config=input_memory_config, mesh_mapper=mesh_mapper
            )
            for golden in goldens
        ]

        def replicated_zeros():
            return ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                device=mesh_device,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        output = replicated_zeros()
        captures = [replicated_zeros() for _ in range(num_iters)]
        ttnn.all_gather(inputs[0], dim=3, cluster_axis=0, output_tensor=output)  # compile: every launch below is cached
        ttnn.synchronize_device(mesh_device)

        for i in range(num_iters):
            result = ttnn.all_gather(inputs[i], dim=3, cluster_axis=0, output_tensor=output)
            ttnn.apply_device_delay(mesh_device, delays)
            ttnn.copy(result, captures[i])
        ttnn.synchronize_device(mesh_device)

        for i, capture in enumerate(captures):
            for d, device_tensor in enumerate(ttnn.get_device_tensors(capture)):
                assert torch.equal(
                    ttnn.to_torch(device_tensor), goldens[i]
                ), f"{scenario}: iteration {i}: device {d} mismatch"
