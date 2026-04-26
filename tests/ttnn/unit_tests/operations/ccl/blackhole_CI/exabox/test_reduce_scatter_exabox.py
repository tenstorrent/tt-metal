# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""Reduce-scatter tests for multi-galaxy exabox mesh configurations (DUAL_BH / QUAD_BH).

Tests cover:
- Async reduce_scatter on 16x4 and 32x4 meshes via shared runner
- Multiple fabric configs (FABRIC_1D, FABRIC_1D_RING) and topologies (Linear, Ring)
- Both cluster axes (0 and 1)
- DRAM and L1 memory configs
- num_links variation (1 and 2 links for BH Galaxy)
"""

import pytest

import ttnn
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl


# ---------------------------------------------------------------------------
# Test: reduce_scatter on 16x4 mesh (DUAL_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["DUAL_BH"])
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            id="fabric_1d-linear",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((16, 4), id="16x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "cluster_axis, num_devices",
    [
        pytest.param(0, 16, id="axis0_16dev"),
        pytest.param(1, 4, id="axis1_4dev"),
    ],
)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2links"])
@pytest.mark.parametrize(
    "rs_input_shape, dim, layout",
    [
        ([1, 1, 64, 512], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16], ids=["bfloat16"])
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=["l1_interleaved"],
)
@pytest.mark.parametrize("enable_trace, num_iters", [(False, 2)], ids=["non-trace"])
@pytest.mark.parametrize("chunks_per_sync", [2])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [8])
def test_reduce_scatter_16x4(
    mesh_device,
    cluster_axis,
    num_devices,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    rs_topology,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
):
    if cluster_axis == 0:
        submesh = mesh_device.create_submesh(ttnn.MeshShape(num_devices, 1))
    else:
        submesh = mesh_device.create_submesh(ttnn.MeshShape(1, num_devices))

    run_reduce_scatter_impl(
        submesh,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )


# ---------------------------------------------------------------------------
# Test: reduce_scatter on 32x4 mesh (QUAD_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            id="fabric_1d-linear",
        ),
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
            ttnn.Topology.Ring,
            id="fabric_1d_ring-ring",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "cluster_axis, num_devices",
    [
        pytest.param(0, 32, id="axis0_32dev"),
        pytest.param(1, 4, id="axis1_4dev"),
    ],
)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2links"])
@pytest.mark.parametrize(
    "rs_input_shape, dim, layout",
    [
        ([1, 1, 64, 512], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 256], 3, ttnn.TILE_LAYOUT),
    ],
    ids=["small", "large"],
)
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16], ids=["bfloat16"])
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
    ],
    ids=["l1_interleaved", "dram_interleaved"],
)
@pytest.mark.parametrize("enable_trace, num_iters", [(False, 2)], ids=["non-trace"])
@pytest.mark.parametrize("chunks_per_sync", [2])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [8])
def test_reduce_scatter_32x4(
    mesh_device,
    cluster_axis,
    num_devices,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    rs_topology,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
):
    if cluster_axis == 0:
        submesh = mesh_device.create_submesh(ttnn.MeshShape(num_devices, 1))
    else:
        submesh = mesh_device.create_submesh(ttnn.MeshShape(1, num_devices))

    run_reduce_scatter_impl(
        submesh,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
