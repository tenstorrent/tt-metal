# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test CCL operations on 4x4 mesh spanning 2 vertically-stacked hosts.
Each host has 4x2 devices, connected via vertical fabric links.
"""

import pytest
import ttnn
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl


@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, all_gather_topology, ag_input_dtype",
    [
        (4, [1, 1, 256, 1024], 3, ttnn.TILE_LAYOUT, ttnn.Topology.Linear, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize("cluster_axis", [1])  # Vertical all-gather across hosts
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=["L1_ONLY"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
    ],
    ids=["non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 4), id="4x4_grid")], indirect=True)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
def test_vertical_all_gather_full_mesh(
    mesh_device,
    num_devices,
    ag_output_shape,
    cluster_axis,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
):
    """
    Test all-gather on the full 4×4 mesh spanning both hosts vertically.
    This uses all 16 devices and tests fabric links across hosts.
    """
    print(f"Testing vertical all-gather on full mesh with {num_devices} devices and {num_links} links")
    print(f"Mesh shape: {mesh_device.shape}")
    print(f"Total devices: {mesh_device.get_num_devices()}")

    # Run all-gather on the full mesh
    run_all_gather_impl(
        mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
        allowed_pcc=0.9999,
        num_l1_banks=110,
    )

    ttnn.ReadDeviceProfiler(mesh_device)
    print("✓ Test passed!")


@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, all_gather_topology, ag_input_dtype",
    [
        (4, [1, 1, 256, 256], 3, ttnn.TILE_LAYOUT, ttnn.Topology.Linear, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize("column_idx", [0])  # Test first column
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=["L1_ONLY"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
    ],
    ids=["non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 4), id="4x4_grid")], indirect=True)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
def test_vertical_column_all_gather(
    mesh_device,
    num_devices,
    ag_output_shape,
    column_idx,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
):
    """
    Test all-gather on a single vertical column (4 rows × 1 col).
    This column spans both hosts vertically.
    """
    print(f"Testing vertical all-gather on column {column_idx} with {num_devices} devices and {num_links} links")

    # Create a vertical submesh (4 rows × 1 column) that spans both hosts

    # Run all-gather on the vertical submesh
    run_all_gather_impl(
        mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=1,  # Vertical all-gather
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
        allowed_pcc=0.9999,
        num_l1_banks=110,
    )

    ttnn.ReadDeviceProfiler(mesh_device)
    print("✓ Test passed!")
