# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_strided_all_gather_async import (
    run_strided_all_gather_impl,
)
from models.common.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_n_or_less_dev,
)


def _create_cluster_submesh(mesh_device, cluster_axis):
    """Create a 1xN (or Nx1) submesh sized to the cluster axis ring.

    The op only operates along cluster_axis, so the non-cluster axis is just
    redundant compute. Sub-meshing keeps the test focused on a single ring.
    """
    submesh_shape = [1, 1]
    submesh_shape[cluster_axis] = mesh_device.shape[cluster_axis]
    return mesh_device.create_submesh(ttnn.MeshShape(tuple(submesh_shape)))


# Isolates the all-gather from the SP=8 / TP=4 fused matmul config in
# test_strided_all_gather_minimal_matmul_async_bh.py, to measure AG performance alone.
# The (8, 4) mesh puts the M shard (other_dim) on axis 0 (SP=8) and the K shard (dim) on
# axis 1 (TP=4); the gather reconstructs full K across the TP group, riding the impl's default
# cluster_axis=1. ag_output_shape is the activation [1, 1, M, K] with M=38912 (4864 per device).
#
# The AG chunk sizing mirrors the fused op's matmul: mm_cores_y = mm_core_grid.y = 8,
# mm_block_h/32 = mm_block_m/32 = 16, mm_block_w/32 = mm_block_k/32 = 8.
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize(
    "ag_output_shape, dim, other_dim, num_workers_per_link, layout, ag_input_dtype, mm_cores_y, mm_block_h, mm_block_w",
    [
        ([1, 1, 4864, 4096], 3, 2, 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, 9, 512, 256),
    ],
    ids=["wan1"],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace,num_iters",
    [
        (True, 3),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_strided_all_gather_async(
    mesh_device,
    ag_output_shape,
    dim,
    other_dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
    num_workers_per_link,
    mm_cores_y,
    mm_block_h,
    mm_block_w,
):
    cluster_axis = 1
    submesh = _create_cluster_submesh(mesh_device, cluster_axis)
    run_strided_all_gather_impl(
        submesh,
        submesh.get_num_devices(),
        ag_output_shape,
        dim,
        other_dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        num_workers_per_link=num_workers_per_link,
        mm_cores_y=mm_cores_y,
        mm_block_h=mm_block_h,
        mm_block_w=mm_block_w,
    )
