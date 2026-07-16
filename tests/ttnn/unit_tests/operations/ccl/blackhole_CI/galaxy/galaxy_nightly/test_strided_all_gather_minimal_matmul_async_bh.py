# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_strided_all_gather_minimal_matmul_async import (
    run_strided_all_gather_minimal_matmul_impl,
)
from models.common.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_n_or_less_dev,
)


# SP=8 / TP=4 on the blackhole galaxy. The (8, 4) mesh puts the M/sequence shard (other_dim) on
# axis 0 (SP=8) and the K/contraction shard (dim) on axis 1 (TP=4), matching the impl's
# shard_dims = [other_dim, dim]. The all-gather reconstructs full K across the TP group, so it
# rides axis 1 -- the impl's default cluster_axis=1. K gathers over TP=4, so per-device K is
# 4096/4 = 32 tiles; mm_block_k is 256 (8 tiles), which divides it.
#
# Core-grid partition (blackhole 12x10 grid): matmul takes the lower mm_core_grid.y rows at width
# mm_core_grid.x; the strided all-gather workers take the rows above, starting at ag_offset, so
# ag_offset.y must equal mm_core_grid.y to keep the two regions disjoint.
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize(
    "M, K, N, dim, other_dim, num_workers_per_link, layout, ag_input_dtype, mm_block_m, mm_block_k, mm_block_n, subblock_h, subblock_w, mm_core_grid, shard_weights, ag_offset",
    [
        (
            38912,  # 4864 per device (SP=8): 152 tiles; against a 16-tile mm_block_m this is 9.5 blocks -> ragged last M-block (hang repro)
            4096,
            1024,
            3,
            2,
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            512,
            256,
            128,
            2,
            2,
            ttnn.CoreCoord(12, 8),
            False,
            (0, 8),
        ),
    ],
    ids=["wan1"],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag, mem_config_mm",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
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
    "use_non_fused",
    [
        False,
    ],
    ids=["fused"],
)
@pytest.mark.parametrize(
    "read_local_slice_from_input",
    [
        True,
    ],
    ids=["read_local"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_strided_all_gather_minimal_matmul_async(
    mesh_device,
    M,
    K,
    N,
    dim,
    other_dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    enable_trace,
    all_gather_topology,
    num_iters,
    num_workers_per_link,
    mm_block_m,
    mm_block_k,
    mm_block_n,
    subblock_h,
    subblock_w,
    mm_core_grid,
    use_non_fused,
    shard_weights,
    ag_offset,
    read_local_slice_from_input,
):
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x < mm_core_grid.x or grid.y < ag_offset[1] + 1:
        pytest.skip(f"Requires worker grid >= {mm_core_grid.x}x{ag_offset[1] + 1}, got {grid.x}x{grid.y}")

    run_strided_all_gather_minimal_matmul_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        M,
        K,
        N,
        dim,
        other_dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        mem_config_mm,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        num_workers_per_link=num_workers_per_link,
        mm_block_m=mm_block_m,
        mm_block_k=mm_block_k,
        mm_block_n=mm_block_n,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        mm_core_grid=mm_core_grid,
        use_non_fused=use_non_fused,
        shard_weights=shard_weights,
        ag_core_grid_offset=ag_offset,
        read_local_slice_from_input=read_local_slice_from_input,
    )
