# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723 review, CI only): the Blackhole Galaxy strided matmul reduce-scatter case of tt_dit's Wan
test (test_strided_reduce_scatter_wan_tg.py) with the fused addcmul that tt_dit's linear layer passes, broadcast and full
gate, so the reduction kernel's addcmul multiply runs on its production path."""
import pytest

import ttnn
from tests.nightly.t3000.ccl.test_minimal_matmul_strided_reduce_scatter_async import (
    run_minimal_matmul_strided_reduce_scatter_impl,
)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis_0", "axis_1"])
@pytest.mark.parametrize(
    "device_params, topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.parametrize("broadcast_gate", [True, False], ids=["broadcast_gate", "full_gate"])
def test_srs_addcmul(mesh_device, num_links, cluster_axis, topology, broadcast_gate):
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        M=9472,
        K=3456,
        N=5120,
        dim=3,
        num_links=num_links,
        input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mem_config_input=mem_config,
        mem_config_mm=mem_config,
        mem_config_rs=mem_config,
        topology=topology,
        mm_block_m=256,
        mm_block_k=128,
        mm_block_n=256,
        subblock_h=2,
        subblock_w=1,
        mm_core_grid=ttnn.CoreCoord(12, 8),
        chunk_width_in_mm_blocks=1,
        num_workers_per_link=5,
        rs_core_grid_offset=ttnn.CoreCoord(0, 8),
        rs_mode="fused",
        cluster_axis=cluster_axis,
        addcmul_scalar=0.5,
        broadcast_gate=broadcast_gate,
    )
