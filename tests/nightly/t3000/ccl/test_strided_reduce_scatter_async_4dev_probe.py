# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Hardware probe: strided_reduce_scatter_async on a 4-device ring.

The shipped strided tests are @skip_for_blackhole and pin (1, 8) meshes; this mirrors
test_strided_reduce_scatter_blocking_sweep at (1, 4) with a few block configs so the strided
family's minimal_ring_reduction compute kernel (migrated onto BlockAccumulate) has 4-device
coverage — including on Blackhole, where it passes with exact PCC (the arch gate on the shipped
tests appears to be unvalidated conservatism).

Shape [4, 1, 416, 2048] dim=3 at ring_size=4: slice_Ht = 13 tiles, slice_Wt = 16 tiles.
"""

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_strided_reduce_scatter_async import run_reduce_scatter_impl


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mm_cores_y, mm_block_ht, mm_block_wt, mm_N_full_block_wt, chunk_width_in_mm_blocks",
    [
        (1, 1, 1, 1, 1),  # finest granularity
        (4, 2, 2, 2, 1),  # multiple N-blocks
        (1, 16, 8, 8, 1),  # coarsest: single M-block (partial: slice_Ht=13), single N-block
        (2, 8, 2, 8, 3),  # partial last chunk (chunk_w=6 into N_block=8)
        (1, 4, 4, 3, 1),  # non-div Wt (slice_Wt=16 % 3): ghost tiles mid-packet -> unicast fallback
    ],
    ids=["finest", "multi_N_blocks", "coarsest", "partial_last_chunk", "non_div_Wt_ghost_tiles"],
)
def test_strided_reduce_scatter_hw_probe(
    mesh_device,
    mm_cores_y,
    mm_block_ht,
    mm_block_wt,
    mm_N_full_block_wt,
    chunk_width_in_mm_blocks,
):
    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        [4, 1, 416, 2048],
        3,  # dim
        1,  # num_links
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        rs_topology=ttnn.Topology.Ring,
        enable_trace=False,
        num_iters=1,
        small_random_ints=True,
        use_barrier=True,
        use_persistent_buffers=True,
        use_strided=True,
        verify_output_shape=True,
        verify_output_pcc=True,
        mm_cores_y=mm_cores_y,
        mm_block_ht=mm_block_ht,
        mm_block_wt=mm_block_wt,
        mm_N_full_block_wt=mm_N_full_block_wt,
        chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
    )
