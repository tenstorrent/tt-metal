# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_strided_all_gather_async import run_strided_all_gather_impl
from models.common.utility_functions import skip_for_blackhole


# tiles_per_chunk needs to be divisible by num_workers_per_link
# mm_cores_y is the number of in0 first col cores
# mm_block_h and mm_block_w is the mm_block of a single mm_core_y
# so the result of one chunk transfer will be mm_cores_y * mm_block_h * mm_block_w, which will be tiles_per_chunk.  tiles_per_chunk % num_workers_per_link must equal 0
@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim, other_dim, num_links, num_workers_per_link, tiles_per_chunk, layout, ag_input_dtype, mm_cores_y, mm_block_h, mm_block_w",
    [
        # ([1, 1, 256, 128], 3, 2, 1, 1, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 32, 32),
        # ([1, 1, 256, 256], 3, 2, 2, 1, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 32, 64),
        # ([1, 1, 512, 256], 3, 2, 2, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 64, 64),
        # # 2 row tests
        # ([1, 1, 512, 128], 3, 2, 1, 1, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 64, 32),
        # ([1, 1, 512, 128], 3, 2, 2, 1, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 64, 32),
        # ([1, 1, 512, 256], 3, 2, 2, 1, 4, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 64, 64),
        # # 4 row tests
        # ([1, 1, 1024, 128], 3, 2, 1, 1, 4, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 128, 32),
        # ([1, 1, 1024, 128], 3, 2, 4, 1, 4, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 128, 32),
        # # Multiple y core tests
        # ([1, 1, 1024, 128], 3, 2, 4, 1, 4, ttnn.TILE_LAYOUT, ttnn.bfloat16, 2, 128, 32),
        # # Full tests
        # ([1, 1, 4096, 2048], 3, 2, 1, 1, 64, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 256, 256),
        # ([1, 1, 4096, 2048], 3, 2, 4, 1, 64, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 256, 256),
        # Wan tests
        ([1, 1, 75776, 5120], 3, 2, 4, 2, 64, ttnn.TILE_LAYOUT, ttnn.bfloat16, 6, 256, 256),
    ],
    ids=[
        # "1tile1chunk1worker1row1link",
        # "1tile1chunk1worker1row2link",
        # "1tile1chunk2worker1row2link",
        # # 2 row tests
        # "2tile1chunk1worker2row1link",
        # "2tile1chunk1worker2row2link",
        # "2tile1chunk1worker2row2col2link",
        # # 4 row tests
        # "4tile1chunk1worker4row1link",
        # "4tile1chunk1worker4row4link",
        # # Multiple y core tests
        # "multiycore",
        # # Full tests
        # "fulltest",
        # "fulltest4link",
        # Wan tests
        "wan1",
    ],
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
        (False, 1),
    ],
    ids=["check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
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
    tiles_per_chunk,
    mm_cores_y,
    mm_block_h,
    mm_block_w,
):
    run_strided_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        tiles_per_chunk=tiles_per_chunk,
        mm_cores_y=mm_cores_y,
        mm_block_h=mm_block_h,
        mm_block_w=mm_block_w,
    )
