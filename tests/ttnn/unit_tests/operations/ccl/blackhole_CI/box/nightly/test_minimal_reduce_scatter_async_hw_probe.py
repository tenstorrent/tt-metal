# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Hardware probe: ring-topology dim-0 reduce_scatter_minimal_async on 4 devices.

The shipped QuietBox nightly covers ring dim-3 and line dim-3/dim-0, but has no ring dim-0 case —
and dim 0 selects the dim_zero_ring_* kernel triple (dim_zero_ring_reduction.cpp), which the
schedule/BlockAccumulate migration touches. This mirrors the sim probe's dim0 case at 4 devices.

The [8, 1, 32, 256] shape gives slice_B = 2 on 4 devices, so the dim-zero schedule's inner batch
loop runs more than once.
"""

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_minimal_reduce_scatter_async_bh import (
    run_reduce_scatter_impl,
)


@pytest.mark.parametrize("num_devices", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "rs_input_shape, dim",
    [
        ([4, 1, 32, 256], 0),
        ([8, 1, 32, 256], 0),
    ],
    ids=["dim0_b4", "dim0_b8_sliceB2"],
)
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("mem_config_input", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
@pytest.mark.parametrize("mem_config_rs", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
@pytest.mark.parametrize("num_iters", [2])  # >1 so the program-cache-hit path is exercised too
@pytest.mark.parametrize("use_barrier", [True])
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        (
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456},
            ttnn.Topology.Ring,
        )
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_reduce_scatter_async_ring_dim0_4dev_hw(
    bh_1d_mesh_device,
    num_devices,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    rs_topology,
    num_iters,
    use_barrier,
):
    run_reduce_scatter_impl(
        bh_1d_mesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology,
        num_iters=num_iters,
        enable_trace=False,
        use_barrier=use_barrier,
    )
