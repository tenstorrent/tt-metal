# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blackhole box coverage for ttnn.experimental.reduce_scatter_minimal_direct.

Shares run_reduce_scatter_minimal_direct_impl with the t3000 module -- the driver takes the device as
an argument, so the bh_1d_mesh_device fixture drops straight in.

The op reads its topology from the hardware instead of taking it as an argument, so it needs a ring
that wraps: the whole 1D mesh. num_devices therefore comes from the fixture (validated by
validate_test) rather than being parametrized, which also lets the same cases run on 4- and 8-device
boxes -- every scatter dim below is 8 pages wide, so it splits evenly either way.
"""

import pytest

import ttnn
from models.common.utility_functions import skip_for_n_or_less_dev, skip_for_wormhole_b0
from tests.nightly.t3000.ccl.test_reduce_scatter_minimal_direct import (
    PERSISTENT_MODES,
    RS_DIRECT_DRAM_MEM_CONFIG,
    RS_DIRECT_SHAPE_IDS,
    RS_DIRECT_SHAPES,
    RS_DIRECT_TRACE_CASES,
    RS_DIRECT_TRACE_IDS,
    run_reduce_scatter_minimal_direct_impl,
)
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("rs_input_shape, dim", RS_DIRECT_SHAPES, ids=RS_DIRECT_SHAPE_IDS)
@pytest.mark.parametrize("enable_trace, num_iters", RS_DIRECT_TRACE_CASES, ids=RS_DIRECT_TRACE_IDS)
@pytest.mark.parametrize("persistent_mode", PERSISTENT_MODES)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}],
    indirect=True,
    ids=["fabric_ring"],
)
def test_reduce_scatter_minimal_direct_ring(
    bh_1d_mesh_device,
    num_links,
    rs_input_shape,
    dim,
    rs_input_dtype,
    enable_trace,
    num_iters,
    persistent_mode,
):
    num_devices = bh_1d_mesh_device.get_num_devices()
    validate_test(num_devices, ttnn.Topology.Ring, bh_1d_mesh_device.shape, 0)
    if rs_input_shape[dim] % num_devices:
        pytest.skip(f"scatter dim {dim} (size {rs_input_shape[dim]}) does not split across {num_devices} devices")

    run_reduce_scatter_minimal_direct_impl(
        bh_1d_mesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        ttnn.TILE_LAYOUT,
        RS_DIRECT_DRAM_MEM_CONFIG,
        RS_DIRECT_DRAM_MEM_CONFIG,
        num_iters=num_iters,
        enable_trace=enable_trace,
        persistent_mode=persistent_mode,
    )


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("enable_trace, num_iters", RS_DIRECT_TRACE_CASES, ids=RS_DIRECT_TRACE_IDS)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}],
    indirect=True,
    ids=["fabric_ring"],
)
def test_reduce_scatter_minimal_direct_sharded(
    bh_1d_mesh_device,
    num_links,
    rs_input_dtype,
    enable_trace,
    num_iters,
):
    """Width-sharded L1 in and out. ttnn.reduce_scatter routes sharded cases to this op, so the sharded
    layout needs coverage even though the op's own staging is always interleaved/L1-sharded on its own
    terms. Buffers are op-allocated (persistent_mode="none"), which is how that dispatch calls it.

    Sized off the ring: input width = num_devices^2 tiles over a num_devices-wide core grid, so the
    input shards num_devices tiles/core and the output (num_devices tiles wide) shards 1 tile/core --
    valid on any ring size this box presents.
    """
    num_devices = bh_1d_mesh_device.get_num_devices()
    validate_test(num_devices, ttnn.Topology.Ring, bh_1d_mesh_device.shape, 0)

    rs_input_shape = [1, 1, 32, 32 * num_devices * num_devices]
    output_shape = list(rs_input_shape)
    output_shape[3] //= num_devices

    def width_sharded(shape):
        return ttnn.create_sharded_memory_config(
            tuple(shape),
            core_grid=ttnn.CoreGrid(y=1, x=num_devices),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    run_reduce_scatter_minimal_direct_impl(
        bh_1d_mesh_device,
        num_devices,
        rs_input_shape,
        3,
        num_links,
        rs_input_dtype,
        ttnn.TILE_LAYOUT,
        width_sharded(rs_input_shape),
        width_sharded(output_shape),
        num_iters=num_iters,
        enable_trace=enable_trace,
        persistent_mode="none",
    )
