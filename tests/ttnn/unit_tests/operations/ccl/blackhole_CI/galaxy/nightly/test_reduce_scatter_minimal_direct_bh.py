# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blackhole galaxy coverage for ttnn.experimental.reduce_scatter_minimal_direct.

Shares run_reduce_scatter_minimal_direct_impl with the t3000 module -- the driver takes the device as
an argument, so the bh_2d_mesh_device fixture drops straight in.

The op reads its topology from the hardware instead of taking it as an argument, so the collective
must span a FULL mesh row (a partial row has no wraparound and is not a ring). num_devices is
therefore taken from the mesh's cluster-axis extent rather than parametrized -- the neighbouring
minimal_async ring test hardcodes a 4-device group on an 8-wide axis and is currently skipped for
hanging, which is exactly the failure mode this avoids.
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


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("cluster_axis", [1], ids=["axis1"])
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("rs_input_shape, dim", RS_DIRECT_SHAPES, ids=RS_DIRECT_SHAPE_IDS)
@pytest.mark.parametrize("enable_trace, num_iters", RS_DIRECT_TRACE_CASES, ids=RS_DIRECT_TRACE_IDS)
# "both" and "none" are the two that matter here: they select whether the writer's start barrier is
# compiled in. The "staging" helper path is covered on t3000 and the blackhole box.
@pytest.mark.parametrize("persistent_mode", [m for m in PERSISTENT_MODES if m != "staging"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}],
    indirect=True,
    ids=["fabric_ring"],
)
def test_reduce_scatter_minimal_direct_ring(
    bh_2d_mesh_device,
    num_links,
    cluster_axis,
    rs_input_shape,
    dim,
    rs_input_dtype,
    enable_trace,
    num_iters,
    persistent_mode,
):
    num_devices = tuple(bh_2d_mesh_device.shape)[cluster_axis]
    if num_devices < 2:
        pytest.skip(f"reduce_scatter needs a ring of at least 2, got {num_devices} on axis {cluster_axis}")
    if rs_input_shape[dim] % num_devices:
        pytest.skip(f"scatter dim {dim} (size {rs_input_shape[dim]}) does not split across {num_devices} devices")

    run_reduce_scatter_minimal_direct_impl(
        bh_2d_mesh_device,
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
        cluster_axis=cluster_axis,
        persistent_mode=persistent_mode,
    )
