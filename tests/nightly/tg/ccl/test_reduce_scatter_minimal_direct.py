# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TG coverage for ttnn.experimental.reduce_scatter_minimal_direct.

Shares run_reduce_scatter_minimal_direct_impl with the t3000 module (same import convention the
minimal_async TG test uses); this module only picks the submesh and the parametrization.

The op derives its topology from the hardware rather than taking it as an argument, so the ring must
be a full mesh row/column -- hence the cluster-axis-0 submesh of the whole 8-device dimension.
"""

import pytest

import ttnn
from models.common.utility_functions import skip_for_blackhole
from tests.nightly.t3000.ccl.test_reduce_scatter_minimal_direct import (
    PERSISTENT_MODES,
    RS_DIRECT_DRAM_MEM_CONFIG,
    RS_DIRECT_SHAPE_IDS,
    RS_DIRECT_SHAPES,
    RS_DIRECT_TRACE_CASES,
    RS_DIRECT_TRACE_IDS,
    run_reduce_scatter_minimal_direct_impl,
)


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("rs_input_shape, dim", RS_DIRECT_SHAPES, ids=RS_DIRECT_SHAPE_IDS)
@pytest.mark.parametrize("enable_trace, num_iters", RS_DIRECT_TRACE_CASES, ids=RS_DIRECT_TRACE_IDS)
# "both" and "none" are the two that matter here: they select whether the writer's start barrier is
# compiled in. The "staging" helper path is covered on t3000/blackhole.
@pytest.mark.parametrize("persistent_mode", [m for m in PERSISTENT_MODES if m != "staging"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}],
    indirect=True,
    ids=["fabric_ring"],
)
def test_reduce_scatter_minimal_direct(
    mesh_device,
    num_links,
    rs_input_shape,
    dim,
    rs_input_dtype,
    enable_trace,
    num_iters,
    persistent_mode,
):
    cluster_axis = 0
    num_devices = tuple(mesh_device.shape)[cluster_axis]
    if num_devices < 2:
        pytest.skip(f"reduce_scatter needs a ring of at least 2, got {num_devices} on axis {cluster_axis}")
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    run_reduce_scatter_minimal_direct_impl(
        submesh_device,
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
