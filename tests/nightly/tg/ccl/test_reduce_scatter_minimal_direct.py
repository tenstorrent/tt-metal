# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TG coverage for ttnn.experimental.reduce_scatter_minimal_direct.

Shares run_reduce_scatter_minimal_direct_impl with the t3000 module (same import convention the
minimal_async TG test uses); this module only picks the mesh and the parametrization.

The op derives its topology from the hardware rather than taking it as an argument, so the ring must
be a full mesh row/column -- hence the cluster-axis-0 submesh of the whole 8-device dimension in the
first test.

The tests after it drop the submesh and run on the whole 8x4 mesh, which is what makes them different:
a cluster_axis there is one independent ring per row (or column) of the OTHER axis, all running at once.
That is how ttnn.reduce_scatter's auto-dispatch reaches this op, it is what hung on Galaxy (#54864), and
until those tests existed nothing covered it.
"""

import pytest

import ttnn
from models.common.utility_functions import skip_for_blackhole
from tests.nightly.t3000.ccl.test_reduce_scatter_minimal_direct import (
    PERSISTENT_MODES,
    RS_DIRECT_DEEPSEEK_SHAPE_IDS,
    RS_DIRECT_DEEPSEEK_SHAPES,
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


# --- Concurrent rings: the whole 8x4 mesh, not an (8,1) submesh of it ----------------------------
#
# The case above, and every other module that drives this op, hands it ONE ring: a 1xN or Nx1 (sub)mesh.
# ttnn.reduce_scatter's auto-dispatch does not -- it reaches this op with a cluster_axis on the whole 2-D
# mesh, which is one INDEPENDENT ring per row (or column) of the other axis, all of them running at once
# and sharing the mesh's links. DeepSeek-V3's TG MoE decode block does exactly that
# (cluster_axis=1, num_links=4 on 8x4), and it hung on every scheduled run: the writers park in the
# start barrier on 12 of the 32 devices (#54864). Nothing covered that configuration, which is why the
# dispatch gate in ttnn.reduce_scatter now keeps such callers on the ring op; these are the tests that
# configuration was missing.
#
# Both cluster axes are worth running: they are rings of different sizes (4 across, 8 down), a different
# number of concurrent rings (8 vs 4), and different physical wiring for the closing link.
@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0", "axis1"])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("rs_input_shape, dim", RS_DIRECT_SHAPES, ids=RS_DIRECT_SHAPE_IDS)
@pytest.mark.parametrize("enable_trace, num_iters", RS_DIRECT_TRACE_CASES, ids=RS_DIRECT_TRACE_IDS)
# "both" and "none" select whether the writer's start barrier -- where the hang parks -- is compiled in.
# The "staging" helper path is covered on t3000/blackhole and is orthogonal to the mesh shape.
@pytest.mark.parametrize("persistent_mode", [m for m in PERSISTENT_MODES if m != "staging"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}],
    indirect=True,
    ids=["fabric_ring"],
)
def test_reduce_scatter_minimal_direct_2d_mesh(
    mesh_device,
    cluster_axis,
    num_links,
    rs_input_shape,
    dim,
    rs_input_dtype,
    enable_trace,
    num_iters,
    persistent_mode,
):
    num_devices = tuple(mesh_device.shape)[cluster_axis]
    if num_devices < 2:
        pytest.skip(f"reduce_scatter needs a ring of at least 2, got {num_devices} on axis {cluster_axis}")
    if rs_input_shape[dim] % num_devices:
        pytest.skip(f"scatter dim {dim} (size {rs_input_shape[dim]}) does not split across {num_devices} devices")

    run_reduce_scatter_minimal_direct_impl(
        mesh_device,
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


# The reported hang, reduced to this op: DeepSeek-V3's own shape and link count on the whole 8x4 mesh,
# under trace, with op-allocated buffers -- which is how ttnn.reduce_scatter's auto-dispatch calls it and
# what compiles the start barrier in. Split out from the matrix above because the link count only means
# anything at this size: the small shapes are one chunk per slice, so num_links clamps back to 1 there.
#
# num_links=None asks for every link the cluster axis has (4 on a Galaxy) rather than hardcoding 4: an
# over-ask is a TT_FATAL, not a skip, and the count a host reports for axis 0 is not guaranteed to match
# axis 1 (see the dispatch-link note in reduce_scatter.cpp).
@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0", "axis1"])
@pytest.mark.parametrize("num_links", [1, None], ids=["1link", "all_links"])
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("rs_input_shape, dim", RS_DIRECT_DEEPSEEK_SHAPES, ids=RS_DIRECT_DEEPSEEK_SHAPE_IDS)
@pytest.mark.parametrize("enable_trace, num_iters", [(True, 3)], ids=["trace"])
@pytest.mark.parametrize("persistent_mode", ["none"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}],
    indirect=True,
    ids=["fabric_ring"],
)
def test_reduce_scatter_minimal_direct_2d_mesh_deepseek(
    mesh_device,
    cluster_axis,
    num_links,
    rs_input_shape,
    dim,
    rs_input_dtype,
    enable_trace,
    num_iters,
    persistent_mode,
):
    num_devices = tuple(mesh_device.shape)[cluster_axis]
    if rs_input_shape[dim] % num_devices:
        pytest.skip(f"scatter dim {dim} (size {rs_input_shape[dim]}) does not split across {num_devices} devices")

    run_reduce_scatter_minimal_direct_impl(
        mesh_device,
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
