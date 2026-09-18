# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TG coverage for ttnn.experimental.reduce_scatter_minimal_direct.

Shares run_reduce_scatter_minimal_direct_impl with the t3000 module (same import convention the
minimal_async TG test uses); this module only picks the mesh and the parametrization.

The op derives its topology from the hardware rather than taking it as an argument, so the ring must
be a full mesh row/column -- hence the "submesh_ring" group below, which cuts a cluster-axis-0 submesh
out of the whole 8-device dimension.

The "multi_ring" group drops the submesh and runs on the whole 8x4 mesh, which is what makes it
different: a cluster_axis there is one independent ring per row (or column) of the OTHER axis, all
running at once. That is how ttnn.reduce_scatter's auto-dispatch reaches this op.
"""

import itertools

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

# "both" and "none" are the two that matter here: they select whether the writer's start barrier --
# where the hang parks -- is compiled in. The "staging" helper path is covered on t3000/blackhole and
# is orthogonal to the mesh shape.
NON_STAGING_MODES = [m for m in PERSISTENT_MODES if m != "staging"]

# None asks for every link the cluster axis has (4 on a Galaxy) rather than hardcoding 4: an over-ask
# is a TT_FATAL, not a skip, and the count a host reports for axis 0 is not guaranteed to match axis 1
# (see the dispatch-link note in reduce_scatter.cpp).
LINK_IDS = {1: "1link", None: "all_links"}


# The cross products below are the honest full matrix, but a few corners of it are not worth a
# Galaxy's time. They stay in the matrix as collection-time skips rather than being filtered out of
# the case list, so `-rs` prints why each one did not run. These are marks, not pytest.skip() calls
# in the body: a mark is evaluated before fixtures, so a skipped case never brings up the mesh.
DEEPSEEK_SHAPES = [shape for shape, _ in RS_DIRECT_DEEPSEEK_SHAPES]


def _skip_reason(rs_input_shape, num_links, enable_trace, persistent_mode):
    """Why this combination is redundant, or None to run it."""
    multi_chunk = rs_input_shape in DEEPSEEK_SHAPES
    if num_links is None and not multi_chunk:
        return "num_links clamps back to 1 at this size -- identical to the 1link case"
    if multi_chunk and not enable_trace:
        return "the big shape is only interesting under trace, which is how the hang reproduces"
    if multi_chunk and persistent_mode != "none":
        return "op-allocated buffers ('none') are what compile the start barrier in at this size"
    return None


def _cases(
    group,
    *,
    use_submesh,
    cluster_axes,
    shapes,
    shape_ids,
    trace_cases,
    trace_ids,
    link_counts,
    persistent_modes,
):
    """Cross-product one group of cases, flattened into pytest.params with a readable id."""
    for (
        cluster_axis,
        (shape_and_dim, shape_id),
        (trace_case, trace_id),
        num_links,
        persistent_mode,
    ) in itertools.product(
        cluster_axes,
        zip(shapes, shape_ids),
        zip(trace_cases, trace_ids),
        link_counts,
        persistent_modes,
    ):
        rs_input_shape, dim = shape_and_dim
        enable_trace, num_iters = trace_case
        skip_reason = _skip_reason(rs_input_shape, num_links, enable_trace, persistent_mode)
        yield pytest.param(
            use_submesh,
            cluster_axis,
            num_links,
            rs_input_shape,
            dim,
            enable_trace,
            num_iters,
            persistent_mode,
            marks=[pytest.mark.skip(reason=skip_reason)] if skip_reason else [],
            id=f"{group}-axis{cluster_axis}-{LINK_IDS[num_links]}-{shape_id}-{trace_id}-{persistent_mode}",
        )


RS_DIRECT_CASES = [
    # One ring on a cluster-axis-0 submesh: the plain single-ring shape/trace matrix.
    *_cases(
        "submesh_ring",
        use_submesh=True,
        cluster_axes=[0],
        shapes=RS_DIRECT_SHAPES,
        shape_ids=RS_DIRECT_SHAPE_IDS,
        trace_cases=RS_DIRECT_TRACE_CASES,
        trace_ids=RS_DIRECT_TRACE_IDS,
        link_counts=[1],
        persistent_modes=NON_STAGING_MODES,
    ),
    # The whole 8x4 mesh, one ring per row/column of the other axis, all concurrent
    *_cases(
        "multi_ring",
        use_submesh=False,
        cluster_axes=[0, 1],
        shapes=RS_DIRECT_SHAPES[:1] + RS_DIRECT_DEEPSEEK_SHAPES,
        shape_ids=RS_DIRECT_SHAPE_IDS[:1] + RS_DIRECT_DEEPSEEK_SHAPE_IDS,
        trace_cases=RS_DIRECT_TRACE_CASES,
        trace_ids=RS_DIRECT_TRACE_IDS,
        link_counts=[1, None],
        persistent_modes=NON_STAGING_MODES,
    ),
]


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "use_submesh, cluster_axis, num_links, rs_input_shape, dim, enable_trace, num_iters, persistent_mode",
    RS_DIRECT_CASES,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}],
    indirect=True,
    ids=["fabric_ring"],
)
def test_reduce_scatter_minimal_direct(
    mesh_device,
    use_submesh,
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

    device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1))) if use_submesh else mesh_device

    run_reduce_scatter_minimal_direct_impl(
        device,
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
