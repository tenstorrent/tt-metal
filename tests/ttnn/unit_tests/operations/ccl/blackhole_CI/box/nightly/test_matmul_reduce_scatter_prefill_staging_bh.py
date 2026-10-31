# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.matmul_reduce_scatter_prefill with a caller-held staging_tensor, on a Blackhole box (1x4 ring) or
a Blackhole Galaxy (8x4 mesh, every 4- and 8-device ring at once); each machine runs the mesh shape it has.

The caller allocates the staging once with ttnn.experimental.matmul_reduce_scatter_prefill_staging and passes it to
every call: the op then allocates nothing per call, and one staging sized for the largest shape serves every smaller
one.
"""

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_matmul_reduce_scatter_prefill_bh import (
    check,
    quantize_like_device,
    random_stacked,
    reference,
    ring_device_params,
    to_mesh,
)
from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0

# (a_shape, w_shape, weight_dtype): an unchunked plan and a chunked one, which need different staging sizes
SHAPES = {
    "kimi_k27_o_proj": ((1, 1, 640, 2048), (2048, 7168), ttnn.bfloat8_b),
    "chunked_n": ((1, 1, 2048, 1024), (1024, 16384), ttnn.bfloat16),
}
MESHES = [(1, 4), (8, 4)]
MESH_IDS = ["box_1x4", "galaxy_8x4"]


def staging_device_params():
    return {**ring_device_params(), "trace_region_size": 1 << 20}


def op_kwargs(cluster_axis):
    return dict(
        cluster_axis=cluster_axis,
        scatter_dim=-1,
        topology=ttnn.Topology.Ring,
        num_links=2,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2),
    )


def skip_unless_group(mesh_device, cluster_axis):
    if tuple(mesh_device.shape)[cluster_axis] < 2:
        pytest.skip(f"axis {cluster_axis} of a {tuple(mesh_device.shape)} mesh has no group")


def operands(mesh_device, name, cluster_axis, seed):
    a_shape, w_shape, weight_dtype = SHAPES[name]
    mesh_shape = tuple(mesh_device.shape)
    a = random_stacked(mesh_shape, a_shape, seed=seed)
    w = quantize_like_device(random_stacked(mesh_shape, w_shape, seed=seed + 1, scale=w_shape[0] ** -0.5), weight_dtype)
    ta = to_mesh(a, mesh_device, ttnn.bfloat16)
    tw = to_mesh(w, mesh_device, weight_dtype)
    return ta, tw, reference(a, w, cluster_axis, -1), weight_dtype


def staging_rows(staging):
    return staging.memory_config().shard_spec.shape[0]


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [staging_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, ids=MESH_IDS, indirect=True)
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0", "axis1"])
@pytest.mark.parametrize("name", SHAPES.keys())
def test_staging_tensor(mesh_device, cluster_axis, name):
    """Repeated calls with one caller-held staging: correct every call, and one cached program."""
    skip_unless_group(mesh_device, cluster_axis)
    kwargs = op_kwargs(cluster_axis)
    ta, tw, expected, weight_dtype = operands(mesh_device, name, cluster_axis, seed=0)
    staging = ttnn.experimental.matmul_reduce_scatter_prefill_staging(ta, tw, **kwargs)
    entries = None
    for _ in range(3):
        out = ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, staging_tensor=staging, **kwargs)
        ttnn.synchronize_device(mesh_device)
        check(out, expected, mesh_device, weight_dtype)
        ttnn.deallocate(out)
        if entries is None:
            entries = mesh_device.num_program_cache_entries()
    assert mesh_device.num_program_cache_entries() == entries, "a repeated call built a new program"
    ttnn.deallocate(staging)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [staging_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, ids=MESH_IDS, indirect=True)
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0", "axis1"])
def test_staging_shared_across_shapes(mesh_device, cluster_axis):
    """One staging sized for the larger plan serves both shapes, in either order, alongside the op's own staging."""
    skip_unless_group(mesh_device, cluster_axis)
    kwargs = op_kwargs(cluster_axis)
    small = operands(mesh_device, "kimi_k27_o_proj", cluster_axis, seed=0)
    large = operands(mesh_device, "chunked_n", cluster_axis, seed=10)
    staging = ttnn.experimental.matmul_reduce_scatter_prefill_staging(large[0], large[1], **kwargs)
    small_staging = ttnn.experimental.matmul_reduce_scatter_prefill_staging(small[0], small[1], **kwargs)
    assert staging_rows(staging) > staging_rows(small_staging), "the chunked plan should need the larger staging"
    ttnn.deallocate(small_staging)
    for ta, tw, expected, weight_dtype in (small, large, small):
        out = ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, staging_tensor=staging, **kwargs)
        ttnn.synchronize_device(mesh_device)
        check(out, expected, mesh_device, weight_dtype)
        ttnn.deallocate(out)
    # the op's own per-call staging afterwards (a staging still held would be L1 its plan does not see)
    ttnn.deallocate(staging)
    ta, tw, expected, weight_dtype = small
    out = ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, **kwargs)
    ttnn.synchronize_device(mesh_device)
    check(out, expected, mesh_device, weight_dtype)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [staging_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, ids=MESH_IDS, indirect=True)
def test_staging_traced(mesh_device):
    """Both shapes captured in one trace on one caller-held staging; new inputs copied in before each replay."""
    cluster_axis = 1
    kwargs = op_kwargs(cluster_axis)
    cases = [operands(mesh_device, name, cluster_axis, seed=10 * i) for i, name in enumerate(SHAPES)]
    staging = ttnn.experimental.matmul_reduce_scatter_prefill_staging(cases[1][0], cases[1][1], **kwargs)
    for ta, tw, _, _ in cases:  # compile both
        ttnn.deallocate(ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, staging_tensor=staging, **kwargs))
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = [
        ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, staging_tensor=staging, **kwargs)
        for ta, tw, _, _ in cases
    ]
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for seed in (20, 30):
            for i, (name, (ta, tw, _, weight_dtype)) in enumerate(zip(SHAPES, cases)):
                a_shape, w_shape, _ = SHAPES[name]
                mesh_shape = tuple(mesh_device.shape)
                a = random_stacked(mesh_shape, a_shape, seed=seed + i)
                w = quantize_like_device(
                    random_stacked(mesh_shape, w_shape, seed=seed + i + 1, scale=w_shape[0] ** -0.5), weight_dtype
                )
                ttnn.copy_host_to_device_tensor(to_mesh(a, mesh_device, ttnn.bfloat16, on_device=False), ta)
                ttnn.copy_host_to_device_tensor(to_mesh(w, mesh_device, weight_dtype, on_device=False), tw)
                cases[i] = (ta, tw, reference(a, w, cluster_axis, -1), weight_dtype)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            for out, (_, _, expected, weight_dtype) in zip(outs, cases):
                check(out, expected, mesh_device, weight_dtype)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(staging)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [staging_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", MESHES, ids=MESH_IDS, indirect=True)
@pytest.mark.parametrize("kind", ["too_small", "dram", "one_device"])
def test_staging_refusals(mesh_device, expect_error, kind):
    """A staging the call cannot use is refused before anything runs."""
    kwargs = op_kwargs(1)
    small = operands(mesh_device, "kimi_k27_o_proj", 1, seed=0)
    large = operands(mesh_device, "chunked_n", 1, seed=10)
    staging = ttnn.experimental.matmul_reduce_scatter_prefill_staging(small[0], small[1], **kwargs)
    ta, tw = small[0], small[1]
    if kind == "too_small":
        ta, tw = large[0], large[1]
    elif kind == "dram":
        staging = ttnn.to_memory_config(staging, ttnn.DRAM_MEMORY_CONFIG)
    else:
        staging = ttnn.get_device_tensors(staging)[0]
    with expect_error(RuntimeError, "staging_tensor"):
        ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, staging_tensor=staging, **kwargs)
