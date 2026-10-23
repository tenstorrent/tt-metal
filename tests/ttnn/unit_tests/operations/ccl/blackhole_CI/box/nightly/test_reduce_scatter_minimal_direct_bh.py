# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blackhole box coverage for ttnn.experimental.reduce_scatter_minimal_direct.

Shares run_reduce_scatter_minimal_direct_impl with the t3000 module -- the driver takes the device as
an argument, so the bh_1d_mesh_device fixture drops straight in.

The op reads its topology from the hardware instead of taking it as an argument, so it needs a ring
that wraps: the whole 1D mesh. num_devices therefore comes from the fixture (validated by
validate_test) rather than being parametrized, which also lets the same cases run on 4- and 8-device
boxes -- every scatter dim below is 8 pages wide, so it splits evenly either way. The 2D case takes
its mesh from the system mesh for the same reason, via bh_torus_mesh_device below.
"""

import pytest

import ttnn
from conftest import reset_fabric, set_fabric
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
from tests.scripts.common import get_updated_device_params
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


@pytest.fixture(scope="function")
def bh_torus_mesh_device(request, silicon_arch_name, silicon_arch_blackhole, device_params):
    """The 2D mesh this box actually presents under the requested torus fabric.

    Neither shared fixture can supply that. `mesh_device` opens the shape the caller parametrizes and
    `bh_2d_mesh_device` opens one hardcoded per device count (4x2 on eight devices) -- but the system
    mesh is a function of the fabric config, and a torus config is not FABRIC_2D. Auto-discovery keeps
    only what it can wire up, and in run 30838619957 a LoudBox that had opened an eight-device 2D mesh
    minutes earlier under FABRIC_2D reported a four-device 2x2 system mesh under TORUS_X. Asking for a
    shape the control plane did not discover aborts inside open_mesh_device, and by then the fixture
    has already latched the fabric
    config process-wide with no path to unlatch it -- which is how one unopenable mesh turned into 236
    errors, every later test in that job dying on the latch instead of on its own merits.

    So: latch the fabric first, ask what came back, open exactly that. Whether the result is big enough
    to scatter over is the test's business, not the fixture's.
    """
    request.node.pci_ids = ttnn.get_pcie_device_ids()

    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    fabric_manager = updated_device_params.pop("fabric_manager", None)
    fabric_router_config = updated_device_params.pop("fabric_router_config", None)
    set_fabric(fabric_config, reliability_mode, fabric_tensix_config, fabric_manager, fabric_router_config)

    try:
        # local_shape() rather than shape(): on a multi-host mesh only this host's devices are ours to
        # open. These boxes are single-host, where the two are the same.
        mesh_shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().local_shape()
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)
    except Exception:
        # Unlatch, or the rest of the session inherits a fabric config it never asked for.
        reset_fabric(fabric_config)
        raise

    try:
        yield mesh_device
    finally:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        reset_fabric(fabric_config)
        del mesh_device


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


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("rs_input_shape, dim", RS_DIRECT_SHAPES, ids=RS_DIRECT_SHAPE_IDS)
@pytest.mark.parametrize("persistent_mode", PERSISTENT_MODES)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1171456}],
    indirect=True,
    ids=["fabric_2d_torus_x"],
)
def test_reduce_scatter_minimal_direct_2d_torus(
    bh_torus_mesh_device,
    num_links,
    rs_input_shape,
    dim,
    rs_input_dtype,
    persistent_mode,
):
    """2D-fabric coverage. On 2D the op cannot route by hop count -- the fabric routes by DESTINATION
    NODE -- so the factory re-derives each send's direction from the control plane. This exercises that
    path; the 1D cases above cannot, since 1D routing is direction-agnostic by construction.

    Mesh orientation matters and is not interchangeable. axis_topology binds a mesh-VIEW axis index to a
    fabric dimension (axis 1 -> X), while the torus config wraps a PHYSICAL dimension, so a view whose
    axis 1 is not the box's X dimension either gets refused (TORUS_X reports that axis Linear) or -- the
    dangerous one -- reported as a Torus wrapping a dimension that does not physically wrap, which the
    factory catches with a single-hop neighbour check rather than hanging. Taking the view straight from
    the system mesh removes the choice: axis 1 of the discovered shape IS the box's X dimension, so
    cluster_axis=1 is correct on any box, and a box whose X dimension does not wrap under TORUS_X skips
    below on its resolved topology instead of running a mislabelled ring.
    """
    mesh_device = bh_torus_mesh_device
    mesh_shape = tuple(mesh_device.shape)
    cluster_axis = 1
    if len(mesh_shape) != 2:
        pytest.skip(f"2D-fabric test needs a 2D system mesh, got {mesh_shape}")
    num_devices = mesh_shape[cluster_axis]
    if num_devices < 2:
        # A one-device axis is not a ring to scatter over, and the split check below cannot catch it:
        # every dim divides by 1. skip_for_n_or_less_dev counts the whole mesh, so it passes here.
        pytest.skip(f"cluster axis {cluster_axis} of mesh {mesh_shape} has {num_devices} device")
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
        num_iters=2,
        enable_trace=False,
        cluster_axis=cluster_axis,
        persistent_mode=persistent_mode,
    )
