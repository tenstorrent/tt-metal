# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.high_bw_all_gather, bit-exact against a torch reference on every chip.

Every output is preallocated and pre-filled with random bytes, so a case also checks what the op must leave untouched.
Covered: TILE and ROW_MAJOR, interleaved and ND-sharded DRAM input into an interleaved DRAM output, every cluster_axis
(0, 1 and the whole mesh), program-cache hits on new buffers, slot select and partial extents, the trace-safe metadata
forms, caller-owned L1_SMALL semaphores (with and without a sub-device), worker-core restrictions, link counts, and
pages larger than the fabric payload.

Runs on any Blackhole mesh: QuietBox (2x2), LoudBox (2x4) or Galaxy (8x4), and under tt-emule except for the cases
that need fast dispatch (trace, sub-devices).
"""

import os

import pytest
import torch
import ttnn

EMULE = bool(os.environ.get("TT_METAL_EMULE_MODE"))
FAST_DISPATCH_ONLY = pytest.mark.skipif(EMULE, reason="trace capture / sub-device managers need fast dispatch")

SLOTS = 4  # cache slots of the slot-select input
KV_ROWS = 480  # per-chip rows of the slot-select input (15 tiles)
NUM_LINKS = 2


def _device_params(fabric_config, payload):
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = payload
    return {
        "fabric_config": fabric_config,
        "fabric_router_config": router,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "l1_small_size": 2048,
        "trace_region_size": 64 * 1024 * 1024,
    }


def _system_mesh():
    n = ttnn.get_num_devices()
    return {4: (2, 2), 8: (2, 4), 32: (8, 4)}.get(n, (1, n))


TORUS_6K = pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY, 6144), id="torus_xy_6k")
FABRIC_2D_6K = pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D, 6144), id="fabric_2d_6k")
TORUS_14K = pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY, 14336), id="torus_xy_14k")

MESH = pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
LAYOUTS = pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
INPUT_MEMORY = pytest.mark.parametrize("nd_sharded", [False, True], ids=["interleaved", "nd_sharded"])


def _random(shape, dtype):
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31 - 1, shape, dtype=torch.int32)
    return torch.randn(shape, dtype=torch.bfloat16)


def _nd_sharded_dram(mesh_device, width):
    """32 consecutive rows per DRAM bank, round-robin over the banks (the GLM sparse-KV cache layout)."""
    banks = mesh_device.dram_grid_size().x
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, 32, width],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def _group(mesh_device, cluster_axis):
    """(gather group size, mesh mapper factory sharding a tensor dim over the group, replicating the other axis)."""
    if cluster_axis is None:
        return mesh_device.get_num_devices(), lambda dim: ttnn.ShardTensorToMesh(mesh_device, dim=dim)
    dims = lambda dim: (dim, None) if cluster_axis == 0 else (None, dim)
    mesh_shape = tuple(mesh_device.shape)
    return mesh_device.shape[cluster_axis], lambda dim: ttnn.ShardTensor2dMesh(
        mesh_device, dims=dims(dim), mesh_shape=mesh_shape
    )


def _skip_singleton(mesh_device, cluster_axis):
    if cluster_axis is not None and mesh_device.shape[cluster_axis] < 2:
        pytest.skip(f"cluster_axis {cluster_axis} is a singleton on mesh {tuple(mesh_device.shape)}")


def _input(mesh_device, host, dtype, layout, mapper, nd_sharded, local_width=None):
    """local_width: each chip's innermost length (the host tensor's, unless the gather is along it)."""
    width = local_width or host.shape[-1]
    memory_config = _nd_sharded_dram(mesh_device, width) if nd_sharded else ttnn.DRAM_MEMORY_CONFIG
    return ttnn.from_torch(
        host, dtype=dtype, layout=layout, device=mesh_device, memory_config=memory_config, mesh_mapper=mapper
    )


def _output(mesh_device, shape, dtype, layout):
    """A preallocated interleaved DRAM output, every chip pre-filled with the same random contents."""
    fill = _random(shape, dtype)
    out = ttnn.from_torch(
        fill,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return out, fill


def _check(out, expected, what):
    for i, t in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(t).reshape(expected.shape).to(expected.dtype)
        if not torch.equal(got, expected):
            bad = (got != expected).nonzero()
            pytest.fail(f"{what}: chip {i} differs from the reference at {len(bad)} elements, first {bad[0].tolist()}")


def _gather(mesh_device, inp, out, **kw):
    kw.setdefault("num_links", NUM_LINKS)
    result = ttnn.experimental.high_bw_all_gather(inp, output_tensor=out, **kw)
    ttnn.synchronize_device(mesh_device)
    return result


# ----------------------------------------------------------------------------------------------------------------------
# Full gathers
# ----------------------------------------------------------------------------------------------------------------------
_GATHER_CASES = [
    pytest.param(ttnn.TILE_LAYOUT, 3, id="tile-dim3"),
    pytest.param(ttnn.TILE_LAYOUT, 2, id="tile-dim2"),
    pytest.param(ttnn.TILE_LAYOUT, 1, id="tile-dim1"),
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, 2, id="row_major-dim2"),
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, 1, id="row_major-dim1"),
]


@pytest.mark.parametrize("device_params", [TORUS_6K, FABRIC_2D_6K], indirect=True)
@MESH
@pytest.mark.parametrize("cluster_axis", [0, 1, None], ids=["axis0", "axis1", "full_mesh"])
@INPUT_MEMORY
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.uint32], ids=["bf16", "uint32"])
@pytest.mark.parametrize("layout,dim", _GATHER_CASES)
def test_gather(mesh_device, cluster_axis, nd_sharded, dtype, layout, dim):
    """Every chip ends with the whole gathered tensor. The second call is a program-cache hit on new input and output
    buffers, so the cached program must take its addresses from the call."""
    _skip_singleton(mesh_device, cluster_axis)
    torch.manual_seed(0)
    G, mapper = _group(mesh_device, cluster_axis)
    local = [1, 2, 64, 256]
    gathered = list(local)
    gathered[dim] *= G
    for call in range(2):
        host = _random(gathered, dtype)
        inp = _input(mesh_device, host, dtype, layout, mapper(dim), nd_sharded, local_width=local[-1])
        out, _ = _output(mesh_device, gathered, dtype, layout)
        result = _gather(mesh_device, inp, out, dim=dim, cluster_axis=cluster_axis)
        assert result.buffer_address() == out.buffer_address(), "the op must return the preallocated output"
        _check(out, host, f"call {call}")


@pytest.mark.parametrize("device_params", [TORUS_6K], indirect=True)
@MESH
@pytest.mark.parametrize("num_links", [1, None], ids=["1_link", "all_links"])
@pytest.mark.parametrize("cluster_axis", [1, None], ids=["axis1", "full_mesh"])
@LAYOUTS
def test_num_links(mesh_device, num_links, cluster_axis, layout):
    _skip_singleton(mesh_device, cluster_axis)
    torch.manual_seed(3)
    G, mapper = _group(mesh_device, cluster_axis)
    gathered = [1, 1, 64 * G, 512]
    host = _random(gathered, ttnn.bfloat16)
    inp = _input(mesh_device, host, ttnn.bfloat16, layout, mapper(2), nd_sharded=False)
    out, _ = _output(mesh_device, gathered, ttnn.bfloat16, layout)
    _gather(mesh_device, inp, out, dim=2, cluster_axis=cluster_axis, num_links=num_links)
    _check(out, host, f"num_links {num_links}")


@pytest.mark.parametrize("device_params", [TORUS_6K, TORUS_14K], indirect=True)
@MESH
@pytest.mark.parametrize("cluster_axis", [1, None], ids=["axis1", "full_mesh"])
@pytest.mark.parametrize("width", [8192, 16384], ids=["16KiB_rows", "32KiB_rows"])
@INPUT_MEMORY
def test_pages_larger_than_payload(mesh_device, cluster_axis, width, nd_sharded):
    """ROW_MAJOR rows larger than the fabric payload: every page is split over several packets, the last one partial.
    32 KiB rows also exceed one NoC packet in the copy cores' ND-sharded conversion."""
    _skip_singleton(mesh_device, cluster_axis)
    torch.manual_seed(2)
    G, mapper = _group(mesh_device, cluster_axis)
    gathered = [1, 1, 64 * G, width]
    host = _random(gathered, ttnn.bfloat16)
    inp = _input(mesh_device, host, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mapper(2), nd_sharded)
    out, _ = _output(mesh_device, gathered, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    _gather(mesh_device, inp, out, dim=2, cluster_axis=cluster_axis)
    _check(out, host, f"{width * 2 // 1024} KiB rows")


# ----------------------------------------------------------------------------------------------------------------------
# Slot select and partial extents: input [SLOTS, 1, KV_ROWS * G, 576], output [1, 1, KV_ROWS * G, 576]
# ----------------------------------------------------------------------------------------------------------------------
def _slot_setup(mesh_device, layout, nd_sharded, cluster_axis=None):
    torch.manual_seed(1)
    G, mapper = _group(mesh_device, cluster_axis)
    host = _random((SLOTS, 1, KV_ROWS * G, 576), ttnn.bfloat16)
    inp = _input(mesh_device, host, ttnn.bfloat16, layout, mapper(2), nd_sharded)
    out, fill = _output(mesh_device, (1, 1, KV_ROWS * G, 576), ttnn.bfloat16, layout)
    return inp, out, G, host, fill


def _slot_expected(expected, host, G, slot, extent):
    """Rank g writes the first extent / G rows of its slot into its fixed KV_ROWS-row region; the rest is untouched."""
    expected = expected.clone()
    active = extent // G
    for g in range(G):
        expected[0, 0, g * KV_ROWS : g * KV_ROWS + active] = host[slot, 0, g * KV_ROWS : g * KV_ROWS + active]
    return expected


@pytest.mark.parametrize("device_params", [TORUS_6K, FABRIC_2D_6K], indirect=True)
@MESH
@pytest.mark.parametrize("cluster_axis", [1, None], ids=["axis1", "full_mesh"])
@INPUT_MEMORY
@LAYOUTS
def test_slot_select_partial_extent(mesh_device, cluster_axis, nd_sharded, layout):
    """input_batch_index + gathered_dim_size: one slot, a block-cyclic prefix; the output's other rows keep their bytes
    across calls (earlier, longer calls leave their rows in place)."""
    _skip_singleton(mesh_device, cluster_axis)
    inp, out, G, host, expected = _slot_setup(mesh_device, layout, nd_sharded, cluster_axis)
    slab = 32 * G  # one 32-row block per chip
    full = KV_ROWS * G
    for slot, extent in [(0, full), (1, slab), (2, 3 * slab), (1, full)]:
        _gather(
            mesh_device,
            inp,
            out,
            dim=2,
            cluster_axis=cluster_axis,
            input_batch_index=slot,
            gathered_dim_size=extent,
        )
        expected = _slot_expected(expected, host, G, slot, extent)
        _check(out, expected, f"slot {slot} extent {extent}")


@pytest.mark.parametrize("device_params", [TORUS_6K], indirect=True)
@MESH
def test_row_major_innermost_dim_rejected(mesh_device, expect_error):
    """Pages move whole: a ROW_MAJOR gather along the innermost dim (every output row mixes ranks) is rejected."""
    G, mapper = _group(mesh_device, None)
    host = _random((1, 1, 64, 64 * G), ttnn.bfloat16)
    innermost = len(host.shape) - 1
    inp = _input(mesh_device, host, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mapper(innermost), nd_sharded=False)
    out, _ = _output(mesh_device, host.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(RuntimeError, "cannot gather a ROW_MAJOR tensor along its innermost dim"):
        ttnn.experimental.high_bw_all_gather(
            inp, dim=innermost, output_tensor=out, cluster_axis=None, num_links=NUM_LINKS
        )


@pytest.mark.parametrize("device_params", [TORUS_6K], indirect=True)
@MESH
def test_tile_extent_must_be_tile_aligned(mesh_device, expect_error):
    """A partial TILE extent that is not a whole number of tiles is rejected, not rounded."""
    G, mapper = _group(mesh_device, None)
    host = _random((1, 1, 64 * G, 64), ttnn.bfloat16)
    inp = _input(mesh_device, host, ttnn.bfloat16, ttnn.TILE_LAYOUT, mapper(2), nd_sharded=False)
    out, _ = _output(mesh_device, (1, 1, 64 * G, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "tile alignment"):
        ttnn.experimental.high_bw_all_gather(
            inp, dim=2, output_tensor=out, cluster_axis=None, num_links=NUM_LINKS, gathered_dim_size=16 * G
        )


@FAST_DISPATCH_ONLY
@pytest.mark.parametrize("device_params", [TORUS_6K], indirect=True)
@MESH
@LAYOUTS
def test_metadata_traced(mesh_device, layout):
    """The trace-safe forms: slot and extent read on device. Captured ONCE; between replays only the user and prefix
    tensors change in place, so a replay that used the captured values would gather the wrong slot or extent."""
    inp, out, G, host, expected = _slot_setup(mesh_device, layout, nd_sharded=True)
    num_layers, layer_idx = 2, 1  # slot = user * num_layers + layer_idx
    slab = 32 * G
    full = KV_ROWS * G

    def scalar(value, device=None):
        return ttnn.from_torch(
            torch.tensor([[[[value]]]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    user, prefix = scalar(0, mesh_device), scalar(0, mesh_device)

    def issue():
        ttnn.experimental.high_bw_all_gather(
            inp,
            dim=2,
            output_tensor=out,
            cluster_axis=None,
            num_links=NUM_LINKS,
            input_batch_index_tensor=user,
            batch_slot_num_layers=num_layers,
            batch_slot_layer_idx=layer_idx,
            gathered_prefix_tensor=prefix,
            gathered_slab_global=slab,
        )

    issue()  # compile (user 0, start 0)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    issue()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    expected = _slot_expected(expected, host, G, layer_idx, slab)
    try:
        for u, start in [(1, slab), (0, 3 * slab), (1, 0), (0, full - slab)]:
            ttnn.copy_host_to_device_tensor(scalar(u), user)
            ttnn.copy_host_to_device_tensor(scalar(start), prefix)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            extent = min(((start + 2 * slab - 1) // slab) * slab, full)
            expected = _slot_expected(expected, host, G, u * num_layers + layer_idx, extent)
            _check(out, expected, f"replay user {u} start {start}")
    finally:
        ttnn.release_trace(mesh_device, tid)


# ----------------------------------------------------------------------------------------------------------------------
# Worker cores and caller-owned semaphores
# ----------------------------------------------------------------------------------------------------------------------
def _full_grid(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])


@pytest.mark.parametrize("device_params", [TORUS_6K], indirect=True)
@MESH
@INPUT_MEMORY
@LAYOUTS
def test_sub_core_grids_top_two_rows(mesh_device, nd_sharded, layout):
    """Every chip, whatever its harvesting and cabling, fits its link workers and copy cores into the two worker rows
    below the Ethernet row (the GLM sparse-MLA overlap region)."""
    inp, out, G, host, expected = _slot_setup(mesh_device, layout, nd_sharded)
    grid = mesh_device.compute_with_storage_grid_size()
    top_rows = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, 1))])
    slab = 32 * G
    for slot, extent in [(0, KV_ROWS * G), (1, slab), (2, 3 * slab)]:
        _gather(
            mesh_device,
            inp,
            out,
            dim=2,
            cluster_axis=None,
            input_batch_index=slot,
            gathered_dim_size=extent,
            sub_core_grids=top_rows,
        )
        expected = _slot_expected(expected, host, G, slot, extent)
        _check(out, expected, f"slot {slot} extent {extent}")


@pytest.mark.parametrize("device_params", [TORUS_6K], indirect=True)
@MESH
@INPUT_MEMORY
@LAYOUTS
def test_external_semaphores_l1_small(mesh_device, nd_sharded, layout):
    """Caller-owned L1_SMALL semaphores on the whole worker grid, no sub-device. The op leaves them at zero, so the
    same pair serves back-to-back calls."""
    inp, out, G, host, expected = _slot_setup(mesh_device, layout, nd_sharded)
    grid = _full_grid(mesh_device)
    ready, data_valid = (ttnn.create_global_semaphore(mesh_device, grid, 0, ttnn.BufferType.L1_SMALL) for _ in range(2))
    ttnn.synchronize_device(mesh_device)
    slab = 32 * G
    for slot, extent in [(0, KV_ROWS * G), (1, slab), (2, 3 * slab), (3, KV_ROWS * G)]:
        _gather(
            mesh_device,
            inp,
            out,
            dim=2,
            cluster_axis=None,
            input_batch_index=slot,
            gathered_dim_size=extent,
            ready_semaphore=ready,
            data_valid_semaphore=data_valid,
        )
        expected = _slot_expected(expected, host, G, slot, extent)
        _check(out, expected, f"slot {slot} extent {extent}")


@pytest.mark.parametrize("device_params", [TORUS_6K], indirect=True)
@MESH
def test_external_semaphores_must_be_l1_small(mesh_device, expect_error):
    """With an L1_SMALL region on the device, semaphores in plain L1 are rejected."""
    inp, out, _, _, _ = _slot_setup(mesh_device, ttnn.ROW_MAJOR_LAYOUT, nd_sharded=False)
    grid = _full_grid(mesh_device)
    ready, data_valid = (ttnn.create_global_semaphore(mesh_device, grid, 0, ttnn.BufferType.L1) for _ in range(2))
    with expect_error(RuntimeError, "external semaphores must use buffer type"):
        ttnn.experimental.high_bw_all_gather(
            inp,
            dim=2,
            output_tensor=out,
            cluster_axis=None,
            num_links=NUM_LINKS,
            input_batch_index=0,
            ready_semaphore=ready,
            data_valid_semaphore=data_valid,
        )


@FAST_DISPATCH_ONLY
@pytest.mark.parametrize("device_params", [TORUS_6K], indirect=True)
@MESH
@LAYOUTS
def test_subdevice_external_semaphores(mesh_device, layout):
    """The GLM overlap path: the op confined to a 4-column sub-device strip, with caller-owned L1_SMALL semaphores."""
    inp, out, G, host, expected = _slot_setup(mesh_device, layout, nd_sharded=True)
    grid = mesh_device.compute_with_storage_grid_size()
    x0 = grid.x - 4
    rest = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x0 - 1, grid.y - 1))])
    strip = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    ready, data_valid = (
        ttnn.create_global_semaphore(mesh_device, strip, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)
    )
    ttnn.synchronize_device(mesh_device)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([rest]), ttnn.SubDevice([strip])], 0)
    mesh_device.load_sub_device_manager(manager)
    sd = ttnn.SubDeviceId(1)
    try:
        for slot, extent in [(1, 32 * G), (2, KV_ROWS * G)]:
            ttnn.experimental.high_bw_all_gather(
                inp,
                dim=2,
                output_tensor=out,
                cluster_axis=None,
                num_links=NUM_LINKS,
                subdevice_id=sd,
                sub_core_grids=strip,
                input_batch_index=slot,
                gathered_dim_size=extent,
                ready_semaphore=ready,
                data_valid_semaphore=data_valid,
            )
            ttnn.synchronize_device(mesh_device, sub_device_ids=[sd])
            expected = _slot_expected(expected, host, G, slot, extent)
            _check(out, expected, f"slot {slot} extent {extent}")
    finally:
        ttnn.synchronize_device(mesh_device)
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)
