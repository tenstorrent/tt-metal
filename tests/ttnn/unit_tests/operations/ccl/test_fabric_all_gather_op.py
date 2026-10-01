# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.fabric_all_gather: equivalence with ttnn.experimental.high_bw_all_gather.

Every case runs both ops on the same input into outputs pre-filled with the same random bytes, and requires the two
outputs to be bit-identical on every chip -- so fabric_all_gather matches high_bw_all_gather's semantics exactly
(slot select, partial extent with untouched bytes outside it, metadata forms, sub-devices, external semaphores),
not just a gather. The cases mirror the GLM prefill call sites (models/demos/deepseek_v3_d_p): TILE width gathers
(dim 3) and a dim-1 gather over the TP axis, and the full-mesh ROW_MAJOR sparse-KV prefix gather (dim 2, cache slots,
ND-sharded DRAM input).

Runs on any Blackhole mesh: QuietBox (2x2), LoudBox (2x4) or Galaxy (8x4). FABRIC_ALL_GATHER_ROWS scales the per-chip rows.
"""

import os

import pytest
import torch
import ttnn

# Compare with high_bw_all_gather too (hardware); under tt-emule only the torch reference (the emulator does not
# implement the fabric mesh API high_bw_all_gather's kernels use).
EMULE = bool(os.environ.get("TT_METAL_EMULE_MODE"))
COMPARE_HIGH_BW = os.environ.get("FABRIC_ALL_GATHER_COMPARE_HIGH_BW", "0" if EMULE else "1") == "1"
ROWS = int(os.environ.get("FABRIC_ALL_GATHER_ROWS", "640"))  # per-chip tokens, as GLM's 5k chunk over SP=8
KV_ROWS = int(os.environ.get("FABRIC_ALL_GATHER_KV_ROWS", "480"))  # per-chip sparse-KV cache rows (the padded 15k job)
SLOTS = 4


def _router(payload):
    cfg = ttnn.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = payload
    return cfg


def _device_params(fabric_config, payload):
    return {
        "fabric_config": fabric_config,
        "fabric_router_config": _router(payload),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "l1_small_size": 2048,
        "trace_region_size": 64 * 1024 * 1024,  # the traced metadata case captures the op
    }


def _system_mesh():
    n = ttnn.get_num_devices()
    return {4: (2, 2), 8: (2, 4), 32: (8, 4)}.get(n, (1, n))


_FABRICS = [
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY, 6144), id="torus_xy_6k"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D, 6144), id="fabric_2d_6k"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY, 14336), id="torus_xy_14k"),
]


def _random(shape, dtype):
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31 - 1, shape, dtype=torch.int32)
    return torch.randn(shape, dtype=torch.bfloat16)


def _device_tensor(mesh_device, host, dtype, layout, mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=mapper,
    )


def _prefilled_pair(mesh_device, shape, dtype, layout):
    """Two outputs with the same random contents (so untouched bytes must also agree), and that content."""
    fill = _random(shape, dtype)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    return (
        _device_tensor(mesh_device, fill, dtype, layout, rep),
        _device_tensor(mesh_device, fill, dtype, layout, rep),
        fill,
    )


def _check(new_out, expected, ref_out, what):
    """new_out must equal the torch reference on every chip (and high_bw_all_gather's output, if it ran)."""
    news = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(new_out)]
    for i, got in enumerate(news):
        if not torch.equal(got.reshape(expected.shape).to(expected.dtype), expected):
            bad = (got.reshape(expected.shape).to(expected.dtype) != expected).nonzero()
            pytest.fail(f"{what}: chip {i} differs from the reference at {len(bad)} elements, first {bad[0].tolist()}")
    if COMPARE_HIGH_BW:
        for i, (got, ref) in enumerate(zip(news, (ttnn.to_torch(t) for t in ttnn.get_device_tensors(ref_out)))):
            assert torch.equal(got, ref), f"{what}: chip {i} differs from high_bw_all_gather"


def _both(mesh_device, inp, ref_out, new_out, **kw):
    if COMPARE_HIGH_BW:
        ttnn.experimental.high_bw_all_gather(inp, output_tensor=ref_out, **kw)
    ttnn.experimental.fabric_all_gather(inp, output_tensor=new_out, **kw)
    ttnn.synchronize_device(mesh_device)


def _meta_scalar(mesh_device, value):
    return ttnn.from_torch(
        torch.tensor([[[[value]]]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )


def _kv_nd_memory_config(mesh_device, width):
    """The GLM sparse-KV cache layout: 32 consecutive rows per DRAM bank, round-robin over banks."""
    banks = mesh_device.dram_grid_size().x
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, 32, width],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


# (name, per-chip input shape, dim) of the TP-axis gathers, as the GLM call sites use them
_AXIS_CASES = [
    ("rms_stats", (1, 1, ROWS, 32), 3),
    ("q_a_latent", (1, 1, ROWS, 512), 3),
    ("indexer_k", (1, 1, ROWS, 32), 3),
    ("kv_stem", (1, 1, ROWS, 576), 1),
    ("rows", (1, 1, ROWS, 576), 2),
]


@pytest.mark.parametrize("device_params", _FABRICS, indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
@pytest.mark.parametrize("cluster_axis", [1, 0])
@pytest.mark.parametrize("case", _AXIS_CASES, ids=lambda c: c[0])
def test_axis_gather_matches_high_bw(mesh_device, cluster_axis, case):
    name, local_shape, dim = case
    G = mesh_device.shape[cluster_axis]
    if G < 2:
        pytest.skip(f"cluster_axis {cluster_axis} is a singleton on mesh {tuple(mesh_device.shape)}")
    torch.manual_seed(0)
    rows, cols = tuple(mesh_device.shape)
    global_shape = list(local_shape)
    global_shape[dim] *= G
    mapper = ttnn.ShardTensor2dMesh(
        mesh_device, dims=(dim, None) if cluster_axis == 0 else (None, dim), mesh_shape=(rows, cols)
    )
    host = _random(global_shape, ttnn.bfloat16)
    inp = _device_tensor(mesh_device, host, ttnn.bfloat16, ttnn.TILE_LAYOUT, mapper)
    out_shape = list(local_shape)
    out_shape[dim] *= G
    ref_out, new_out, _ = _prefilled_pair(mesh_device, out_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    for _ in range(2):  # a program-cache miss, then a hit
        _both(mesh_device, inp, ref_out, new_out, dim=dim, cluster_axis=cluster_axis, num_links=2)
        # the other mesh axis replicates, so every chip's gathered tensor is the whole host tensor
        _check(new_out, host, ref_out, f"{name} axis {cluster_axis}")


@pytest.mark.parametrize("device_params", _FABRICS, indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
@pytest.mark.parametrize("cluster_axis", [1, None])
@pytest.mark.parametrize("width", [8192, 16384], ids=["16KiB_rows", "32KiB_rows"])
@pytest.mark.parametrize("nd_sharded", [False, True], ids=["interleaved", "nd_sharded"])
def test_pages_larger_than_payload(mesh_device, cluster_axis, width, nd_sharded):
    """ROW_MAJOR rows larger than the fabric payload: every page is split over several packets (3 for 16 KiB at a 6 KiB
    payload, the last one partial). 32 KiB rows also exceed one NoC packet in the copy cores' ND-sharded conversion."""
    torch.manual_seed(2)
    rows = 64
    if cluster_axis is None:
        G = mesh_device.get_num_devices()
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    else:
        G = mesh_device.shape[cluster_axis]
        dims = (None, 2) if cluster_axis == 1 else (2, None)
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    host = _random((1, 1, rows * G, width), ttnn.bfloat16)
    mem = _kv_nd_memory_config(mesh_device, width) if nd_sharded else ttnn.DRAM_MEMORY_CONFIG
    inp = _device_tensor(mesh_device, host, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mapper, memory_config=mem)
    ref_out, new_out, _ = _prefilled_pair(mesh_device, (1, 1, rows * G, width), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    _both(mesh_device, inp, ref_out, new_out, dim=2, cluster_axis=cluster_axis, num_links=2)
    _check(
        new_out, host, ref_out, f"{width * 2 // 1024} KiB rows, cluster_axis {cluster_axis}, nd_sharded {nd_sharded}"
    )


@pytest.mark.parametrize("device_params", _FABRICS[:1], indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
def test_tile_extent_must_be_tile_aligned(mesh_device, expect_error):
    """A partial extent of a TILE gather dim that is not a whole number of tiles is rejected (as high_bw_all_gather
    does), not rounded to whole tiles."""
    G = mesh_device.shape[1]
    host = _random((1, 1, 64 * G, 64), ttnn.bfloat16)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 2))
    inp = _device_tensor(mesh_device, host, ttnn.bfloat16, ttnn.TILE_LAYOUT, mapper)
    _, new_out, _ = _prefilled_pair(mesh_device, (1, 1, 64 * G, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "tile alignment"):
        ttnn.experimental.fabric_all_gather(
            inp, dim=2, output_tensor=new_out, cluster_axis=1, num_links=2, gathered_dim_size=16 * G
        )


def _kv_setup(mesh_device, nd_sharded, dtype=ttnn.bfloat16):
    torch.manual_seed(1)
    G = mesh_device.get_num_devices()
    width = 576
    host = _random((SLOTS, 1, KV_ROWS * G, width), dtype)
    mem = _kv_nd_memory_config(mesh_device, width) if nd_sharded else ttnn.DRAM_MEMORY_CONFIG
    inp = _device_tensor(
        mesh_device, host, dtype, ttnn.ROW_MAJOR_LAYOUT, ttnn.ShardTensorToMesh(mesh_device, dim=2), memory_config=mem
    )
    ref_out, new_out, fill = _prefilled_pair(mesh_device, (1, 1, KV_ROWS * G, width), dtype, ttnn.ROW_MAJOR_LAYOUT)
    return inp, ref_out, new_out, G, host, fill


def _kv_expected(host, fill, G, slot, extent):
    """Rank g writes the first extent / G rows of its slot into its fixed KV_ROWS-row region; the rest is untouched."""
    out = fill.clone()
    active = extent // G
    for g in range(G):
        out[0, 0, g * KV_ROWS : g * KV_ROWS + active] = host[slot, 0, g * KV_ROWS : g * KV_ROWS + active]
    return out


@pytest.mark.parametrize("device_params", _FABRICS, indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
@pytest.mark.parametrize("nd_sharded", [True, False], ids=["nd_sharded", "interleaved"])
def test_full_mesh_kv_prefix_matches_high_bw(mesh_device, nd_sharded):
    """The sparse-KV prefix gather: cluster_axis=None, a cache slot, and a growing block-cyclic extent."""
    inp, ref_out, new_out, G, host, fill = _kv_setup(mesh_device, nd_sharded)
    slab = 32 * G  # one 32-row block per chip
    full = KV_ROWS * G
    for slot, extent in [(0, full), (1, slab), (2, 3 * slab), (1, full)]:
        extent = min(extent, full)
        _both(
            mesh_device,
            inp,
            ref_out,
            new_out,
            dim=2,
            cluster_axis=None,
            num_links=2,
            input_batch_index=slot,
            gathered_dim_size=extent,
        )
        # the reference accumulates: earlier (longer) calls left their rows in place
        fill = _kv_expected(host, fill, G, slot, extent)
        _check(new_out, fill, ref_out, f"kv prefix slot {slot} extent {extent}")


@pytest.mark.parametrize("device_params", _FABRICS, indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
@pytest.mark.parametrize("nd_sharded", [True, False], ids=["nd_sharded", "interleaved"])
def test_full_mesh_kv_prefix_on_top_two_rows(mesh_device, nd_sharded):
    """The sparse-MLA overlap region's KV gather is confined to the two worker rows below the Ethernet row
    (tt_ccl sparse-MLA overlap profiles). Every chip, whatever its harvesting and cabling, must fit its link workers
    and copy cores there. No sub-device manager, so this also runs under tt-emule."""
    inp, ref_out, new_out, G, host, fill = _kv_setup(mesh_device, nd_sharded)
    grid = mesh_device.compute_with_storage_grid_size()
    top_rows = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, 1))])
    slab = 32 * G
    full = KV_ROWS * G
    for slot, extent in [(0, full), (1, slab), (2, 3 * slab)]:
        _both(
            mesh_device,
            inp,
            ref_out,
            new_out,
            dim=2,
            cluster_axis=None,
            num_links=2,
            input_batch_index=slot,
            gathered_dim_size=extent,
            sub_core_grids=top_rows,
        )
        fill = _kv_expected(host, fill, G, slot, extent)
        _check(new_out, fill, ref_out, f"top-two-rows kv prefix slot {slot} extent {extent}")


@pytest.mark.skipif(EMULE, reason="trace capture / sub-device managers need fast dispatch (hardware)")
@pytest.mark.parametrize("device_params", _FABRICS[:1], indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
def test_full_mesh_kv_prefix_metadata_traced_matches_high_bw(mesh_device):
    """The trace-safe form: slot and extent read on device. Each op is captured ONCE; between replays only the user and
    prefix tensors change in place (different slots, growing and shrinking extents), so a replay that used the captured
    metadata would gather the wrong slot or extent."""
    inp, ref_out, new_out, G, host, fill = _kv_setup(mesh_device, nd_sharded=True)
    num_layers, layer_idx = 2, 1  # slot = user * num_layers + layer_idx
    slab = 32 * G
    user = _meta_scalar(mesh_device, 0)
    prefix = _meta_scalar(mesh_device, 0)

    def host_scalar(value):
        return ttnn.from_torch(
            torch.tensor([[[[value]]]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def issue(op, out):
        op(
            inp,
            dim=2,
            output_tensor=out,
            cluster_axis=None,
            num_links=2,
            input_batch_index_tensor=user,
            batch_slot_num_layers=num_layers,
            batch_slot_layer_idx=layer_idx,
            gathered_prefix_tensor=prefix,
            gathered_slab_global=slab,
        )

    ops = [(ttnn.experimental.fabric_all_gather, new_out)]
    if COMPARE_HIGH_BW:
        ops.insert(0, (ttnn.experimental.high_bw_all_gather, ref_out))
    traces = []
    for op, out in ops:
        issue(op, out)  # compile (user 0, start 0)
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        issue(op, out)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        traces.append(tid)
    ttnn.synchronize_device(mesh_device)
    fill = _kv_expected(host, fill, G, layer_idx, slab)  # the warm-up and capture calls
    try:
        for u, start in [(1, slab), (0, 3 * slab), (1, 0), (0, KV_ROWS * G - slab)]:
            ttnn.copy_host_to_device_tensor(host_scalar(u), user)
            ttnn.copy_host_to_device_tensor(host_scalar(start), prefix)
            for tid in traces:
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            extent = min(((start + slab + slab - 1) // slab) * slab, KV_ROWS * G)
            fill = _kv_expected(host, fill, G, u * num_layers + layer_idx, extent)
            _check(new_out, fill, ref_out, f"traced replay user {u} start {start}")
    finally:
        for tid in traces:
            ttnn.release_trace(mesh_device, tid)


@pytest.mark.skipif(EMULE, reason="trace capture / sub-device managers need fast dispatch (hardware)")
@pytest.mark.parametrize("device_params", _FABRICS[:1], indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
def test_full_mesh_kv_prefix_subdevice_external_semaphores(mesh_device):
    """The GLM overlap path: a 4-column sub-device strip and caller-owned L1_SMALL semaphores."""
    inp, ref_out, new_out, G, host, fill = _kv_setup(mesh_device, nd_sharded=True)
    grid = mesh_device.compute_with_storage_grid_size()
    x0 = grid.x - 4
    rest = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x0 - 1, grid.y - 1))])
    strip = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    sems = {
        op: [ttnn.create_global_semaphore(mesh_device, strip, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
        for op in ("ref", "new")
    }
    ttnn.synchronize_device(mesh_device)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([rest]), ttnn.SubDevice([strip])], 0)
    mesh_device.load_sub_device_manager(manager)
    sd = ttnn.SubDeviceId(1)
    try:
        for slot, extent in [(1, 32 * G), (2, KV_ROWS * G)]:
            kw = dict(
                dim=2,
                cluster_axis=None,
                num_links=2,
                subdevice_id=sd,
                sub_core_grids=strip,
                input_batch_index=slot,
                gathered_dim_size=extent,
            )
            if COMPARE_HIGH_BW:
                ttnn.experimental.high_bw_all_gather(
                    inp,
                    output_tensor=ref_out,
                    ready_semaphore=sems["ref"][0],
                    data_valid_semaphore=sems["ref"][1],
                    **kw,
                )
            ttnn.experimental.fabric_all_gather(
                inp, output_tensor=new_out, ready_semaphore=sems["new"][0], data_valid_semaphore=sems["new"][1], **kw
            )
            ttnn.synchronize_device(mesh_device, sub_device_ids=[sd])
            fill = _kv_expected(host, fill, G, slot, extent)
            _check(new_out, fill, ref_out, f"subdevice slot {slot} extent {extent}")
    finally:
        ttnn.synchronize_device(mesh_device)
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)


# ----------------------------------------------------------------------------------------------------------------------
# Host-side cost: the GLM notrace jobs issue 408 of these calls per chunk forward, so per-call host time adds up.
# ----------------------------------------------------------------------------------------------------------------------
HOST_CALLS = int(os.environ.get("FABRIC_ALL_GATHER_HOST_CALLS", "50"))


def _host_times(mesh_device, op, inp, out, **kw):
    """(first call incl. program build, median enqueue time of a cache-hit call), in microseconds."""
    import statistics
    import time

    t0 = time.perf_counter()
    op(inp, output_tensor=out, **kw)
    first = time.perf_counter() - t0
    ttnn.synchronize_device(mesh_device)
    samples = []
    for _ in range(HOST_CALLS):
        t0 = time.perf_counter()
        op(inp, output_tensor=out, **kw)
        samples.append(time.perf_counter() - t0)
    ttnn.synchronize_device(mesh_device)
    return first * 1e6, statistics.median(samples) * 1e6


@pytest.mark.skipif(EMULE, reason="host timing is only meaningful with fast dispatch on hardware")
@pytest.mark.parametrize("device_params", _FABRICS[:1], indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
def test_host_dispatch_cost(mesh_device):
    """Per-call host time of fabric_all_gather vs high_bw_all_gather, for the GLM call shapes."""
    rows, cols = tuple(mesh_device.shape)
    report = []
    # a TP-axis width gather (sites A-D) and the full-mesh sparse-KV prefix gather (site F)
    tp = cols if cols > 1 else rows
    axis = 1 if cols > 1 else 0
    host = _random((1, 1, ROWS, 32 * tp), ttnn.bfloat16)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3) if axis == 1 else (3, None), mesh_shape=(rows, cols))
    inp = _device_tensor(mesh_device, host, ttnn.bfloat16, ttnn.TILE_LAYOUT, mapper)
    kv_inp, _, _, G, _, _ = _kv_setup(mesh_device, nd_sharded=True)
    for name, op in [
        ("high_bw_all_gather", ttnn.experimental.high_bw_all_gather),
        ("fabric_all_gather", ttnn.experimental.fabric_all_gather),
    ]:
        out, _, _ = _prefilled_pair(mesh_device, (1, 1, ROWS, 32 * tp), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        first, hit = _host_times(mesh_device, op, inp, out, dim=3, cluster_axis=axis, num_links=2)
        report.append(f"{name:<20} width gather  first {first:10.0f} us   cache hit {hit:8.1f} us/call")
        kv_out, _, _ = _prefilled_pair(mesh_device, (1, 1, KV_ROWS * G, 576), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        first, hit = _host_times(
            mesh_device,
            op,
            kv_inp,
            kv_out,
            dim=2,
            cluster_axis=None,
            num_links=2,
            input_batch_index=1,
            gathered_dim_size=32 * G,
        )
        report.append(f"{name:<20} kv prefix     first {first:10.0f} us   cache hit {hit:8.1f} us/call")
    print("\nFABRIC_ALL_GATHER_HOST\n" + "\n".join(report), flush=True)
