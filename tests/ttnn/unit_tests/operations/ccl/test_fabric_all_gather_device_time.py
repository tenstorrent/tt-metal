# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device performance of ttnn.experimental.fabric_all_gather, with high_bw_all_gather as the reference.

Timing: each call pattern is captured once as a trace of CALLS back-to-back calls and replayed REPLAYS times; the
median replay time / CALLS is the time per call (host dispatch excluded, the fence between calls included).

Every result line (prefix FABRIC_ALL_GATHER_PERF) reports
  in            bytes one chip receives per second: (G - 1) * shard bytes / time
  busiest link  bytes per second on the busiest link direction, from the op's schedule (_busiest_link_shards),
                and its share of the per-direction link peak (48.5 GB/s QuietBox, 27 GB/s Galaxy;
                FABRIC_ALL_GATHER_LINK_PEAK_GBPS overrides).

Tests
  test_device_time                 mixed shapes, QuietBox-sized
  test_kv_prefix_device_time       GLM KV cache gather, ND-sharded vs interleaved input
  test_kv_gather_on_overlap_strip  the sparse-MLA overlap region standalone (gather / top-k / both)
  test_kv_gather_payload_sweep     GLM KV cache gather across fabric payload sizes
  test_glm_kv_gather               GLM KV cache gather, Galaxy-sized shards: cache format x context x placement
  test_glm_kv_gather_traced_prefix the production call: slot + growing extent read on device, under trace
  test_glm_overlap_window          gather on the top two rows vs top-k on the rest, at the Galaxy's per-chip Q
  test_tile_full_mesh              16384 x 576 bf16 TILE per chip, full mesh, 14 KiB (the example's 177 GB/s setup)
"""

import os
import statistics
import time

import pytest
import torch
import ttnn

CALLS = int(os.environ.get("FABRIC_ALL_GATHER_TIME_CALLS", "20"))
REPLAYS = int(os.environ.get("FABRIC_ALL_GATHER_TIME_REPLAYS", "5"))
OPS = os.environ.get("FABRIC_ALL_GATHER_TIME_OPS", "fabric,high_bw").split(",")
NUM_LINKS = 2
OP_FNS = {"fabric": ttnn.experimental.fabric_all_gather, "high_bw": ttnn.experimental.high_bw_all_gather}
GLM_PAYLOAD = 6144  # GLM53Config.FABRIC_PAYLOAD_SIZE (tied to the MoE token migration)
GALAXY_CHIPS = 32  # the GLM KV cache is deduped over SP x TP = 32 chips on the Galaxy


def _device_params(fabric_config, payload):
    cfg = ttnn.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = payload
    return {
        "fabric_config": fabric_config,
        "fabric_router_config": cfg,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "l1_small_size": 2048,
        "trace_region_size": 64 * 1024 * 1024,
    }


def _system_mesh():
    n = ttnn.get_num_devices()
    return {4: (2, 2), 8: (2, 4), 32: (8, 4)}.get(n, (1, n))


def _torus_param(payload):
    return pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY, payload), id=f"torus_xy_{payload}")


_FABRICS = [
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D, 14336), id="fabric_2d_14k"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D, 6144), id="fabric_2d_6k"),
]


# ---------------------------------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------------------------------


def _link_peak_gbps(num_devices):
    env = os.environ.get("FABRIC_ALL_GATHER_LINK_PEAK_GBPS")
    if env:
        return float(env)
    return 27.0 if num_devices >= 32 else 48.5


def _busiest_link_shards(op, mesh_shape, cluster_axis, fabric_config, num_links=NUM_LINKS):
    """Shards the busiest link direction carries in one call.

    Assumes a fully wired mesh, where a full-mesh snake closes iff the mesh is a torus or has an even side (as
    resolve_mesh_ring_plan does). fabric_all_gather: closed even rings are balanced (the opposite shard goes half each way): G/2 - 1/2 shards per
    direction; a full-mesh gather on a torus with both sides >= 3 runs two edge-disjoint Hamiltonian cycles that each
    carry half of every shard. high_bw_all_gather: its rings are not balanced, G/2 shards on the busier direction. An
    open line carries G - 1 shards on its middle links. Split over num_links links.
    """
    rows, cols = mesh_shape
    torus = fabric_config == ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    if cluster_axis is None:
        G = rows * cols
        closed = torus or rows % 2 == 0 or cols % 2 == 0  # a snake closes on a torus or an even side
        cycles = 2 if (op == "fabric" and torus and rows >= 3 and cols >= 3) else 1
    else:
        G = mesh_shape[cluster_axis]
        closed = torus and G > 2
        cycles = 1
    if not closed:
        shards = G - 1
    elif op == "fabric":
        shards = G / 2 - 0.5 if G % 2 == 0 else (G - 1) / 2
    else:
        shards = G / 2
    return shards / cycles / num_links


def _report(tag, op, seconds, G, shard_bytes, busiest_shards, peak):
    agg = (G - 1) * shard_bytes / seconds / 1e9
    link = busiest_shards * shard_bytes / seconds / 1e9
    print(
        f"\nFABRIC_ALL_GATHER_PERF {tag:<44} {op:<8} G={G:<2} {seconds * 1e6:9.1f} us   in {agg:7.1f} GB/s/chip   "
        f"busiest link {link:5.1f} GB/s ({100 * link / peak:3.0f}% of {peak:g})",
        flush=True,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------


def _time_per_call(mesh_device, issue):
    issue()  # compile + program cache
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(CALLS):
        issue()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # warm
    samples = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) / CALLS)
    ttnn.release_trace(mesh_device, tid)
    return statistics.median(samples)


def _time_replays(mesh_device, issue, replays=10):
    """Median wall time of one blocking trace replay of `issue` (one iteration per replay: nothing overlaps across
    iterations, like the model's overlap region, which joins before the next layer)."""
    issue()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    issue()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    samples = []
    for _ in range(replays):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        samples.append(time.perf_counter() - t0)
    ttnn.release_trace(mesh_device, tid)
    return statistics.median(samples)


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


# GLM KV cache formats: (dtype, row width in elements, row bytes). scaled-fp8 rows pack 512 fp8 latents, 4 fp32 scales
# and 64 bf16 RoPE values (kv_cache_utils.MlaKvCacheGeometry.packed_row_bytes).
_KV_FORMATS = {
    "bf16": (ttnn.bfloat16, 576, 1152),
    "scaled_fp8": (ttnn.fp8_e4m3, 656, 656),
}


def _on_device(mesh_device, host, dtype, layout, memory_config, mesh_mapper):
    """from_torch; fp8 goes in as interleaved bf16, is typecast on device and then moved to memory_config (from_torch
    cannot pack fp8 into every layout, typecast cannot write ND-sharded row-major pages)."""
    if dtype != ttnn.fp8_e4m3:
        return ttnn.from_torch(
            host, dtype=dtype, layout=layout, device=mesh_device, memory_config=memory_config, mesh_mapper=mesh_mapper
        )
    tensor = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    tensor = ttnn.typecast(tensor, dtype)
    return tensor if memory_config == ttnn.DRAM_MEMORY_CONFIG else ttnn.to_memory_config(tensor, memory_config)


def _kv_cache(mesh_device, rows_per_chip, fmt, nd_sharded=True, slots=2):
    """[slots, 1, rows * G, width] cache, sharded over the mesh along dim 2 (every chip owns rows_per_chip rows)."""
    G = mesh_device.get_num_devices()
    dtype, width, _ = _KV_FORMATS[fmt]
    return _on_device(
        mesh_device,
        torch.randn((slots, 1, rows_per_chip * G, width), dtype=torch.bfloat16),
        dtype,
        ttnn.ROW_MAJOR_LAYOUT,
        _kv_nd_memory_config(mesh_device, width) if nd_sharded else ttnn.DRAM_MEMORY_CONFIG,
        ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )


def _replicated_output(mesh_device, shape, dtype, layout):
    return _on_device(
        mesh_device,
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype,
        layout,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _top_rows(mesh_device, rows=2):
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, rows - 1))])


def _fabric_config(request):
    return request.node.callspec.params["device_params"]["fabric_config"]


def _payload_id(request):
    return request.node.callspec.id.split("-")[-1]


# ---------------------------------------------------------------------------------------------------------------------
# QuietBox-sized
# ---------------------------------------------------------------------------------------------------------------------

# (name, per-chip shape, layout, dim, cluster_axis)
_CASES = [
    ("full_mesh_18MiB", (1, 1, 16384, 576), ttnn.TILE_LAYOUT, 2, None),
    ("full_mesh_2MiB", (1, 1, 2048, 576), ttnn.TILE_LAYOUT, 2, None),
    ("rms_stats", (1, 1, 640, 32), ttnn.TILE_LAYOUT, 3, 1),
    ("q_a_latent", (1, 1, 640, 512), ttnn.TILE_LAYOUT, 3, 1),
    ("kv_stem", (1, 1, 640, 576), ttnn.TILE_LAYOUT, 1, 1),
    ("kv_prefix_rm", (1, 1, 480, 576), ttnn.ROW_MAJOR_LAYOUT, 2, None),
]


@pytest.mark.parametrize("device_params", _FABRICS, indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c[0])
def test_device_time(mesh_device, case, request):
    name, local_shape, layout, dim, cluster_axis = case
    rows, cols = tuple(mesh_device.shape)
    G = rows * cols if cluster_axis is None else mesh_device.shape[cluster_axis]
    torch.manual_seed(0)
    global_shape = list(local_shape)
    global_shape[dim] *= G
    if cluster_axis is None:
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=dim)
    else:
        mapper = ttnn.ShardTensor2dMesh(
            mesh_device, dims=(dim, None) if cluster_axis == 0 else (None, dim), mesh_shape=(rows, cols)
        )
    inp = ttnn.from_torch(
        torch.randn(global_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    out_shape = list(local_shape)
    out_shape[dim] *= G
    shard_bytes = 2 * torch.Size(local_shape).numel()  # bf16, no tile padding in these shapes
    peak = _link_peak_gbps(mesh_device.get_num_devices())
    for key in OPS:
        out = _replicated_output(mesh_device, out_shape, ttnn.bfloat16, layout)
        seconds = _time_per_call(
            mesh_device,
            lambda: OP_FNS[key](inp, dim=dim, output_tensor=out, cluster_axis=cluster_axis, num_links=NUM_LINKS),
        )
        busiest = _busiest_link_shards(key, (rows, cols), cluster_axis, _fabric_config(request))
        _report(f"{name} {_payload_id(request)}", key, seconds, G, shard_bytes, busiest, peak)


@pytest.mark.parametrize(
    "device_params",
    _FABRICS if os.environ.get("FABRIC_ALL_GATHER_TIME_KV_ALL_PAYLOADS") else _FABRICS[1:],
    indirect=True,
)  # GLM runs 6 KiB
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
@pytest.mark.parametrize("input_layout", ["nd_sharded", "interleaved"])
@pytest.mark.parametrize(
    "kv_rows", [int(r) for r in os.environ.get("FABRIC_ALL_GATHER_TIME_KV_ROWS", "480,1728").split(",")]
)
def test_kv_prefix_device_time(mesh_device, kv_rows, input_layout, request):
    """The GLM sparse-KV prefix gather: [slots, 1, rows, 576] bf16 ROW_MAJOR cache, slot 1 selected, full extent."""
    G = mesh_device.get_num_devices()
    inp = _kv_cache(mesh_device, kv_rows, "bf16", nd_sharded=input_layout == "nd_sharded")
    peak = _link_peak_gbps(G)
    for key in OPS:
        out = _replicated_output(mesh_device, (1, 1, kv_rows * G, 576), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        seconds = _time_per_call(
            mesh_device,
            lambda: OP_FNS[key](
                inp,
                dim=2,
                output_tensor=out,
                cluster_axis=None,
                num_links=NUM_LINKS,
                input_batch_index=1,
                gathered_dim_size=kv_rows * G,
            ),
        )
        busiest = _busiest_link_shards(key, tuple(mesh_device.shape), None, _fabric_config(request))
        _report(f"kv{kv_rows} {_payload_id(request)} {input_layout}", key, seconds, G, kv_rows * 1152, busiest, peak)


def _overlap_window(
    mesh_device, request, kv_rows, fmt, nd_sharded, q_local, topk_keys, topk_grid, gather_grid, tag, exploratory=False
):
    """The KV gather on the full grid, on gather_grid alone, top-k on topk_grid alone, and both concurrently (host time
    per blocking replay). Only an exploratory placement may fail to run (reported, not raised)."""
    G = mesh_device.get_num_devices()
    dtype, width, row_bytes = _KV_FORMATS[fmt]
    inp = _kv_cache(mesh_device, kv_rows, fmt, nd_sharded=nd_sharded)
    scores = ttnn.from_torch(
        torch.randn((1, 1, q_local, topk_keys), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    peak = _link_peak_gbps(G)
    for key in OPS:
        out = _replicated_output(mesh_device, (1, 1, kv_rows * G, width), dtype, ttnn.ROW_MAJOR_LAYOUT)
        gather_kw = dict(
            dim=2,
            output_tensor=out,
            cluster_axis=None,
            num_links=NUM_LINKS,
            input_batch_index=1,
            gathered_dim_size=kv_rows * G,
        )
        busiest = _busiest_link_shards(key, tuple(mesh_device.shape), None, _fabric_config(request))
        full_grid = _time_replays(mesh_device, lambda: OP_FNS[key](inp, **gather_kw))
        sems = [ttnn.create_global_semaphore(mesh_device, gather_grid, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
        ttnn.synchronize_device(mesh_device)
        manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([topk_grid]), ttnn.SubDevice([gather_grid])], 0)
        mesh_device.load_sub_device_manager(manager)
        try:
            gather = lambda: OP_FNS[key](
                inp,
                **gather_kw,
                subdevice_id=ttnn.SubDeviceId(1),
                sub_core_grids=gather_grid,
                ready_semaphore=sems[0],
                data_valid_semaphore=sems[1],
            )
            topk = lambda: ttnn.experimental.topk_large_indices(
                scores, k=2048, subdevice_id=ttnn.SubDeviceId(0), sub_core_grids=topk_grid
            )

            def both():
                topk()
                gather()

            try:
                region_alone = _time_replays(mesh_device, gather)
                together = _time_replays(mesh_device, both)
            except RuntimeError as e:  # e.g. the op does not fit in an exploratory region
                if not exploratory:
                    raise
                ttnn.synchronize_device(mesh_device)
                print(f"\nFABRIC_ALL_GATHER_PERF {tag} {key} does not run: {str(e).splitlines()[0][:160]}", flush=True)
                continue
            topk_alone = _time_replays(mesh_device, topk)
        finally:
            ttnn.synchronize_device(mesh_device)
            mesh_device.clear_loaded_sub_device_manager()
            mesh_device.remove_sub_device_manager(manager)
        _report(f"{tag} full-grid", key, full_grid, G, kv_rows * row_bytes, busiest, peak)
        _report(f"{tag} region", key, region_alone, G, kv_rows * row_bytes, busiest, peak)
        print(
            f"\nFABRIC_ALL_GATHER_PERF {tag} {key} top-k ({topk_grid.num_cores()} cores, Q={q_local}, "
            f"{topk_keys} keys) {topk_alone * 1e6:.1f} us   both {together * 1e6:.1f} us",
            flush=True,
        )


@pytest.mark.parametrize("device_params", _FABRICS, indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
@pytest.mark.parametrize("region", ["cols8-10", "row0", "rows0-1"])
@pytest.mark.parametrize("input_layout", ["nd_sharded", "interleaved"])
@pytest.mark.parametrize("cache_tokens", [14080, 129280])  # QuietBox warm / long (sparse-MLA perf proxy)
def test_kv_gather_on_overlap_strip(mesh_device, cache_tokens, input_layout, region, request):
    """The sparse-MLA overlap region standalone, QuietBox proxy (Q = 320 query rows per chip): the KV gather on a
    region of the grid, top-k on the rest."""
    G = mesh_device.get_num_devices()
    grid = mesh_device.compute_with_storage_grid_size()
    if region == "cols8-10":  # the previous overlap profile: top-k on columns 0..7, the gather on the rest
        topk_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, grid.y - 1))])
        gather_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    else:  # the gather on the row(s) right below the Ethernet row, top-k on the rest
        rows = 1 if region == "row0" else 2
        gather_grid = _top_rows(mesh_device, rows)
        topk_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _overlap_window(
        mesh_device,
        request,
        cache_tokens // G,
        "bf16",
        input_layout == "nd_sharded",
        320,
        cache_tokens,
        topk_grid,
        gather_grid,
        f"cache{cache_tokens} {_payload_id(request)} {input_layout} {region}",
        exploratory=True,
    )


_SWEEP_PAYLOADS = [
    int(p)
    for p in os.environ.get("FABRIC_ALL_GATHER_SWEEP_PAYLOADS", "4608,6144,6912,9216,11520,13824,14336").split(",")
]
_SWEEP_KV_ROWS = [int(r) for r in os.environ.get("FABRIC_ALL_GATHER_SWEEP_KV_ROWS", "480,1760,4000,16160").split(",")]


@pytest.mark.parametrize("device_params", [_torus_param(p) for p in _SWEEP_PAYLOADS], indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
def test_kv_gather_payload_sweep(mesh_device, request):
    """The GLM sparse-KV cache gather across fabric payload sizes: ND-sharded bf16 cache, per-chip rows as on the
    Galaxy (context / 32 chips), on the full grid and on the two worker rows below the Ethernet row."""
    G = mesh_device.get_num_devices()
    top_rows = _top_rows(mesh_device)
    peak = _link_peak_gbps(G)
    for kv_rows in _SWEEP_KV_ROWS:
        inp = _kv_cache(mesh_device, kv_rows, "bf16")
        for key in OPS:
            out = _replicated_output(mesh_device, (1, 1, kv_rows * G, 576), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            busiest = _busiest_link_shards(key, tuple(mesh_device.shape), None, _fabric_config(request))
            for placement, cores in (("full", {}), ("top2rows", {"sub_core_grids": top_rows})):
                seconds = _time_per_call(
                    mesh_device,
                    lambda: OP_FNS[key](
                        inp,
                        dim=2,
                        output_tensor=out,
                        cluster_axis=None,
                        num_links=NUM_LINKS,
                        input_batch_index=1,
                        gathered_dim_size=kv_rows * G,
                        **cores,
                    ),
                )
                tag = f"sweep {_payload_id(request)} rows{kv_rows} {placement}"
                _report(tag, key, seconds, G, kv_rows * 1152, busiest, peak)
            out.deallocate()
        inp.deallocate()


# ---------------------------------------------------------------------------------------------------------------------
# Galaxy-sized GLM KV cache gather (also runs on smaller meshes with the same per-chip shard sizes)
# ---------------------------------------------------------------------------------------------------------------------

# context tokens; per-chip cache rows = context / 32 (the Galaxy dedups the cache over all 32 chips)
_GLM_CONTEXTS = [
    int(c) for c in os.environ.get("FABRIC_ALL_GATHER_GLM_CONTEXTS", "15360,56320,128000,517120").split(",")
]
_GLM_PAYLOADS = [int(p) for p in os.environ.get("FABRIC_ALL_GATHER_GLM_PAYLOADS", "6144,9216,13824").split(",")]


@pytest.mark.parametrize("device_params", [_torus_param(p) for p in _GLM_PAYLOADS], indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
@pytest.mark.parametrize("fmt", list(_KV_FORMATS))
def test_glm_kv_gather(mesh_device, fmt, request):
    """The GLM sparse-KV cache gather (ND-sharded cache, one slot, full extent) at Galaxy per-chip shard sizes, on the
    two worker rows below the Ethernet row (the model's overlap region) and on the full grid. At GLM's payload
    (6144 B) every format and placement; at the other payloads bf16 on the top two rows only."""
    payload = int(_payload_id(request).split("_")[-1])
    if payload != GLM_PAYLOAD and fmt != "bf16":
        pytest.skip("payload what-if: bf16 only")
    G = mesh_device.get_num_devices()
    dtype, width, row_bytes = _KV_FORMATS[fmt]
    top_rows = _top_rows(mesh_device)
    placements = [("top2rows", {"sub_core_grids": top_rows})] + ([("full", {})] if payload == GLM_PAYLOAD else [])
    peak = _link_peak_gbps(G)
    for context in _GLM_CONTEXTS:
        kv_rows = context // GALAXY_CHIPS
        inp = _kv_cache(mesh_device, kv_rows, fmt)
        for key in OPS:
            out = _replicated_output(mesh_device, (1, 1, kv_rows * G, width), dtype, ttnn.ROW_MAJOR_LAYOUT)
            busiest = _busiest_link_shards(key, tuple(mesh_device.shape), None, _fabric_config(request))
            for placement, cores in placements:
                seconds = _time_per_call(
                    mesh_device,
                    lambda: OP_FNS[key](
                        inp,
                        dim=2,
                        output_tensor=out,
                        cluster_axis=None,
                        num_links=NUM_LINKS,
                        input_batch_index=1,
                        gathered_dim_size=kv_rows * G,
                        **cores,
                    ),
                )
                tag = f"glm {fmt} {payload} ctx{context / 1000:.0f}k {placement}"
                _report(tag, key, seconds, G, kv_rows * row_bytes, busiest, peak)
            out.deallocate()
        inp.deallocate()


def _meta_scalar(mesh_device, value):
    return ttnn.from_torch(
        torch.tensor([[[[value]]]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )


@pytest.mark.parametrize("device_params", [_torus_param(GLM_PAYLOAD)], indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
def test_glm_kv_gather_traced_prefix(mesh_device, request):
    """The production call: a 55k-token bf16 cache gathered as chunked prefill grows it (chunks 1, 6, 11 of 11), with
    the slot and the populated prefix read from device tensors (trace-safe path), on the top two rows."""
    G = mesh_device.get_num_devices()
    kv_rows = 56320 // GALAXY_CHIPS
    capacity = kv_rows * G
    slab = capacity // 11  # one chunk
    inp = _kv_cache(mesh_device, kv_rows, "bf16", slots=2)
    user, layers, layer_idx = _meta_scalar(mesh_device, 0), 2, 1  # slot = user * layers + layer_idx = 1
    top_rows = _top_rows(mesh_device)
    peak = _link_peak_gbps(G)
    for key in OPS:
        out = _replicated_output(mesh_device, (1, 1, capacity, 576), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        busiest = _busiest_link_shards(key, tuple(mesh_device.shape), None, _fabric_config(request))
        for chunk in (1, 6, 11):
            prefix = _meta_scalar(mesh_device, (chunk - 1) * slab)  # populated start: extent = chunk * slab
            seconds = _time_per_call(
                mesh_device,
                lambda: OP_FNS[key](
                    inp,
                    dim=2,
                    output_tensor=out,
                    cluster_axis=None,
                    num_links=NUM_LINKS,
                    input_batch_index_tensor=user,
                    batch_slot_num_layers=layers,
                    batch_slot_layer_idx=layer_idx,
                    gathered_prefix_tensor=prefix,
                    gathered_slab_global=slab,
                    sub_core_grids=top_rows,
                ),
            )
            tag = f"glm traced prefix chunk {chunk}/11 ctx55k top2rows"
            _report(tag, key, seconds, G, chunk * slab // G * 1152, busiest, peak)


@pytest.mark.parametrize("device_params", [_torus_param(GLM_PAYLOAD)], indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
@pytest.mark.parametrize("context", [56320, 517120])
def test_glm_overlap_window(mesh_device, context, request):
    """The sparse-MLA overlap region with the model's profile: the KV gather on the top two worker rows, top-k on the
    rest, at the per-chip query rows of the mesh (a 5120-token chunk over all chips; 320 on meshes under 16 chips, the
    QuietBox sparse-MLA proxy) with the full context as top-k keys."""
    G = mesh_device.get_num_devices()
    grid = mesh_device.compute_with_storage_grid_size()
    q_local = 5120 // G if G >= 16 else 320
    topk_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _overlap_window(
        mesh_device,
        request,
        context // GALAXY_CHIPS,
        "bf16",
        True,
        q_local,
        context,
        topk_grid,
        _top_rows(mesh_device),
        f"glm overlap ctx{context / 1000:.0f}k",
    )


@pytest.mark.parametrize("device_params", [_torus_param(14336)], indirect=True)
@pytest.mark.parametrize("mesh_device", [_system_mesh()], indirect=True)
def test_tile_full_mesh(mesh_device, request):
    """16384 x 576 bf16 TILE, interleaved, per chip (18 MiB), full mesh, 14 KiB payload, 2 links: the setup of the
    fabric_all_gather example's 177 GB/s per chip (Galaxy, two Hamiltonian cycles)."""
    G = mesh_device.get_num_devices()
    inp = ttnn.from_torch(
        torch.randn((1, 1, 16384 * G, 576), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    peak = _link_peak_gbps(G)
    for key in OPS:
        out = _replicated_output(mesh_device, (1, 1, 16384 * G, 576), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        seconds = _time_per_call(
            mesh_device,
            lambda: OP_FNS[key](inp, dim=2, output_tensor=out, cluster_axis=None, num_links=NUM_LINKS),
        )
        busiest = _busiest_link_shards(key, tuple(mesh_device.shape), None, _fabric_config(request))
        _report("tile 18MiB/chip full mesh 14336", key, seconds, G, 16384 * 576 * 2, busiest, peak)
        out.deallocate()
