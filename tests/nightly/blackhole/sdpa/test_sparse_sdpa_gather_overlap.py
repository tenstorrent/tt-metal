# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa on a restricted core grid, and sparse_sdpa overlapped with a next-layer KVPE prefetch.

GLM-5.3 prefill shape on the 8x4 Galaxy (SP8 x TP4, 64 heads resharded head->seq, TOPK 2048, latent 512 +
rope 64). The overlap case captures each configuration as a trace and times back-to-back replays on the host:
neither the realtime profiler nor the Tracy device profiler reports traced sub-device-1 gathers correctly.
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import golden, make_inputs, pcc

K_DIM, V_DIM, TOPK, H = 576, 512, 2048, 64
SP, TP = 8, 4

_TORUS_XY = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "l1_small_size": 2048,
    "trace_region_size": 64 * 1024 * 1024,
}


def _cols(first, last, rows=10):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(first, 0), ttnn.CoreCoord(last, rows - 1))])


# KVPE cache formats: production is scaled FP8 (512 fp8 latents | 4 fp32 block scales | 64 bf16 rope = 656 B rows).
_KV_FORMATS = {
    "fp8": (ttnn.transformer.SparseKVFormat.SCALED_FP8, ttnn.fp8_e4m3, 656),
    "bf16": (ttnn.transformer.SparseKVFormat.BF16, ttnn.bfloat16, K_DIM),
}
_KV_FMT = pytest.mark.parametrize("kv_fmt", list(_KV_FORMATS))


def _kv_cache(mesh_device, batch, tokens, kv_fmt, mesh_mapper, seed=0):
    """A KVPE cache in kv_fmt. Scaled FP8 is packed on device by the same ops the model's cache writer uses."""
    gen = torch.Generator().manual_seed(seed)
    common = dict(
        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mesh_mapper
    )
    if kv_fmt == "bf16":
        host = torch.randn(batch, 1, tokens, K_DIM, generator=gen).to(torch.bfloat16)
        return ttnn.from_torch(host, dtype=ttnn.bfloat16, **common)
    latent = torch.randn(batch, 1, tokens, V_DIM, generator=gen).to(torch.bfloat16)
    rope = torch.randn(batch, 1, tokens, K_DIM - V_DIM, generator=gen).to(torch.bfloat16)
    fp8, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
        ttnn.from_torch(latent, dtype=ttnn.bfloat16, **common), round_scale_to_power_of_two=True
    )
    return ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(
        fp8, scales, ttnn.from_torch(rope, dtype=ttnn.bfloat16, **common)
    )


def _zero_kv_buffer(mesh_device, tokens, kv_fmt):
    _, dtype, width = _KV_FORMATS[kv_fmt]
    zeros = ttnn.from_torch(
        torch.zeros(1, 1, tokens, width, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return zeros if dtype == ttnn.bfloat16 else ttnn.typecast(zeros, dtype)


def _sparse_sdpa(q, kv, idx, chunk_local, kv_fmt, k_chunk=128, **grid):
    return ttnn.transformer.sparse_sdpa(
        q,
        kv,
        idx,
        V_DIM,
        kv_format=_KV_FORMATS[kv_fmt][0],
        scale=K_DIM**-0.5,
        k_chunk_size=k_chunk,
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=chunk_local,
        block_cyclic_cache_tp_sharded=True,
        **grid,
    )


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("cols", [(0, 7), (2, 9), (0, 0)], ids=["80core", "offset80", "10core"])
def test_sparse_sdpa_sub_core_grids_matches_golden(mesh_device, cols):
    """A restricted grid only changes which core owns which tokens; results must not change."""
    Hs, S, T, topk, kc = 32, 96, 1024, 128, 64
    q, kv, indices = make_inputs(Hs, S, T, topk, K_DIM, lambda s: 1 + (s * 7) % topk, seed=3)
    ref = golden(q, kv, indices, K_DIM**-0.5, V_DIM)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_q = ttnn.from_torch(q.to(torch.bfloat16), dtype=ttnn.bfloat16, **common)
    tt_kv = ttnn.from_torch(kv.to(torch.bfloat16), dtype=ttnn.bfloat16, **common)
    tt_idx = ttnn.from_torch(indices.to(torch.int32), dtype=ttnn.uint32, **common)
    run = lambda **grid: ttnn.to_torch(
        ttnn.transformer.sparse_sdpa(
            tt_q, tt_kv, tt_idx, V_DIM, kv_format=ttnn.transformer.SparseKVFormat.BF16, k_chunk_size=kc, **grid
        )
    )
    full = run()
    restricted = run(sub_core_grids=_cols(*cols))
    assert torch.equal(full, restricted), "restricted grid changed the output"
    assert pcc(restricted, ref) >= 0.99


def _glm_inputs(mesh_device, chunk_global, cache_tokens, kv_fmt):
    """Per-chip GLM shapes after the head->seq reshard: q [1,64,chunk/(sp*tp),576], natural-position
    indices into a populated prefix, and a replicated block-cyclic gather buffer of cache_tokens rows."""
    s_local = chunk_global // (SP * TP)
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(1, H, s_local * SP * TP, K_DIM, generator=gen).to(torch.bfloat16)
    idx = torch.randint(0, cache_tokens, (1, 1, s_local * SP * TP, TOPK), generator=gen, dtype=torch.int32)
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)  # 32-way, matches the post-reshard sharding
    tt_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, mesh_mapper=shard_seq, **common)
    tt_idx = ttnn.from_torch(idx, dtype=ttnn.uint32, mesh_mapper=shard_seq, **common)
    tt_kv = _kv_cache(mesh_device, 1, cache_tokens, kv_fmt, ttnn.ReplicateTensorToMesh(mesh_device), seed=1)
    return tt_q, tt_kv, tt_idx, s_local * TP  # chunk_local = per-SP-row chunk


def _program_us(mesh_device, run, kernel_tag):
    _, programs = profile_realtime_program_merged(mesh_device, run)
    times = [p["duration_ns"] for p in programs.values() if any(kernel_tag in k for k in p["kernel_sources"])]
    return max(times) / 1e3 if times else float("nan")


_PERF = pytest.mark.skipif(os.environ.get("CI") == "true", reason="local production-shape perf qualification")
_CHUNKS = pytest.mark.parametrize("chunk_global", [5120, 2048], ids=["5k", "2k"])


@run_for_blackhole()
@_PERF
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@_CHUNKS
@_KV_FMT
def test_sparse_sdpa_glm_core_count_perf(mesh_device, chunk_global, kv_fmt):
    """sparse_sdpa alone on 120 / 100 / 80 cores at the GLM shape."""
    grid = mesh_device.compute_with_storage_grid_size()
    tt_q, tt_kv, tt_idx, chunk_local = _glm_inputs(mesh_device, chunk_global, chunk_global * 51, kv_fmt)
    rows = []
    for name, sub in [("full", None), ("100", _cols(0, 9, grid.y)), ("80", _cols(0, 7, grid.y))]:
        kw = {} if sub is None else {"sub_core_grids": sub}
        run = lambda: _sparse_sdpa(tt_q, tt_kv, tt_idx, chunk_local, kv_fmt, **kw)
        run()
        ttnn.synchronize_device(mesh_device)
        samples = sorted(_program_us(mesh_device, run, "sparse_sdpa") for _ in range(5))
        rows.append((name, samples[len(samples) // 2]))
    logger.info(f"sparse_sdpa GLM {chunk_global} {kv_fmt} per-program median us: {rows}")


@run_for_blackhole()
@_PERF
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@_CHUNKS
@pytest.mark.parametrize("prefix_slabs", [10, 40], ids=["prefix10slabs", "prefix40slabs"])
@_KV_FMT
def test_sparse_sdpa_overlap_with_kvpe_prefetch(mesh_device, chunk_global, prefix_slabs, kv_fmt):
    """sparse_sdpa (80 cores, sub-device 0) || next-layer prefix prefetch (40 cores, sub-device 1), then the
    resume, swept over the prefetch quota (PREFETCH_QUOTAS) against sparse_sdpa + one full gather."""
    grid = mesh_device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (12, 10):
        pytest.skip(f"needs the 12x10 Galaxy grid, got {grid}")
    sdpa_grid, gather_grid = _cols(0, 7), _cols(8, 11)
    n_dev = mesh_device.get_num_devices()
    slab_global = chunk_global
    cache_slabs = prefix_slabs + 1
    cache_tokens = cache_slabs * slab_global

    tt_q, tt_kv, tt_idx, chunk_local = _glm_inputs(mesh_device, chunk_global, cache_tokens, kv_fmt)
    # Two-layer KVPE cache striped over all 32 chips (the gather source) and the second gather buffer.
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cache = _kv_cache(mesh_device, 2, cache_tokens, kv_fmt, ttnn.ShardTensorToMesh(mesh_device, dim=2), seed=2)
    next_buffer = _zero_kv_buffer(mesh_device, cache_tokens, kv_fmt)

    def scalar(v):
        return ttnn.from_torch(
            torch.tensor([[[[v]]]], dtype=torch.int32),
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            **common,
        )

    slot, start, progress = scalar(0), scalar(prefix_slabs * slab_global), scalar(0)

    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([sdpa_grid]), ttnn.SubDevice([gather_grid])], 0)
    sems = [ttnn.create_global_semaphore(mesh_device, gather_grid, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
    ttnn.synchronize_device(mesh_device)

    def gather(**window):
        return ttnn.experimental.high_bw_all_gather(
            cache,
            dim=2,
            output_tensor=next_buffer,
            cluster_axis=None,
            subdevice_id=ttnn.SubDeviceId(1),
            sub_core_grids=gather_grid,
            num_links=2,
            input_batch_index_tensor=slot,
            batch_slot_num_layers=2,
            batch_slot_layer_idx=1,
            gathered_prefix_tensor=start,
            gathered_slab_global=slab_global,
            ready_semaphore=sems[0],
            data_valid_semaphore=sems[1],
            **window,
        )

    sdpa = lambda: _sparse_sdpa(
        tt_q, tt_kv, tt_idx, chunk_local, kv_fmt, subdevice_id=ttnn.SubDeviceId(0), sub_core_grids=sdpa_grid
    )
    full_gather = lambda: gather()
    quotas = [int(q) for q in os.environ.get("PREFETCH_QUOTAS", "3,6,9,12,15,18,21,24,30,40").split(",")]
    prefetch = lambda q: gather(prefix_only=True, window_max_slabs=q, progress_tensor=progress)
    resume = lambda: gather(window_start_tensor=progress)

    # Each measured phase is a captured trace, so both programs of an overlap launch together on every
    # chip as they would in the traced model, instead of trailing by the eager dispatch gap.
    traces = {}

    def capture(name, *fns):
        for fn in fns:  # compile before capture
            fn()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for fn in fns:
            fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        traces[name] = tid

    def replay_us(name, n=20):
        # Back-to-back replays, timed on the host: with traces the launch gap is a few us, and every
        # configuration is measured the same way.
        ttnn.execute_trace(mesh_device, traces[name], cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh_device, traces[name], cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) / n * 1e6

    mesh_device.load_sub_device_manager(manager)
    try:
        capture("solo_sdpa", sdpa)
        capture("solo_full_gather", full_gather)
        for q in quotas:
            capture(f"overlap_q{q}", sdpa, lambda q=q: prefetch(q))
            capture(f"resume_q{q}", resume)
            capture(f"prefetch_q{q}", lambda q=q: prefetch(q))
        base_sdpa, base_gather = replay_us("solo_sdpa"), replay_us("solo_full_gather")
        base = base_sdpa + base_gather
        lines = [
            f"GLM {kv_fmt} chunk {chunk_global}, prefix {prefix_slabs} slabs: sdpa {base_sdpa:.0f} + gather {base_gather:.0f} = {base:.0f}us"
        ]
        lines.append(f"{'quota':>6}{'prefetch':>10}{'overlap':>9}{'resume':>8}{'total':>8}{'saved':>8}")
        for q in quotas:
            pre, ovl, res = replay_us(f"prefetch_q{q}"), replay_us(f"overlap_q{q}"), replay_us(f"resume_q{q}")
            lines.append(
                f"{q:>6}{pre:>10.0f}{ovl:>9.0f}{res:>8.0f}{ovl + res:>8.0f}{100 * (base - ovl - res) / base:>7.1f}%"
            )
        logger.info("\n" + "\n".join(lines))
        # Leave the buffer from the largest quota's pair for the exactness check.
        ttnn.execute_trace(mesh_device, traces[f"overlap_q{quotas[-1]}"], cq_id=0, blocking=True)
        ttnn.execute_trace(mesh_device, traces[f"resume_q{quotas[-1]}"], cq_id=0, blocking=True)
        signpost("done")
    finally:
        ttnn.synchronize_device(mesh_device)
        for tid in traces.values():
            ttnn.release_trace(mesh_device, tid)
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)

    # Device typecasts span both sub-devices, so check exactness once the overlap manager is gone.
    # The overlapped prefetch + resume must have gathered layer 1's prefix exactly.
    local_rows = cache_tokens // n_dev
    rows = (prefix_slabs + 1) * (slab_global // n_dev)
    # FP8 shards cannot go to torch directly; a device typecast maps each byte the same way on both sides.
    as_bf16 = lambda t: t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)
    sources = [ttnn.to_torch(t)[1, 0, :rows] for t in ttnn.get_device_tensors(as_bf16(cache))]
    shards = ttnn.get_device_tensors(as_bf16(next_buffer))
    for chip in (0, n_dev - 1):
        actual = ttnn.to_torch(shards[chip])
        for rank in range(n_dev):
            lo = rank * local_rows
            # Packed FP8 rows carry raw scale bytes that may decode as NaN; compare them as equal.
            torch.testing.assert_close(
                actual[0, 0, lo : lo + rows], sources[rank], rtol=0, atol=0, equal_nan=True, msg=f"{chip} {rank}"
            )


# ---------------------------------------------------------------------------------------------------------
# PROTOTYPE index-driven KV fetch: owners push only the rows each chip's top-k needs, plus the current chunk's
# slab, into the chip's compact buffer; sparse_sdpa reads it through indices rewritten to compact slots.
# Needed-row sets are synthetic (attention sink + random clusters) so distinct rows per chip and their locality
# can be swept until real top-k statistics exist.
# ---------------------------------------------------------------------------------------------------------

_TORUS_XY_GLM = {**_TORUS_XY}
_GLM_ROUTER = ttnn.FabricRouterConfig()
_GLM_ROUTER.max_packet_payload_size_bytes = int(
    os.environ.get("SPARSE_PAYLOAD", "6144")
)  # GLM53Config.FABRIC_PAYLOAD_SIZE
_TORUS_XY_GLM["fabric_router_config"] = _GLM_ROUTER
_DRAM_BANKS = 8


def _needed_prefix_rows(rng, prefix_tokens, distinct, cluster_len, sink=64):
    """Synthetic needed prefix rows for one chip: attention sink plus random clusters of cluster_len rows."""
    rows = np.zeros(prefix_tokens, dtype=bool)
    rows[: min(sink, prefix_tokens)] = True
    target = min(distinct, prefix_tokens)
    while rows.sum() < target:
        starts = rng.integers(0, max(1, prefix_tokens - cluster_len), size=max(1, (target - rows.sum()) // cluster_len))
        for s0 in starts:
            rows[s0 : s0 + cluster_len] = True
    return np.flatnonzero(rows)


def _owner_local(n, stripe_chunk, n_dev):
    blk = n // stripe_chunk
    return blk % n_dev, (blk // n_dev) * stripe_chunk + n % stripe_chunk


def _compact_plan(needed_by_chip, n_dev, stripe_chunk, workers, pages_per_packet, buffer_rows, base_offsets=None):
    """Stream pages per owner/worker, each chip's natural-row -> compact-slot map, and each chip's region end.

    Owner o's rows for dest r sit in r's buffer at base(o, r) in stream order; packet j of a stream starts at
    base + (j % banks) + (j // banks) * P * banks with its P rows banks pages apart (one contiguous bank lane).
    """
    unit = pages_per_packet * _DRAM_BANKS
    streams = [[[] for _ in range(workers)] for _ in range(n_dev)]
    load = np.zeros((n_dev, workers), dtype=np.int64)
    slot_maps, ends = [], []
    for dest, rows in enumerate(needed_by_chip):
        owner, local = _owner_local(rows, stripe_chunk, n_dev)
        slot_map = np.full(buffer_rows, -1, dtype=np.int32)
        base = 0 if base_offsets is None else base_offsets[dest]
        for o in range(n_dev):
            sel = owner == o
            o_rows, o_local = rows[sel], local[sel]
            order = np.argsort(o_local)
            o_rows, o_local = o_rows[order], o_local[order]
            k = np.arange(len(o_local))
            j, i = k // pages_per_packet, k % pages_per_packet
            slot_map[o_rows] = base + (j % _DRAM_BANKS) + (j // _DRAM_BANKS) * unit + i * _DRAM_BANKS
            if len(o_local):
                w = int(np.argmin(load[o]))
                streams[o][w].append((dest, base, o_local))
                load[o, w] += len(o_local)
            base += -(-len(o_local) // unit) * unit
        assert base <= buffer_rows, f"compact buffer overflow: {base} > {buffer_rows}"
        slot_maps.append(slot_map)
        ends.append(base)
    words = 16 + max(sum(3 + len(l) for _, _, l in streams[o][w]) for o in range(n_dev) for w in range(workers))
    words += -words % 16
    host = np.zeros((n_dev, 1, workers, words), dtype=np.int64)
    for o in range(n_dev):
        for w in range(workers):
            host[o, 0, w, 0] = len(streams[o][w])
            pos = 16
            for dest, base, local in streams[o][w]:
                host[o, 0, w, pos : pos + 3] = (dest, base, len(local))
                host[o, 0, w, pos + 3 : pos + 3 + len(local)] = local
                pos += 3 + len(local)
    return torch.from_numpy(host.astype(np.int32)), slot_maps, int(load.max()), ends


@run_for_blackhole()
@_PERF
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY_GLM], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@_CHUNKS
@pytest.mark.parametrize("prefix_slabs", [40], ids=["prefix40slabs"])
@_KV_FMT
def test_sparse_push_vs_full_gather(mesh_device, chunk_global, prefix_slabs, kv_fmt):
    grid = mesh_device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (12, 10):
        pytest.skip(f"needs the 12x10 Galaxy grid, got {grid}")
    gather_grid = _cols(8, 11)
    n_dev = mesh_device.get_num_devices()
    workers = int(os.environ.get("SPARSE_WORKERS", "2"))
    slab_global = chunk_global
    stripe_chunk = slab_global // n_dev
    prefix_tokens = prefix_slabs * slab_global
    cache_tokens = (prefix_slabs + 1) * slab_global
    page_bytes = {"fp8": 704, "bf16": 1152}[kv_fmt]
    pages_per_packet = int(os.environ.get("SPARSE_PAYLOAD", "6144")) // page_bytes
    current_rows = np.arange(prefix_tokens, cache_tokens)

    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cache = _kv_cache(mesh_device, 2, cache_tokens, kv_fmt, ttnn.ShardTensorToMesh(mesh_device, dim=2), seed=2)
    dense_out = _zero_kv_buffer(mesh_device, cache_tokens, kv_fmt)
    compact_out = _zero_kv_buffer(mesh_device, cache_tokens, kv_fmt)

    def scalar(v):
        return ttnn.from_torch(
            torch.tensor([[[[v]]]], dtype=torch.int32),
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            **common,
        )

    slot, start = scalar(0), scalar(prefix_tokens)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([_cols(0, 7)]), ttnn.SubDevice([gather_grid])], 0)
    sems = [ttnn.create_global_semaphore(mesh_device, gather_grid, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
    ttnn.synchronize_device(mesh_device)

    gather_kw = dict(
        dim=2,
        cluster_axis=None,
        subdevice_id=ttnn.SubDeviceId(1),
        sub_core_grids=gather_grid,
        num_links=workers,
        input_batch_index_tensor=slot,
        batch_slot_num_layers=2,
        batch_slot_layer_idx=1,
        ready_semaphore=sems[0],
        data_valid_semaphore=sems[1],
    )
    full_gather = lambda: ttnn.experimental.high_bw_all_gather(
        cache, output_tensor=dense_out, gathered_prefix_tensor=start, gathered_slab_global=slab_global, **gather_kw
    )

    def timed(fn, n=20):
        fn()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        us = (time.perf_counter() - t0) / n * 1e6
        ttnn.release_trace(mesh_device, tid)
        return us

    distincts = [int(x) for x in os.environ.get("SPARSE_DISTINCT", "2048,8192,32768,65536").split(",")]
    clusters = [int(x) for x in os.environ.get("SPARSE_CLUSTER", "1,16,128").split(",")]
    lines, plan = [], None
    mesh_device.load_sub_device_manager(manager)
    try:
        t_full = timed(full_gather)
        lines.append(f"GLM {kv_fmt} chunk {chunk_global}, prefix {prefix_tokens} tokens: full gather {t_full:.0f}us")
        lines.append(f"{'distinct/chip':>14}{'cluster':>8}{'rows/worker':>12}{'push':>8}{'vs full':>9}")
        for distinct in distincts:
            for cluster in clusters:
                rng = np.random.default_rng(distinct * 1000 + cluster)
                prefix_needed = [_needed_prefix_rows(rng, prefix_tokens, distinct, cluster) for _ in range(n_dev)]
                if os.environ.get("SPARSE_NEIGHBORS_ONLY"):
                    # Diagnostic: every row comes from a direct torus neighbour, so every packet is one hop.
                    rows_, cols_ = tuple(mesh_device.shape)
                    for c in range(n_dev):
                        r0, c0 = divmod(c, cols_)
                        nbrs = {
                            ((r0 + dr) % rows_) * cols_ + (c0 + dc) % cols_
                            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
                        }
                        cand = np.arange(prefix_tokens)
                        cand = cand[np.isin(_owner_local(cand, stripe_chunk, n_dev)[0], list(nbrs))]
                        prefix_needed[c] = rng.choice(cand, size=min(distinct, len(cand)), replace=False)
                    needed = prefix_needed
                else:
                    needed = [np.concatenate([p, current_rows]) for p in prefix_needed]
                host_streams, slot_maps, worker_rows, _ = _compact_plan(
                    needed, n_dev, stripe_chunk, workers, pages_per_packet, cache_tokens
                )
                streams = ttnn.from_torch(
                    host_streams, dtype=ttnn.uint32, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0), **common
                )
                push = lambda: ttnn.experimental.high_bw_all_gather(
                    cache, output_tensor=compact_out, push_runs_tensor=streams, **gather_kw
                )
                t_push = timed(push)
                lines.append(
                    f"{distinct:>14}{cluster:>8}{worker_rows:>12}{t_push:>8.0f}{100 * (t_full - t_push) / t_full:>8.1f}%"
                )
                plan = (prefix_needed, slot_maps)
        logger.info("\n" + "\n".join(lines))
    finally:
        ttnn.synchronize_device(mesh_device)
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)

    # End to end on the last configuration: sparse_sdpa over the compact buffer with rewritten indices must
    # equal sparse_sdpa over the dense gather with natural indices.
    prefix_needed, slot_maps = plan
    s_local = chunk_global // n_dev
    rng = np.random.default_rng(7)
    natural = np.zeros((n_dev, s_local, TOPK), dtype=np.int64)
    for c in range(n_dev):
        for q in range(s_local):
            t = prefix_tokens + c * s_local + q  # this chip's query rows are its own current-chunk stripe
            recent = np.arange(max(prefix_tokens, t - 1023), t + 1)
            older = rng.choice(prefix_needed[c], size=TOPK - len(recent), replace=False)
            natural[c, q] = np.concatenate([recent, older])
    compact = np.stack([slot_maps[c][natural[c]] for c in range(n_dev)])
    assert (compact >= 0).all()

    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    to_idx = lambda a: ttnn.from_torch(
        torch.from_numpy(a.reshape(1, 1, n_dev * s_local, TOPK).astype(np.int32)),
        dtype=ttnn.uint32,
        mesh_mapper=shard_seq,
        **common,
    )
    q = ttnn.from_torch(
        torch.randn(1, H, n_dev * s_local, K_DIM).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=shard_seq,
        **common,
    )
    out_dense = ttnn.to_torch(
        _sparse_sdpa(q, dense_out, to_idx(natural), s_local * TP, kv_fmt),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
    )
    out_compact = ttnn.to_torch(
        ttnn.transformer.sparse_sdpa(
            q,
            compact_out,
            to_idx(compact),
            V_DIM,
            kv_format=_KV_FORMATS[kv_fmt][0],
            scale=K_DIM**-0.5,
            k_chunk_size=128,
        ),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
    )
    assert torch.equal(out_dense, out_compact), "compact-buffer sparse_sdpa differs from the dense-gather path"


# ---------------------------------------------------------------------------------------------------------
# Sparse push with real GLM-5.3 top-k at real context length, overlapped like a shared layer: while layer L's
# sparse_sdpa reads its compact buffer (80 cores), layer L+1's needed prefix rows (same indices -- shared
# layers reuse them) are pushed into L+1's compact buffer (40 cores); then the current slab is pushed as the
# remainder. Compared with today's sparse_sdpa + dense full-prefix gather.
# ---------------------------------------------------------------------------------------------------------

_TRACE_1M = Path(
    os.environ.get(
        "GLM53_1M_TRACE", "/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/glm53-1020k-last5120"
    )
)
_TRACE_1M_START = 1039360  # prompt position of the trace's first row (a whole number of 5120 slabs)


@run_for_blackhole()
@_PERF
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY_GLM], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_sparse_overlap_real_topk(mesh_device):
    from safetensors.numpy import load_file

    if not _TRACE_1M.exists():
        pytest.skip(f"1M GLM-5.3 trace not found at {_TRACE_1M}")
    grid = mesh_device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (12, 10):
        pytest.skip(f"needs the 12x10 Galaxy grid, got {grid}")
    kv_fmt, chunk_global = "fp8", 5120
    sdpa_grid, gather_grid = _cols(0, 7), _cols(8, 11)
    n_dev = mesh_device.get_num_devices()
    workers = 2
    slab_global = chunk_global
    stripe_chunk = slab_global // n_dev
    s_local = chunk_global // n_dev
    prefix_tokens = _TRACE_1M_START
    cache_tokens = prefix_tokens + slab_global
    page_bytes = 704
    pages_per_packet = int(os.environ.get("SPARSE_PAYLOAD", "6144")) // page_bytes
    current_rows = np.arange(prefix_tokens, cache_tokens)
    layers = [int(x) for x in os.environ.get("TOPK_LAYERS", "14,58,62").split(",")]

    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    cache = _kv_cache(mesh_device, 2, cache_tokens, kv_fmt, ttnn.ShardTensorToMesh(mesh_device, dim=2), seed=2)
    dense_out = _zero_kv_buffer(mesh_device, cache_tokens, kv_fmt)
    compact = [_zero_kv_buffer(mesh_device, cache_tokens, kv_fmt) for _ in range(2)]  # layer L, layer L+1
    q = ttnn.from_torch(
        torch.randn(1, H, n_dev * s_local, K_DIM).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=shard_seq,
        **common,
    )

    def scalar(v):
        return ttnn.from_torch(
            torch.tensor([[[[v]]]], dtype=torch.int32),
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            **common,
        )

    user, start = scalar(0), scalar(prefix_tokens)
    to_idx = lambda a: ttnn.from_torch(
        torch.from_numpy(a.reshape(1, 1, n_dev * s_local, TOPK).astype(np.int32)),
        dtype=ttnn.uint32,
        mesh_mapper=shard_seq,
        **common,
    )
    to_streams = lambda h: ttnn.from_torch(
        h, dtype=ttnn.uint32, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0), **common
    )
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([sdpa_grid]), ttnn.SubDevice([gather_grid])], 0)
    sems = [ttnn.create_global_semaphore(mesh_device, gather_grid, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
    ttnn.synchronize_device(mesh_device)

    def gather_kw(layer_slot):
        return dict(
            dim=2,
            cluster_axis=None,
            subdevice_id=ttnn.SubDeviceId(1),
            sub_core_grids=gather_grid,
            num_links=workers,
            input_batch_index_tensor=user,
            batch_slot_num_layers=2,
            batch_slot_layer_idx=layer_slot,
            ready_semaphore=sems[0],
            data_valid_semaphore=sems[1],
        )

    def timed(fn, n=10):
        fn()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        us = (time.perf_counter() - t0) / n * 1e6
        ttnn.release_trace(mesh_device, tid)
        return us

    lines = [
        f"GLM-5.3 real top-k, fp8, chunk {chunk_global}, prefix {prefix_tokens} tokens (8x4 TORUS_XY)",
        f"{'layer':>5}{'max rows':>9} | {'today: sdpa + gather = total':>30} | "
        f"{'new: sdpa||prefetch + remainder = total':>40}{'saved':>8} | {'prefetch alone':>14}",
    ]
    last = None
    for layer in layers:
        f = _TRACE_1M / "dsa" / f"dsa_topk_indices_layer_{layer}" / "rows_00000000_00005120.safetensors"
        natural = next(iter(load_file(str(f)).values())).astype(np.int64).reshape(n_dev, s_local, TOPK)
        prefix_needed = []
        for c in range(n_dev):
            u = np.unique(natural[c])
            prefix_needed.append(u[u < prefix_tokens])
        plan_kw = dict(n_dev=n_dev, stripe_chunk=stripe_chunk, workers=workers, pages_per_packet=pages_per_packet)
        pre_streams, pre_maps, _, ends = _compact_plan(prefix_needed, buffer_rows=cache_tokens, **plan_kw)
        rem_streams, rem_maps, _, _ = _compact_plan(
            [current_rows] * n_dev, buffer_rows=cache_tokens, base_offsets=ends, **plan_kw
        )
        maps = [np.where(p >= 0, p, r) for p, r in zip(pre_maps, rem_maps)]
        compact_idx = np.stack([maps[c][natural[c]] for c in range(n_dev)])
        assert (compact_idx >= 0).all()
        pre_t, rem_t = to_streams(pre_streams), to_streams(rem_streams)
        idx_natural, idx_compact = to_idx(natural), to_idx(compact_idx)

        push = lambda streams, slot_buf, layer_slot: ttnn.experimental.high_bw_all_gather(
            cache, output_tensor=slot_buf, push_runs_tensor=streams, **gather_kw(layer_slot)
        )
        sdpa_compact = lambda: ttnn.transformer.sparse_sdpa(
            q,
            compact[0],
            idx_compact,
            V_DIM,
            kv_format=_KV_FORMATS[kv_fmt][0],
            scale=K_DIM**-0.5,
            k_chunk_size=128,
            subdevice_id=ttnn.SubDeviceId(0),
            sub_core_grids=sdpa_grid,
        )
        full_gather = lambda: ttnn.experimental.high_bw_all_gather(
            cache,
            output_tensor=dense_out,
            gathered_prefix_tensor=start,
            gathered_slab_global=slab_global,
            **gather_kw(1),
        )
        mesh_device.load_sub_device_manager(manager)
        try:
            # Layer L's compact buffer is filled beforehand (by its own prefetch + remainder in the model).
            push(pre_t, compact[0], 0)
            push(rem_t, compact[0], 0)
            t_gather = timed(full_gather)
            t_prefetch = timed(lambda: push(pre_t, compact[1], 1))
            t_overlap = timed(lambda: (sdpa_compact(), push(pre_t, compact[1], 1)))
            t_rem = timed(lambda: push(rem_t, compact[1], 1))
        finally:
            ttnn.synchronize_device(mesh_device)
            mesh_device.clear_loaded_sub_device_manager()
        # Today's sparse_sdpa runs on the full grid over the dense gather, with natural block-cyclic indices.
        sdpa_today = lambda: _sparse_sdpa(q, dense_out, idx_natural, s_local * TP, kv_fmt)
        t_sdpa = timed(sdpa_today)
        today, new = t_sdpa + t_gather, t_overlap + t_rem
        lines.append(
            f"{layer:>5}{max(len(p) for p in prefix_needed):>9} | {t_sdpa:>8.0f} + {t_gather:>7.0f} = {today:>7.0f}  | "
            f"{t_overlap:>15.0f} + {t_rem:>9.0f} = {new:>7.0f}  {100 * (today - new) / today:>6.1f}% | {t_prefetch:>14.0f}"
        )
        last = (sdpa_today, idx_compact)
    logger.info("\n" + "\n".join(lines))
    mesh_device.remove_sub_device_manager(manager)

    # Exactness on the last layer: layer L+1's compact buffer through rewritten real indices equals the dense
    # gather of layer L+1 through natural indices.
    sdpa_today, idx_compact = last
    out_dense = ttnn.to_torch(sdpa_today(), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
    out_compact = ttnn.to_torch(
        ttnn.transformer.sparse_sdpa(
            q, compact[1], idx_compact, V_DIM, kv_format=_KV_FORMATS[kv_fmt][0], scale=K_DIM**-0.5, k_chunk_size=128
        ),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
    )
    assert torch.equal(out_dense, out_compact), "compact-buffer sparse_sdpa differs from the dense-gather path"


# ---------------------------------------------------------------------------------------------------------
# Crossover over prefix length (shared-layer schedule, fp8, 5k chunk): today (sparse_sdpa + dense gather) vs
# dense overlap (sparse_sdpa || dense prefix prefetch with the knee quota, + resume) vs sparse overlap
# (sparse_sdpa || needed-row push, + current-slab push). Needed rows per chip = min(prefix, SPARSE_DCAP),
# drawn uniformly from the prefix; the bitmap / exchange / index-rewrite bookkeeping is not included.
# ---------------------------------------------------------------------------------------------------------


@run_for_blackhole()
@_PERF
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY_GLM], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_gather_schemes_vs_prefix(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (12, 10):
        pytest.skip(f"needs the 12x10 Galaxy grid, got {grid}")
    kv_fmt, chunk_global = "fp8", 5120
    sdpa_grid, gather_grid = _cols(0, 7), _cols(8, 11)
    n_dev = mesh_device.get_num_devices()
    workers = 2
    slab_global = chunk_global
    stripe_chunk = slab_global // n_dev
    s_local = chunk_global // n_dev
    prefix_list = [int(x) for x in os.environ.get("PREFIX_SLABS", "1,2,4,8,16,32,64,128,203").split(",")]
    quota = int(os.environ.get("PREFETCH_QUOTA_SLABS", "21"))
    dcap = int(os.environ.get("SPARSE_DCAP", "30000"))
    cache_tokens = (max(prefix_list) + 1) * slab_global
    pages_per_packet = int(os.environ.get("SPARSE_PAYLOAD", "6144")) // 704

    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    cache = _kv_cache(mesh_device, 2, cache_tokens, kv_fmt, ttnn.ShardTensorToMesh(mesh_device, dim=2), seed=2)
    dense = [_zero_kv_buffer(mesh_device, cache_tokens, kv_fmt) for _ in range(2)]  # layer L, layer L+1
    compact = [_zero_kv_buffer(mesh_device, cache_tokens, kv_fmt) for _ in range(2)]
    q = ttnn.from_torch(
        torch.randn(1, H, n_dev * s_local, K_DIM).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=shard_seq,
        **common,
    )

    def scalar(v):
        return ttnn.from_torch(
            torch.tensor([[[[v]]]], dtype=torch.int32),
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            **common,
        )

    user, start, progress = scalar(0), scalar(0), scalar(0)
    to_idx = lambda a: ttnn.from_torch(
        torch.from_numpy(a.reshape(1, 1, n_dev * s_local, TOPK).astype(np.int32)),
        dtype=ttnn.uint32,
        mesh_mapper=shard_seq,
        **common,
    )
    to_streams = lambda h: ttnn.from_torch(
        h, dtype=ttnn.uint32, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0), **common
    )
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([sdpa_grid]), ttnn.SubDevice([gather_grid])], 0)
    sems = [ttnn.create_global_semaphore(mesh_device, gather_grid, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
    ttnn.synchronize_device(mesh_device)

    def kw(layer_slot):
        return dict(
            dim=2,
            cluster_axis=None,
            subdevice_id=ttnn.SubDeviceId(1),
            sub_core_grids=gather_grid,
            num_links=workers,
            input_batch_index_tensor=user,
            batch_slot_num_layers=2,
            batch_slot_layer_idx=layer_slot,
            ready_semaphore=sems[0],
            data_valid_semaphore=sems[1],
        )

    def timed(fn, n=10):
        fn()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        us = (time.perf_counter() - t0) / n * 1e6
        ttnn.release_trace(mesh_device, tid)
        return us

    lines = [
        f"fp8, chunk {chunk_global}, dense quota {quota} slabs, sparse rows/chip = min(prefix, {dcap}) (8x4 TORUS_XY)",
        f"{'prefix':>8} | {'today: sdpa + gather = total':>30} | {'dense: overlap + resume = total':>32}{'vs':>7} | "
        f"{'sparse: overlap + remainder = total':>36}{'vs':>7}",
    ]
    rng = np.random.default_rng(0)
    for p_slabs in prefix_list:
        prefix = p_slabs * slab_global
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[[[prefix]]]], dtype=torch.int32), dtype=ttnn.uint32), start
        )
        # Natural indices for today / dense: each query takes 2048 rows (with repeats allowed across queries)
        # from its chip's needed set plus the current slab; compact indices are the same rows rewritten.
        needed = [np.sort(rng.choice(prefix, size=min(prefix, dcap), replace=False)) for _ in range(n_dev)]
        cur = np.arange(prefix, prefix + slab_global)
        natural = np.zeros((n_dev, s_local, TOPK), dtype=np.int64)
        for c in range(n_dev):
            pool = np.concatenate([needed[c], cur])
            for qi in range(s_local):
                natural[c, qi] = rng.choice(pool, size=TOPK, replace=len(pool) < TOPK)
        plan_kw = dict(n_dev=n_dev, stripe_chunk=stripe_chunk, workers=workers, pages_per_packet=pages_per_packet)
        pre_s, pre_maps, _, ends = _compact_plan(needed, buffer_rows=cache_tokens, **plan_kw)
        rem_s, rem_maps, _, _ = _compact_plan([cur] * n_dev, buffer_rows=cache_tokens, base_offsets=ends, **plan_kw)
        maps = [np.where(p >= 0, p, r) for p, r in zip(pre_maps, rem_maps)]
        idx_nat = to_idx(natural)
        idx_cmp = to_idx(np.stack([maps[c][natural[c]] for c in range(n_dev)]))
        pre_t, rem_t = to_streams(pre_s), to_streams(rem_s)

        sdpa80_dense = lambda: _sparse_sdpa(
            q, dense[0], idx_nat, s_local * TP, kv_fmt, subdevice_id=ttnn.SubDeviceId(0), sub_core_grids=sdpa_grid
        )
        sdpa80_compact = lambda: ttnn.transformer.sparse_sdpa(
            q,
            compact[0],
            idx_cmp,
            V_DIM,
            kv_format=_KV_FORMATS[kv_fmt][0],
            scale=K_DIM**-0.5,
            k_chunk_size=128,
            subdevice_id=ttnn.SubDeviceId(0),
            sub_core_grids=sdpa_grid,
        )
        gather_args = dict(gathered_prefix_tensor=start, gathered_slab_global=slab_global)
        full_gather = lambda: ttnn.experimental.high_bw_all_gather(
            cache, output_tensor=dense[1], **gather_args, **kw(1)
        )
        prefetch = lambda: ttnn.experimental.high_bw_all_gather(
            cache,
            output_tensor=dense[1],
            prefix_only=True,
            window_max_slabs=quota,
            progress_tensor=progress,
            **gather_args,
            **kw(1),
        )
        resume = lambda: ttnn.experimental.high_bw_all_gather(
            cache, output_tensor=dense[1], window_start_tensor=progress, **gather_args, **kw(1)
        )
        push = lambda streams, buf: ttnn.experimental.high_bw_all_gather(
            cache, output_tensor=buf, push_runs_tensor=streams, **kw(1)
        )
        mesh_device.load_sub_device_manager(manager)
        try:
            push(pre_t, compact[0])
            push(rem_t, compact[0])
            t_gather = timed(full_gather)
            d_ovl, d_res = timed(lambda: (sdpa80_dense(), prefetch())), timed(resume)
            s_ovl, s_rem = timed(lambda: (sdpa80_compact(), push(pre_t, compact[1]))), timed(
                lambda: push(rem_t, compact[1])
            )
        finally:
            ttnn.synchronize_device(mesh_device)
            mesh_device.clear_loaded_sub_device_manager()
        t_sdpa = timed(lambda: _sparse_sdpa(q, dense[0], idx_nat, s_local * TP, kv_fmt))
        t_today, t_dense, t_sparse = t_sdpa + t_gather, d_ovl + d_res, s_ovl + s_rem
        pct = lambda t: f"{100 * (t - t_today) / t_today:>+6.1f}%"
        lines.append(
            f"{prefix:>8} | {t_sdpa:>8.0f} + {t_gather:>7.0f} = {t_today:>7.0f} | "
            f"{d_ovl:>9.0f} + {d_res:>7.0f} = {t_dense:>7.0f}{pct(t_dense)} | "
            f"{s_ovl:>9.0f} + {s_rem:>9.0f} = {t_sparse:>7.0f}{pct(t_sparse)}"
        )
    logger.info("\n" + "\n".join(lines))
    mesh_device.remove_sub_device_manager(manager)


# ---------------------------------------------------------------------------------------------------------
# Real 1M top-k with the on-device plan: layer L's sparse_sdpa (sub-device 0) overlapped with layer L+1's push of
# its prefix rows (sub-device 1), then L+1's remainder push (rows of the current chunk) before its own sparse_sdpa.
# The plan (sparse_push_plan + table all-gather) runs on the full grid and is timed separately.
# ---------------------------------------------------------------------------------------------------------


@run_for_blackhole()
@_PERF
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY_GLM], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_sparse_overlap_device_plan(mesh_device):
    from safetensors.numpy import load_file

    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import _shard_over_mesh

    if not _TRACE_1M.exists():
        pytest.skip(f"1M GLM-5.3 trace not found at {_TRACE_1M}")
    grid = mesh_device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (12, 10):
        pytest.skip(f"needs the 12x10 Galaxy grid, got {grid}")
    kv_fmt, chunk_global = "fp8", 5120
    sdpa_grid, gather_grid = _cols(0, 7), _cols(8, 11)
    n_dev = mesh_device.get_num_devices()
    links = 2
    stripe_chunk = chunk_global // n_dev
    s_local = chunk_global // n_dev
    prefix_tokens = _TRACE_1M_START
    cache_tokens = prefix_tokens + chunk_global
    page_bytes = 704
    pages_per_packet = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // page_bytes)
    buffer_rows = cache_tokens + n_dev * pages_per_packet * _DRAM_BANKS
    layers = [int(x) for x in os.environ.get("TOPK_LAYERS", "14,58,62").split(",")]

    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    cache = _kv_cache(mesh_device, 2, cache_tokens, kv_fmt, shard_seq, seed=2)
    dense_out = _zero_kv_buffer(mesh_device, cache_tokens, kv_fmt)
    compact = [_zero_kv_buffer(mesh_device, buffer_rows, kv_fmt) for _ in range(2)]  # layer L, layer L+1
    q = ttnn.from_torch(
        torch.randn(1, H, n_dev * s_local, K_DIM).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=shard_seq,
        **common,
    )

    def scalar(v):
        return ttnn.from_torch(
            torch.tensor([[[[v]]]], dtype=torch.int32), dtype=ttnn.uint32, mesh_mapper=replicate, **common
        )

    user, start = scalar(0), scalar(prefix_tokens)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([sdpa_grid]), ttnn.SubDevice([gather_grid])], 0)
    sems = [ttnn.create_global_semaphore(mesh_device, gather_grid, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
    full_grid = _cols(0, grid.x - 1)
    full_sems = [ttnn.create_global_semaphore(mesh_device, full_grid, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
    progress = ttnn.from_torch(
        torch.zeros(1, 1, 4 * links * 2, 16, dtype=torch.int32), dtype=ttnn.uint32, mesh_mapper=replicate, **common
    )
    ttnn.synchronize_device(mesh_device)

    def gather_kw(layer_slot):
        return dict(
            dim=2,
            cluster_axis=None,
            subdevice_id=ttnn.SubDeviceId(1),
            sub_core_grids=gather_grid,
            num_links=links,
            input_batch_index_tensor=user,
            batch_slot_num_layers=2,
            batch_slot_layer_idx=layer_slot,
            ready_semaphore=sems[0],
            data_valid_semaphore=sems[1],
        )

    def timed(fn, n=10):
        fn()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        us = (time.perf_counter() - t0) / n * 1e6
        ttnn.release_trace(mesh_device, tid)
        return us

    tables = None
    lines = [
        f"GLM-5.3 real top-k, fp8, chunk {chunk_global}, prefix {prefix_tokens} tokens, on-device plan (8x4 TORUS_XY)",
        f"{'layer':>5} | {'today: sdpa + gather = total':>30} | {'serial push: plan/3 + push + sdpa':>34} | "
        f"{'overlap: plan/3 + sdpa||prefix + remainder':>44}{'saved':>8} | "
        f"{'fused: plan/3 + sdpa+prefix + resume':>38}{'saved':>8}",
    ]
    last = None
    for layer in layers:
        f = _TRACE_1M / "dsa" / f"dsa_topk_indices_layer_{layer}" / "rows_00000000_00005120.safetensors"
        natural = next(iter(load_file(str(f)).values())).astype(np.int64).reshape(n_dev, s_local, TOPK)
        idx_natural = ttnn.from_torch(
            torch.from_numpy(natural.reshape(1, 1, n_dev * s_local, TOPK).astype(np.int32)),
            dtype=ttnn.uint32,
            mesh_mapper=shard_seq,
            **common,
        )

        def plan():
            table, compact_idx = ttnn.experimental.sparse_push_plan(
                idx_natural,
                num_devices=n_dev,
                stripe_chunk=stripe_chunk,
                max_rows=cache_tokens,
                pages_per_packet=pages_per_packet,
                banks=_DRAM_BANKS,
                chunk_start=start,
            )
            _shard_over_mesh(table, mesh_device)
            ttnn.experimental.high_bw_all_gather(table, dim=2, output_tensor=tables, cluster_axis=None, num_links=links)
            return compact_idx

        if tables is None:
            pages = ttnn.experimental.sparse_push_plan(
                idx_natural,
                num_devices=n_dev,
                stripe_chunk=stripe_chunk,
                max_rows=cache_tokens,
                pages_per_packet=pages_per_packet,
                banks=_DRAM_BANKS,
            )[0].shape[2]
            tables = ttnn.from_torch(
                torch.zeros(1, 1, pages * n_dev, 1024, dtype=torch.int32),
                dtype=ttnn.uint32,
                mesh_mapper=replicate,
                **common,
            )
        compact_idx = plan()
        t_plan = timed(plan)

        def push(buf, layer_slot, part):
            return ttnn.experimental.high_bw_all_gather(
                cache,
                output_tensor=buf,
                push_runs_tensor=tables,
                push_stripe_chunk=stripe_chunk,
                push_max_rows=cache_tokens,
                push_part=part,
                **gather_kw(layer_slot),
            )

        sdpa_compact = lambda buf: ttnn.transformer.sparse_sdpa(
            q,
            buf,
            compact_idx,
            V_DIM,
            kv_format=_KV_FORMATS[kv_fmt][0],
            scale=K_DIM**-0.5,
            k_chunk_size=128,
            subdevice_id=ttnn.SubDeviceId(0),
            sub_core_grids=sdpa_grid,
        )
        full_gather = lambda: ttnn.experimental.high_bw_all_gather(
            cache,
            output_tensor=dense_out,
            gathered_prefix_tensor=start,
            gathered_slab_global=chunk_global,
            **gather_kw(1),
        )
        mesh_device.load_sub_device_manager(manager)
        try:
            push(compact[0], 0, 0)  # layer L's buffer, filled beforehand (by its own prefetch + remainder)
            t_gather = timed(full_gather)
            t_push = timed(lambda: push(compact[1], 1, 0))
            t_overlap = timed(lambda: (sdpa_compact(compact[0]), push(compact[1], 1, 1)))
            t_rem = timed(lambda: push(compact[1], 1, 2))
            t_sdpa_sub = timed(lambda: sdpa_compact(compact[0]))
            push(compact[1], 1, 1)
            push(compact[1], 1, 2)
        finally:
            ttnn.synchronize_device(mesh_device)
            mesh_device.clear_loaded_sub_device_manager()
        # Fused: sparse_sdpa on columns 0-7 with layer L+1's prefix push in the same program on 8-11 (no sub-devices),
        # stopping when attention is done; then the resume push of whatever it did not send, plus the remainder.
        fused = lambda: ttnn.transformer.sparse_sdpa(
            q,
            compact[0],
            compact_idx,
            V_DIM,
            kv_format=_KV_FORMATS[kv_fmt][0],
            scale=K_DIM**-0.5,
            k_chunk_size=128,
            sub_core_grids=sdpa_grid,
            push_input=cache,
            push_table=tables,
            push_output=compact[1],
            push_progress=progress,
            push_stripe_chunk=stripe_chunk,
            push_max_rows=cache_tokens,
            push_batch_index=1,
            push_num_links=links,
            push_ready_semaphore=full_sems[0],
            push_data_valid_semaphore=full_sems[1],
            push_core_grid=gather_grid,
        )
        resume = lambda: ttnn.experimental.high_bw_all_gather(
            cache,
            dim=2,
            output_tensor=compact[1],
            cluster_axis=None,
            num_links=links,
            input_batch_index=1,
            push_runs_tensor=tables,
            push_stripe_chunk=stripe_chunk,
            push_max_rows=cache_tokens,
            push_part=3,
            push_progress_tensor=progress,
            ready_semaphore=full_sems[0],
            data_valid_semaphore=full_sems[1],
        )
        t_fused = timed(fused)
        t_fused_pair = timed(lambda: (fused(), resume()))
        t_resume = t_fused_pair - t_fused
        sent = ttnn.to_torch(progress, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:, 0, :, 0]
        logger.info(f"layer {layer}: fused prefix packets sent per sender (chip 0): {sent[0].tolist()}")
        ttnn.deallocate(compact[1])
        compact[1] = _zero_kv_buffer(mesh_device, buffer_rows, kv_fmt)
        fused()
        prog = ttnn.to_torch(progress, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:, 0, :, :3]
        resume()
        if os.environ.get("FUSED_DEBUG"):
            ref = _zero_kv_buffer(mesh_device, buffer_rows, kv_fmt)
            ttnn.experimental.high_bw_all_gather(
                cache,
                dim=2,
                output_tensor=ref,
                cluster_axis=None,
                num_links=links,
                input_batch_index=1,
                push_runs_tensor=tables,
                push_stripe_chunk=stripe_chunk,
                push_max_rows=cache_tokens,
                push_part=0,
                ready_semaphore=full_sems[0],
                data_valid_semaphore=full_sems[1],
            )
            raw = (
                lambda t: ttnn.to_torch(
                    ttnn.typecast(t, ttnn.bfloat16), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
                )[0, 0]
                .contiguous()
                .view(torch.int16)
            )
            got, want = raw(compact[1]), raw(ref)
            bad = (got != want).any(-1).nonzero().flatten()
            logger.info(
                f"layer {layer} FUSED_DEBUG chip0: {bad.numel()} mismatched rows, first {bad[:16].tolist()}; "
                f"progress [sent, finished, turn] {prog[0].tolist()}"
            )
            ttnn.deallocate(ref)
        sdpa_today = lambda: _sparse_sdpa(q, dense_out, idx_natural, s_local * TP, kv_fmt)
        t_sdpa = timed(sdpa_today)
        today = t_sdpa + t_gather
        fused_total = t_plan / 3 + t_fused + t_resume
        serial = t_plan / 3 + t_push + t_sdpa_sub
        new = t_plan / 3 + t_overlap + t_rem
        lines.append(
            f"{layer:>5} | {t_sdpa:>8.0f} + {t_gather:>7.0f} = {today:>7.0f}  | "
            f"{t_plan / 3:>6.0f} + {t_push:>6.0f} + {t_sdpa_sub:>6.0f} = {serial:>6.0f} | "
            f"{t_plan / 3:>6.0f} + {t_overlap:>12.0f} + {t_rem:>9.0f} = {new:>7.0f} {100 * (today - new) / today:>6.1f}% | "
            f"{t_plan / 3:>6.0f} + {t_fused:>12.0f} + {t_resume:>8.0f} = {fused_total:>7.0f} "
            f"{100 * (today - fused_total) / today:>6.1f}%"
        )
        last = (sdpa_today, compact_idx)
    logger.info("\n" + "\n".join(lines))
    mesh_device.remove_sub_device_manager(manager)

    # Exactness on the last layer: layer L+1's buffer, filled by prefix + remainder pushes, through the device-rewritten
    # indices equals the dense gather of layer L+1 through natural indices.
    sdpa_today, compact_idx = last
    out_dense = ttnn.to_torch(sdpa_today(), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
    out_compact = ttnn.to_torch(
        ttnn.transformer.sparse_sdpa(
            q, compact[1], compact_idx, V_DIM, kv_format=_KV_FORMATS[kv_fmt][0], scale=K_DIM**-0.5, k_chunk_size=128
        ),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
    )
    assert torch.equal(out_dense, out_compact), "fused prefix push + resume differs from the dense-gather path"


# ---------------------------------------------------------------------------------------------------------
# Link efficiency of the table push vs hop count: every requester needs ROWS rows from each owner at a fixed
# torus offset (1, 2 or 3 rows away, both directions), so the per-link byte load is known and only the hop count
# changes. Reports push time and effective GB/s on the busiest link.
# ---------------------------------------------------------------------------------------------------------


def _busiest_direction_mb(n_dev, mesh_r, mesh_c, idx, stripe_chunk, page_bytes):
    """MB on the busiest (chip, direction) for the push of `idx` [requester, rows, topk], N/S then E/W shortest paths."""
    load = {}
    for d in range(n_dev):
        v = idx[d].reshape(-1)
        v = np.unique(v[v != 0xFFFFFFFF])
        per_owner = np.bincount((v // stripe_chunk) % n_dev, minlength=n_dev)
        rd, cd = divmod(d, mesh_c)
        for o in range(n_dev):
            if o == d or not per_owner[o]:
                continue
            r, c = divmod(o, mesh_c)
            for axis, n, (a, b) in (("NS", mesh_r, (r, rd)), ("EW", mesh_c, (c, cd))):
                step = (b - a) % n
                step = step if step <= n // 2 else step - n
                for _ in range(abs(step)):
                    key = (r, c, axis, step > 0)
                    load[key] = load.get(key, 0) + per_owner[o] * page_bytes
                    if axis == "NS":
                        r = (r + (1 if step > 0 else -1)) % n
                    else:
                        c = (c + (1 if step > 0 else -1)) % n
    return max(load.values()) / 1e6


@run_for_blackhole()
@_PERF
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY_GLM], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_sparse_push_hop_efficiency(mesh_device):
    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import _shard_over_mesh

    kv_fmt, chunk_global = "fp8", 5120
    n_dev, mesh_r, mesh_c = mesh_device.get_num_devices(), 8, 4
    links, stripe_chunk, s_local = 2, chunk_global // 32, chunk_global // 32
    cache_tokens = 1044480
    rows_per_owner = int(os.environ.get("HOP_ROWS", "2048"))
    page_bytes = 704
    pages_per_packet = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // page_bytes)
    buffer_rows = cache_tokens + n_dev * pages_per_packet * _DRAM_BANKS
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    cache = _kv_cache(mesh_device, 1, cache_tokens, kv_fmt, shard_seq, seed=3)
    out = _zero_kv_buffer(mesh_device, buffer_rows, kv_fmt)
    grid = mesh_device.compute_with_storage_grid_size()
    sems = [
        ttnn.create_global_semaphore(mesh_device, _cols(0, grid.x - 1), 0, ttnn.BufferType.L1_SMALL) for _ in range(2)
    ]
    rng = np.random.default_rng(0)
    blocks_per_owner = cache_tokens // stripe_chunk // n_dev

    def owner_rows(o, count):
        j = rng.choice(blocks_per_owner - 1, size=count // stripe_chunk + 1, replace=False)
        rows = ((j[:, None] * n_dev + o) * stripe_chunk + np.arange(stripe_chunk)[None, :]).reshape(-1)
        return rows[:count]

    def timed(fn, n=10):
        fn()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        us = (time.perf_counter() - t0) / n * 1e6
        ttnn.release_trace(mesh_device, tid)
        return us

    lines = [f"table push, {rows_per_owner} rows per (owner, requester), 704 B rows, {links} links (8x4 TORUS_XY)"]
    for hops in os.environ.get("HOP_CASES", "1,2,3,all").split(","):
        # Requester (r, c) needs rows from the owners `hops` rows north and south of it, or from every owner.
        idx = np.full((n_dev, s_local, TOPK), 0xFFFFFFFF, dtype=np.int64)
        for d in range(n_dev):
            r, c = divmod(d, mesh_c)
            if hops == "all":
                owners = [o for o in range(n_dev) if o != d]
            else:
                h = int(hops)
                owners = [((r + h) % mesh_r) * mesh_c + c, ((r - h) % mesh_r) * mesh_c + c]
            need = np.concatenate([owner_rows(o, rows_per_owner) for o in owners])
            flat = idx[d].reshape(-1)
            flat[: need.size] = need
        idx_tt = ttnn.from_torch(
            torch.from_numpy(idx.reshape(1, 1, n_dev * s_local, TOPK).astype(np.uint32).view(np.int32)),
            dtype=ttnn.uint32,
            mesh_mapper=shard_seq,
            **common,
        )
        table, _ = ttnn.experimental.sparse_push_plan(
            idx_tt,
            num_devices=n_dev,
            stripe_chunk=stripe_chunk,
            max_rows=cache_tokens,
            pages_per_packet=pages_per_packet,
            banks=_DRAM_BANKS,
        )
        _shard_over_mesh(table, mesh_device)
        tables = ttnn.from_torch(
            torch.zeros(1, 1, table.shape[2] * n_dev, 1024, dtype=torch.int32),
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
            **common,
        )
        ttnn.experimental.high_bw_all_gather(table, dim=2, output_tensor=tables, cluster_axis=None, num_links=links)
        push = lambda: ttnn.experimental.high_bw_all_gather(
            cache,
            dim=2,
            output_tensor=out,
            cluster_axis=None,
            num_links=links,
            push_runs_tensor=tables,
            push_stripe_chunk=stripe_chunk,
            push_max_rows=cache_tokens,
            ready_semaphore=sems[0],
            data_valid_semaphore=sems[1],
        )
        us = timed(push)
        # Busiest direction's bytes (both links of it), from a dimension-ordered shortest-path route model.
        mb = _busiest_direction_mb(n_dev, mesh_r, mesh_c, idx, stripe_chunk, page_bytes)
        lines.append(
            f"  {hops} hop(s): {us:7.0f} us, busiest direction {mb:6.2f} MB -> {mb * 1e6 / us / 1e3:5.1f} GB/s "
            f"(pushed {(idx != 0xFFFFFFFF).sum() * page_bytes / n_dev / 1e6:.1f} MB per chip)"
        )
        for t in (idx_tt, table, tables):
            ttnn.deallocate(t)
    logger.info("\n" + "\n".join(lines))
