# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa on a restricted core grid, and sparse_sdpa overlapped with a next-layer KVPE prefetch.

GLM-5.3 prefill shape on the 8x4 Galaxy (SP8 x TP4, 64 heads resharded head->seq, TOPK 2048, latent 512 +
rope 64). The overlap case captures each configuration as a trace and times back-to-back replays on the host:
neither the realtime profiler nor the Tracy device profiler reports traced sub-device-1 gathers correctly.
"""

import os
import time
from functools import partial
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


# ---------------------------------------------------------------------------------------------------------
# PROTOTYPE multicast-tree push (SPARSE_PUSH_TREE=1): each prefix row is sent once per link of a tree from its owner
# to every requester that wants it, one hop at a time (see kernels/tree_push_reader.cpp). Host-planned.
# ---------------------------------------------------------------------------------------------------------

_E, _W, _N, _S = 0, 1, 2, 3  # get_neighbors order; N = row - 1, E = col + 1


def _ring_cover(offsets, n):
    """Smallest (forward, backward) ring extents covering offsets 1..n-1 from 0; balanced on ties."""
    best = None
    for fwd in range(n):
        back = max([n - t for t in offsets if t > fwd], default=0)
        key = (fwd + back, max(fwd, back))
        if best is None or key < best[0]:
            best = (key, fwd, back)
    return best[1], best[2]


def _tree_edges(rel_dests, order, mesh_shape):
    """Edges (from, dir, to, depth) of a tree from (0, 0) to rel_dests [(row, col)]; order "yx" walks the owner's
    column first and fans out along rows, "xy" the transpose."""
    R, C = mesh_shape
    yx = order == "yx"
    # (major ring length, minor ring length, major/minor -> chip, fwd/back dirs)
    nmaj, nmin = (R, C) if yx else (C, R)
    to_rc = (lambda a, b: (a % R, b % C)) if yx else (lambda a, b: (b % R, a % C))
    dmaj = (_S, _N) if yx else (_E, _W)
    dmin = (_E, _W) if yx else (_S, _N)
    dests = [(r, c) if yx else (c, r) for r, c in rel_dests]
    majors = sorted({a for a, _ in dests} - {0})
    fwd, back = _ring_cover(majors, nmaj)
    edges = []
    for t in range(fwd):
        edges.append((to_rc(t, 0), dmaj[0], to_rc(t + 1, 0), t))
    for t in range(back):
        edges.append((to_rc(-t, 0), dmaj[1], to_rc(-t - 1, 0), t))
    for a in {a for a, _ in dests}:
        minors = sorted({b for aa, b in dests if aa == a} - {0})
        if not minors:
            continue
        depth0 = a if a <= fwd else nmaj - a
        f2, b2 = _ring_cover(minors, nmin)
        for t in range(f2):
            edges.append((to_rc(a, t), dmin[0], to_rc(a, t + 1), depth0 + t))
        for t in range(b2):
            edges.append((to_rc(a, -t), dmin[1], to_rc(a, -t - 1), depth0 + t))
    return edges


def _tree_plan(natural, n_dev, mesh_shape, stripe_chunk, pages_per_packet, buffer_rows, links=2, k_per=2):
    """Host plan for the tree push: stream pages [n_dev, 1, senders, words], each requester's natural-row -> page
    map, and stats. Requester d needs the rows in natural[d] (negative = masked)."""
    R, C = mesh_shape
    B, P = _DRAM_BANKS, pages_per_packet
    unit = P * B
    senders = links * 4 * k_per
    needed = [np.unique(v[v >= 0]) for v in natural.reshape(n_dev, -1)]
    req_of = np.concatenate([np.full(len(v), d) for d, v in enumerate(needed)])
    uniq, inv = np.unique(np.concatenate(needed), return_inverse=True)
    masks = np.zeros(len(uniq), dtype=np.int64)
    np.bitwise_or.at(masks, inv, np.left_shift(1, req_of).astype(np.int64))
    owners, local = _owner_local(uniq, stripe_chunk, n_dev)
    remote = masks & ~np.left_shift(1, owners).astype(np.int64)

    # Group rows by (owner, remote requesters); pick the Y-first or X-first tree per group, keeping links balanced.
    order = np.lexsort((remote, owners))
    keys = np.stack([owners[order], remote[order]], 1)
    cuts = np.flatnonzero(np.any(keys[1:] != keys[:-1], 1)) + 1
    groups = [g for g in np.split(order, cuts) if remote[g[0]]]
    groups.sort(key=len, reverse=True)
    load = np.zeros((n_dev, 4))
    buckets = {}  # (depth, from, dir, keep) -> [uniq row ids]
    max_depth = 0
    for g in groups:
        o = int(owners[g[0]])
        ro, co = divmod(o, C)
        m = int(remote[g[0]])
        dests = [divmod(d, C) for d in range(n_dev) if m >> d & 1]
        rel = [((r - ro) % R, (c - co) % C) for r, c in dests]
        best = None
        for tree_order in ("yx", "xy"):
            edges = [
                (((a[0] + ro) % R) * C + (a[1] + co) % C, dr, ((b[0] + ro) % R) * C + (b[1] + co) % C, h)
                for a, dr, b, h in _tree_edges(rel, tree_order, mesh_shape)
            ]
            trial = load.copy()
            for a, dr, _, _ in edges:
                trial[a, dr] += len(g)
            if best is None or (trial.max(), len(edges)) < best[0]:
                best = ((trial.max(), len(edges)), trial, edges)
        _, load, edges = best
        for a, dr, b, h in edges:
            buckets.setdefault((h, a, dr, bool(m >> b & 1)), []).append(g)
            max_depth = max(max_depth, h)

    neighbour = lambda a, dr: (((a // C + (dr == _S) - (dr == _N)) % R) * C + (a % C + (dr == _E) - (dr == _W)) % C)
    nu = len(uniq)
    loc = np.full((n_dev, nu), np.iinfo(np.int64).min, dtype=np.int64)  # page (>= 0) or staging placeholder
    arr_w = np.full((n_dev, nu), -1, dtype=np.int64)
    arr_i = np.zeros((n_dev, nu), dtype=np.int64)
    compact_lanes = np.zeros(n_dev, dtype=np.int64)
    stage_lanes = np.zeros(n_dev, dtype=np.int64)
    sent = np.zeros((n_dev, senders), dtype=np.int64)
    rr = np.zeros((n_dev, 4), dtype=np.int64)
    expected = np.zeros((n_dev, senders), dtype=np.int64)
    items = [[[] for _ in range(senders)] for _ in range(n_dev)]  # (depth, waits, kind, target, srcs)
    lane_rel = lambda j: (j % B) + (j // B) * unit

    # Requesters' own rows: local copies into their compact buffers.
    for d in range(n_dev):
        ids = np.flatnonzero((owners == d) & (masks >> d & 1).astype(bool))
        for p, start in enumerate(range(0, len(ids), P)):
            chunk = ids[start : start + P]
            j = compact_lanes[d]
            compact_lanes[d] += 1
            loc[d, chunk] = lane_rel(j) + np.arange(len(chunk)) * B
            items[d][p % senders].append((0, {}, 3, int(lane_rel(j)), local[chunk]))

    for key in sorted(buckets):
        h, a, dr, keep = key
        b = neighbour(a, dr)
        ids = np.concatenate(buckets[key])
        for start in range(0, len(ids), P):
            chunk = ids[start : start + P]
            i = np.arange(len(chunk)) * B
            pick = rr[a, dr] % (links * k_per)
            rr[a, dr] += 1
            w = ((pick // k_per) * 4 + dr) * k_per + pick % k_per
            if keep:
                j = compact_lanes[b]
                compact_lanes[b] += 1
                target = int(lane_rel(j))
                loc[b, chunk] = target + i
            else:
                j = stage_lanes[b]
                stage_lanes[b] += 1
                target = -int(lane_rel(j)) - 1
                loc[b, chunk] = -(lane_rel(j) + i) - 1
            if h == 0:
                kind, srcs, waits = 1, local[chunk], {}
            else:
                kind, srcs = 2, loc[a, chunk].copy()
                assert (srcs != np.iinfo(np.int64).min).all(), "row forwarded before it arrived"
                waits = {}
                for s, k in zip(arr_w[a, chunk], arr_i[a, chunk]):
                    waits[int(s)] = max(waits.get(int(s), 0), int(k) + 1)
            arr_w[b, chunk] = w
            arr_i[b, chunk] = sent[a, w]
            sent[a, w] += 1
            expected[b, w] += 1
            items[a][w].append((h, waits, kind, target, srcs))

    compact_pages = -(-compact_lanes.max() // B) * unit
    stage_base = -(-compact_pages // unit) * unit
    stage_pages = -(-stage_lanes.max() // B) * unit
    assert stage_base + stage_pages <= buffer_rows, f"tree staging overflow: {stage_base + stage_pages} > {buffer_rows}"
    page_of = lambda v: np.where(v >= 0, v, stage_base - v - 1)

    pages = []
    for c in range(n_dev):
        for w in range(senders):
            words = [0] + [int(x) for x in expected[c]] + [0] * (31 - senders)
            waited = {}
            its = sorted(items[c][w], key=lambda it: (it[0], max(it[1].values(), default=0)))
            level = 0
            for h, waits, kind, target, srcs in its:
                while level < h:
                    words.append(6)
                    level += 1
                for s, k in sorted(waits.items()):
                    if k > waited.get(s, 0):
                        words += [4, s, k]
                        waited[s] = k
                words += [kind, int(page_of(np.int64(target))), len(srcs)]
                words += [int(x) for x in (page_of(srcs) if kind == 2 else srcs)]
            while level < max_depth:
                words.append(6)
                level += 1
            words.append(0)
            pages.append(words)
    width = max(len(w) for w in pages)
    width += -width % 16
    host = np.zeros((n_dev * senders, width), dtype=np.int64)
    for idx, w in enumerate(pages):
        host[idx, : len(w)] = w
    slot_maps = np.full((n_dev, natural.max() + 1), -1, dtype=np.int64)
    for d in range(n_dev):
        ids = np.searchsorted(uniq, needed[d])
        slot_maps[d, needed[d]] = page_of(loc[d, ids])
        assert (loc[d, ids] >= 0).all(), "requester row not in its compact buffer"
    stats = dict(
        depth=max_depth + 1,
        busiest_dir_mb=load.max() * 704 / 1e6,
        staged_rows=int(stage_lanes.max() * P),
        packets=int(sent.sum()),
        stream_words=width,
    )
    return torch.from_numpy(host.reshape(n_dev, 1, senders, width).astype(np.int32)), slot_maps, stats


@run_for_blackhole()
@_PERF
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY_GLM], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_sparse_push_tree(mesh_device):
    from safetensors.numpy import load_file

    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import _shard_over_mesh

    if not _TRACE_1M.exists():
        pytest.skip(f"1M GLM-5.3 trace not found at {_TRACE_1M}")
    assert os.environ.get("SPARSE_PUSH_TREE", "0") != "0", "run with SPARSE_PUSH_TREE=1"
    kv_fmt, chunk_global = "fp8", 5120
    n_dev = mesh_device.get_num_devices()
    mesh_shape = tuple(mesh_device.shape)
    links = 2
    stripe_chunk = s_local = chunk_global // n_dev
    prefix_tokens = _TRACE_1M_START
    cache_tokens = prefix_tokens + chunk_global
    page_bytes = 704
    pages_per_packet = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // page_bytes)
    buffer_rows = cache_tokens + n_dev * pages_per_packet * _DRAM_BANKS
    layers = [int(x) for x in os.environ.get("TOPK_LAYERS", "62").split(",")]
    k_per = max(1, int(os.environ.get("SPARSE_PUSH_MUX_K", "2")))  # senders per channel; 0 = direct (one)

    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    cache = _kv_cache(mesh_device, 2, cache_tokens, kv_fmt, shard_seq, seed=2)
    dense_out = _zero_kv_buffer(mesh_device, cache_tokens, kv_fmt)
    routed_out = _zero_kv_buffer(mesh_device, buffer_rows, kv_fmt)
    tree_out = _zero_kv_buffer(mesh_device, buffer_rows, kv_fmt)
    q = ttnn.from_torch(
        torch.randn(1, H, n_dev * s_local, K_DIM).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=shard_seq,
        **common,
    )
    scalar = lambda v: ttnn.from_torch(
        torch.tensor([[[[v]]]], dtype=torch.int32), dtype=ttnn.uint32, mesh_mapper=replicate, **common
    )
    user, start = scalar(0), scalar(prefix_tokens)
    grid = mesh_device.compute_with_storage_grid_size()
    sems = [
        ttnn.create_global_semaphore(mesh_device, _cols(0, grid.x - 1), 0, ttnn.BufferType.L1_SMALL) for _ in range(2)
    ]
    gather_kw = dict(
        dim=2,
        cluster_axis=None,
        num_links=links,
        input_batch_index_tensor=user,
        batch_slot_num_layers=2,
        batch_slot_layer_idx=1,
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

    ttnn.experimental.high_bw_all_gather(
        cache, output_tensor=dense_out, gathered_prefix_tensor=start, gathered_slab_global=chunk_global, **gather_kw
    )
    lines = [f"GLM-5.3 real top-k, fp8, 5k chunk at {prefix_tokens} tokens: routed table push vs one-hop tree push"]
    for layer in layers:
        f = _TRACE_1M / "dsa" / f"dsa_topk_indices_layer_{layer}" / "rows_00000000_00005120.safetensors"
        natural = next(iter(load_file(str(f)).values())).astype(np.int64).reshape(n_dev, s_local, TOPK)
        idx_natural = ttnn.from_torch(
            torch.from_numpy(natural.reshape(1, 1, n_dev * s_local, TOPK).astype(np.int32)),
            dtype=ttnn.uint32,
            mesh_mapper=shard_seq,
            **common,
        )
        # Today's push: device plan, table all-gather, multi-hop routed push.
        table, _ = ttnn.experimental.sparse_push_plan(
            idx_natural,
            num_devices=n_dev,
            stripe_chunk=stripe_chunk,
            max_rows=cache_tokens,
            pages_per_packet=pages_per_packet,
            banks=_DRAM_BANKS,
            chunk_start=start,
        )
        _shard_over_mesh(table, mesh_device)
        tables = ttnn.from_torch(
            torch.zeros(1, 1, table.shape[2] * n_dev, 1024, dtype=torch.int32),
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
            **common,
        )
        ttnn.experimental.high_bw_all_gather(table, dim=2, output_tensor=tables, cluster_axis=None, num_links=links)
        routed = lambda: ttnn.experimental.high_bw_all_gather(
            cache,
            output_tensor=routed_out,
            push_runs_tensor=tables,
            push_stripe_chunk=stripe_chunk,
            push_max_rows=cache_tokens,
            **gather_kw,
        )
        t_routed = timed(routed)

        if os.environ.get("TREE_PLAN") == "device":
            rank = ttnn.from_torch(
                torch.tensor([[[[d] + [0] * 15]] for d in range(n_dev)], dtype=torch.int32),
                dtype=ttnn.uint32,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
                **common,
            )
            paths = ttnn.from_torch(
                torch.from_numpy(_dim_path_tables(n_dev, mesh_shape).astype(np.uint32).view(np.int32)).reshape(
                    1, 1, n_dev, 256
                ),
                dtype=ttnn.uint32,
                mesh_mapper=replicate,
                **common,
            )
            tree_plan = lambda: ttnn.experimental.sparse_push_tree_plan(
                tables,
                idx_natural,
                rank,
                paths,
                mesh_rows=mesh_shape[0],
                mesh_cols=mesh_shape[1],
                stripe_chunk=stripe_chunk,
                max_rows=cache_tokens,
                pages_per_packet=pages_per_packet,
                banks=_DRAM_BANKS,
            )

            # Planning chains as each path runs them: today's plan op (with its rewrite) + table gather, and the plan op
            # without rewrite + table gather + the dimension-ordered planner (from those tables).
            def base_plan(rewrite):
                t, _ = ttnn.experimental.sparse_push_plan(
                    idx_natural,
                    num_devices=n_dev,
                    stripe_chunk=stripe_chunk,
                    max_rows=cache_tokens,
                    pages_per_packet=pages_per_packet,
                    banks=_DRAM_BANKS,
                    chunk_start=start,
                    rewrite=rewrite,
                )
                _shard_over_mesh(t, mesh_device)
                ttnn.experimental.high_bw_all_gather(t, dim=2, output_tensor=tables, cluster_axis=None, num_links=links)

            t_today_plan = timed(lambda: base_plan(True))
            t_slim_plan = timed(lambda: base_plan(False))
            base_plan(False)
            streams, idx_tree, *_ = tree_plan()
            ttnn.synchronize_device(mesh_device)
            t_plan = timed(lambda: tree_plan())
            t_chain = timed(lambda: (base_plan(False), tree_plan()))
            logger.info(
                f"layer {layer}: planning today (plan op + table gather) {t_today_plan:.0f} us; dimension-ordered: plan op "
                f"without rewrite + table gather {t_slim_plan:.0f} us + planner {t_plan:.0f} us = {t_chain:.0f} us"
            )
            ref, ref_maps, stats = _tree_plan_records(
                natural,
                n_dev,
                mesh_shape,
                stripe_chunk,
                pages_per_packet,
                buffer_rows,
                links=links,
                k_per=k_per,
                dim=True,
            )
            got = ttnn.to_torch(streams, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:, 0].numpy()
            ref = ref[:, 0].numpy()
            if os.environ.get("TREE_DUMP"):
                np.savez(os.environ["TREE_DUMP"], got=got, ref=ref)
            bad = 0
            for c in range(n_dev):
                for w in range(ref.shape[1]):
                    kinds = ref[c, w, 32::32]
                    end = 32 + 32 * (int(np.flatnonzero(kinds == 0)[0]) + 1)
                    if not np.array_equal(got[c, w, :end], ref[c, w, :end]):
                        bad += 1
                        if bad <= 3:
                            diff = np.flatnonzero(got[c, w, :end] != ref[c, w, :end])
                            logger.info(
                                f"chip {c} sender {w}: {diff.size} words differ, first {diff[:8].tolist()}: "
                                f"got {got[c, w, diff[:8]].tolist()} want {ref[c, w, diff[:8]].tolist()}"
                            )
            want_idx = np.where(natural >= 0, np.take_along_axis(ref_maps[:, None, :], np.maximum(natural, 0), 2), -1)
            got_idx = ttnn.to_torch(idx_tree, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2)).numpy()
            idx_bad = int((got_idx.reshape(n_dev, s_local, TOPK).astype(np.int64) != want_idx).sum())
            logger.info(
                f"layer {layer}: device tree plan {t_plan:.0f} us; {bad} sender pages differ, {idx_bad} indices"
            )
            assert bad == 0 and idx_bad == 0, "device tree plan differs from the host reference"
            tree = lambda: ttnn.experimental.high_bw_all_gather(
                cache, output_tensor=tree_out, push_runs_tensor=streams, **gather_kw
            )
            tree()
            ttnn.synchronize_device(mesh_device)
            t_tree = timed(tree)
            lines.append(
                f"  layer {layer}: routed {t_routed:6.0f} us, tree {t_tree:6.0f} us ({t_routed / t_tree:.2f}x); "
                f"device tree plan {t_plan:.0f} us"
            )
            logger.info(lines[-1])
            out_dense = ttnn.to_torch(
                _sparse_sdpa(q, dense_out, idx_natural, s_local * TP, kv_fmt),
                mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
            )
            out_tree = ttnn.to_torch(
                ttnn.transformer.sparse_sdpa(
                    q,
                    tree_out,
                    idx_tree,
                    V_DIM,
                    kv_format=_KV_FORMATS[kv_fmt][0],
                    scale=K_DIM**-0.5,
                    k_chunk_size=128,
                ),
                mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
            )
            assert torch.equal(out_dense, out_tree), f"layer {layer}: device-planned tree push differs"
            continue
        t0 = time.perf_counter()
        planner = {
            "local": _tree_plan_local,
            "level": partial(_tree_plan_local, per_packet=False),
            "records": _tree_plan_records,
            "dim": partial(_tree_plan_records, dim=True),
        }.get(os.environ.get("TREE_PLAN"), _tree_plan)
        streams_host, slot_maps, stats = planner(
            natural, n_dev, mesh_shape, stripe_chunk, pages_per_packet, buffer_rows, links=links, k_per=k_per
        )
        t_host = time.perf_counter() - t0
        streams = ttnn.from_torch(
            streams_host, dtype=ttnn.uint32, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0), **common
        )
        tree = lambda: ttnn.experimental.high_bw_all_gather(
            cache, output_tensor=tree_out, push_runs_tensor=streams, **gather_kw
        )
        logger.info(f"layer {layer}: first tree push (untraced)")
        tree()
        ttnn.synchronize_device(mesh_device)
        logger.info(f"layer {layer}: first tree push done")
        t_tree = timed(tree)
        lines.append(
            f"  layer {layer}: routed {t_routed:6.0f} us, tree {t_tree:6.0f} us ({t_routed / t_tree:.2f}x); "
            f"tree depth {stats['depth']}, {stats['packets']} packets, plan {t_host:.0f} s on host"
        )
        logger.info(lines[-1])

        # Exactness: sparse_sdpa over the tree's compact buffer equals sparse_sdpa over the dense gather.
        compact = np.where(natural >= 0, np.take_along_axis(slot_maps[:, None, :], np.maximum(natural, 0), 2), -1)
        idx_tree = ttnn.from_torch(
            torch.from_numpy(compact.reshape(1, 1, n_dev * s_local, TOPK).astype(np.int32)),
            dtype=ttnn.uint32,
            mesh_mapper=shard_seq,
            **common,
        )
        out_dense = ttnn.to_torch(
            _sparse_sdpa(q, dense_out, idx_natural, s_local * TP, kv_fmt),
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
        )
        out_tree = ttnn.to_torch(
            ttnn.transformer.sparse_sdpa(
                q, tree_out, idx_tree, V_DIM, kv_format=_KV_FORMATS[kv_fmt][0], scale=K_DIM**-0.5, k_chunk_size=128
            ),
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
        )
        assert torch.equal(out_dense, out_tree), f"layer {layer}: tree push differs from the dense-gather path"
        for t in (idx_natural, table, tables, streams, idx_tree):
            ttnn.deallocate(t)
    logger.info("\n" + "\n".join(lines))


def _tree_order(x, rel_dests, mesh_shape):
    """Device-computable tree order for row x: fewer edges, row parity on ties."""
    ny, nx = (len(_tree_edges(rel_dests, o, mesh_shape)) for o in ("yx", "xy"))
    return "yx" if ny < nx or (ny == nx and x & 1) else "xy"


_TREE_STAGE_ROWS = 65536  # staging rows per incoming direction


def _tree_plan_local(
    natural, n_dev, mesh_shape, stripe_chunk, pages_per_packet, buffer_rows, links=2, k_per=2, per_packet=True
):
    """Same contract as _tree_plan, with rules each chip can evaluate on its own (the device planner's):
    per-row tree order (_tree_order); requester b's buffer = [own rows | rows arriving from each direction] in
    regions of R_b pages, then fixed staging regions per direction; on each edge rows ordered by (depth, keep
    first, row) and packets dealt round-robin over the direction's senders; level h waits for every incoming
    stream to have delivered all of its packets of levels < h. With per_packet, a packet instead waits only for the
    packets that carried its rows here: the n-th packet of an edge is packet n // senders_per_dir of sender n % it."""
    R, C = mesh_shape
    B, P = _DRAM_BANKS, pages_per_packet
    unit = P * B
    per_dir = links * k_per
    senders = 4 * per_dir
    needed = [np.unique(v[v >= 0]) for v in natural.reshape(n_dev, -1)]
    req_of = np.concatenate([np.full(len(v), d) for d, v in enumerate(needed)])
    uniq, inv = np.unique(np.concatenate(needed), return_inverse=True)
    masks = np.zeros(len(uniq), dtype=np.int64)
    np.bitwise_or.at(masks, inv, np.left_shift(1, req_of).astype(np.int64))
    owners, local = _owner_local(uniq, stripe_chunk, n_dev)
    remote = masks & ~np.left_shift(1, owners).astype(np.int64)
    lane_rel = lambda j: (j % B) + (j // B) * unit
    region = [-(-(-(-(len(needed[b]) + 8 * P) // P)) // B) * unit for b in range(n_dev)]  # pages
    stage_pages = -(-(-(-_TREE_STAGE_ROWS // P)) // B) * unit
    keep_base = lambda b, dr: (1 + dr) * region[b]
    stage_base = lambda b, dr: 5 * region[b] + dr * stage_pages
    assert max(5 * r + 4 * stage_pages for r in region) <= buffer_rows, "tree regions exceed the buffer"
    neighbour = lambda a, dr: ((a // C + (dr == _S) - (dr == _N)) % R) * C + (a % C + (dr == _E) - (dr == _W)) % C

    # Edge records: (from a, dir, depth, keep, row id).
    recs = []
    max_depth = 0
    cache = {}
    for u in np.flatnonzero(remote):
        o, m = int(owners[u]), int(remote[u])
        ro, co = divmod(o, C)
        key = (o, m, int(uniq[u]) & 1)
        if key not in cache:
            rel = [(((d // C) - ro) % R, ((d % C) - co) % C) for d in range(n_dev) if m >> d & 1]
            order = _tree_order(int(uniq[u]), rel, mesh_shape)
            cache[key] = [
                (((a[0] + ro) % R) * C + (a[1] + co) % C, dr, ((b[0] + ro) % R) * C + (b[1] + co) % C, h)
                for a, dr, b, h in _tree_edges(rel, order, mesh_shape)
            ]
        for a, dr, b, h in cache[key]:
            recs.append((a, dr, h, 0 if m >> b & 1 else 1, int(uniq[u]), u))
            max_depth = max(max_depth, h)
    recs.sort()

    slot = np.full((n_dev, len(uniq)), -1, dtype=np.int64)  # where each row sits on each chip (page)
    packets = {}  # (a, dr) -> [(h, keep, target, row ids)]
    lanes = {}  # (b, dr, keep) -> lanes used
    for b in range(n_dev):  # requester's own rows
        ids = np.flatnonzero((owners == b) & (masks >> b & 1).astype(bool))
        for k, u in enumerate(ids):
            slot[b, u] = lane_rel(k // P) + (k % P) * B
    i = 0
    while i < len(recs):
        a, dr, h, keep = recs[i][:4]
        j = i
        while j < len(recs) and recs[j][:4] == (a, dr, h, keep):
            j += 1
        b = neighbour(a, dr)
        ids = [r[5] for r in recs[i:j]]
        base = keep_base(b, dr) if keep == 0 else stage_base(b, dr)
        for s0 in range(0, len(ids), P):
            lane = lanes.get((b, dr, keep), 0)
            lanes[(b, dr, keep)] = lane + 1
            target = base + lane_rel(lane)
            chunk = ids[s0 : s0 + P]
            for k, u in enumerate(chunk):
                slot[b, u] = target + k * B
            packets.setdefault((a, dr), []).append((h, keep, target, chunk))
        i = j
    assert max(lanes.get((b, dr, 1), 0) for b in range(n_dev) for dr in range(4)) * P <= _TREE_STAGE_ROWS

    items = [[[] for _ in range(senders)] for _ in range(n_dev)]
    sent = np.zeros((n_dev, senders, max_depth + 1), dtype=np.int64)  # packets per (stream, level)
    arrived = {}  # (chip, row id) -> (stream, packets of that stream up to and including the row's)
    for (a, dr), pk in packets.items():
        b = neighbour(a, dr)
        for n, (h, keep, target, chunk) in enumerate(pk):
            pick = n % per_dir
            w = ((pick // k_per) * 4 + dr) * k_per + pick % k_per
            for u in chunk:
                arrived[(b, u)] = (w, n // per_dir + 1)
    for (a, dr), pk in packets.items():
        for n, (h, keep, target, chunk) in enumerate(pk):
            pick = n % per_dir
            w = ((pick // k_per) * 4 + dr) * k_per + pick % k_per
            srcs = local[chunk] if h == 0 else slot[a, chunk]
            waits = {}
            if h and per_packet:
                for u in chunk:
                    s_, k_ = arrived[(a, u)]
                    waits[s_] = max(waits.get(s_, 0), k_)
            items[a][w].append((h, 1 if h == 0 else 2, target, srcs, waits))
            sent[a, w, h] += 1
    for b in range(n_dev):
        ids = np.flatnonzero((owners == b) & (masks >> b & 1).astype(bool))
        for p, s0 in enumerate(range(0, len(ids), P)):
            items[b][p % senders].append((0, 3, lane_rel(s0 // P), local[ids[s0 : s0 + P]], {}))

    # Incoming per (stream, level) at chip c: sent by the neighbour whose direction points at c.
    incoming = np.zeros((n_dev, senders, max_depth + 1), dtype=np.int64)
    for a in range(n_dev):
        for w in range(senders):
            incoming[neighbour(a, (w // k_per) % 4), w] += sent[a, w]
    pages = []
    for c in range(n_dev):
        cum = np.cumsum(incoming[c], axis=1)
        for w in range(senders):
            words = [0] + [int(x) for x in cum[:, -1]] + [0] * (31 - senders)
            its = sorted(items[c][w], key=lambda it: it[0])
            level = 0
            waited = {}
            for h, kind, target, srcs, waits in its:
                while level < h:
                    words.append(6)
                    level += 1
                    if not per_packet:
                        for s in range(senders):
                            if cum[s, level - 1]:
                                words += [4, s, int(cum[s, level - 1])]
                for s, k in sorted(waits.items()):
                    if k > waited.get(s, 0):
                        words += [4, s, k]
                        waited[s] = k
                words += [kind, int(target), len(srcs)] + [int(x) for x in srcs]
            while level < max_depth:
                words.append(6)
                level += 1
            words.append(0)
            pages.append(words)
    width = max(len(w) for w in pages)
    width += -width % 16
    host = np.zeros((n_dev * senders, width), dtype=np.int64)
    for k, w in enumerate(pages):
        host[k, : len(w)] = w
    slot_maps = np.full((n_dev, natural.max() + 1), -1, dtype=np.int64)
    for d in range(n_dev):
        ids = np.searchsorted(uniq, needed[d])
        assert (slot[d, ids] >= 0).all(), "requester row without a slot"
        slot_maps[d, needed[d]] = slot[d, ids]
    stats = dict(depth=max_depth + 1, packets=int(sent.sum()), stream_words=width)
    return torch.from_numpy(host.reshape(n_dev, 1, senders, width).astype(np.int32)), slot_maps, stats


_TREE_LEVELS = 12  # tree depths a plan may use
_DIM_LEVELS = 6  # dimension-ordered paths: at most 4 + 2 hops (sparse_push_tree_plan's levels)


def _dim_path(o, d, mesh_shape):
    """Dimension-ordered path owner o -> chip d as [(from, dir, to)]: along the owner's column to d's row, then along
    that row, each the short way round; half-way ties go S / E when the owner's row + column is even."""
    R, C = mesh_shape
    (ro, co), (rd, cd) = divmod(o, C), divmod(d, C)
    fwd = (ro + co) % 2 == 0
    path, r, c = [], ro, co
    s = (rd - ro) % R
    south = s < R - s or (s == R - s and fwd)
    for _ in range(s if south else R - s):
        nr = (r + (1 if south else -1)) % R
        path.append((r * C + c, _S if south else _N, nr * C + c))
        r = nr
    e = (cd - co) % C
    east = e < C - e or (e == C - e and fwd)
    for _ in range(e if east else C - e):
        nc = (c + (1 if east else -1)) % C
        path.append((r * C + c, _E if east else _W, r * C + nc))
        c = nc
    return path


def _dim_path_tables(n_dev, mesh_shape):
    """Per chip a, for sparse_push_tree_plan: [via x32 | in_of x32 | depth x32 | beyond x128] over owners o, where via
    = requesters whose path from o passes a, in_of = arrival dir | hop << 8 (4 = a is o), depth = hops o -> a, beyond
    [4 o + dir] = requesters whose path continues from a in that direction."""
    t = np.zeros((n_dev, 256), dtype=np.int64)
    t[:, 32:64] = 4
    for o in range(n_dev):
        for d in range(n_dev):
            if d == o:
                continue
            for h, (u, dr, v) in enumerate(_dim_path(o, d, mesh_shape)):
                t[v, o] |= 1 << d
                t[v, 32 + o] = dr | h << 8
                t[v, 64 + o] = h + 1
                t[u, 96 + 4 * o + dr] |= 1 << d
    return t


def _dim_edges(o, mask, n_dev, mesh_shape):
    """Edges (from, dir, to, depth) of the union of o's dimension-ordered paths to the requesters in mask."""
    edges = {}
    for d in range(n_dev):
        if mask >> d & 1:
            for h, (u, dr, v) in enumerate(_dim_path(o, d, mesh_shape)):
                edges[(u, dr, v)] = h
    return [(u, dr, v, h) for (u, dr, v), h in edges.items()]


def _tree_plan_records(
    natural, n_dev, mesh_shape, stripe_chunk, pages_per_packet, buffer_rows, links=2, k_per=2, dim=False
):
    """Reference for the device tree planner (SPARSE_PUSH_TREE=2 stream format): _tree_plan_local's rules, with
    fixed-size records so every row's words have a position computable from per-edge group counts alone.

    Sender page: words 1..16 = packets this chip receives per neighbour sender; 128 B record r at word 32 + 32 r:
    [kind, target, n, 0 x 5, (src, arrival) x P, 0...], arrival = stream << 24 | packets of that stream through the
    row's.
    Sender w holds this chip's local packets q = w, w + 16, ..., then its edge's packets n = pick, pick + 4, ... (pick
    = w's index among the 4 senders of its direction) with a level marker after each level and an end record. On an
    edge, rows are grouped by (level, keep), ascending; packet n and the receiver lane follow from group counts."""
    R, C = mesh_shape
    B, P, LV = _DRAM_BANKS, pages_per_packet, (_DIM_LEVELS if dim else _TREE_LEVELS)
    unit, RW = P * B, 32
    assert 8 + 2 * P <= RW
    per_dir = links * k_per
    senders = 4 * per_dir
    needed = [np.unique(v[v >= 0]) for v in natural.reshape(n_dev, -1)]
    req_of = np.concatenate([np.full(len(v), d) for d, v in enumerate(needed)])
    uniq, inv = np.unique(np.concatenate(needed), return_inverse=True)
    masks = np.zeros(len(uniq), dtype=np.int64)
    np.bitwise_or.at(masks, inv, np.left_shift(1, req_of).astype(np.int64))
    owners, local = _owner_local(uniq, stripe_chunk, n_dev)
    remote = masks & ~np.left_shift(1, owners).astype(np.int64)
    lane_rel = lambda j: (j % B) + (j // B) * unit
    region = [-(-(-(-len(needed[b]) // P) + LV) // B) * unit for b in range(n_dev)]
    stage = -(-(-(-_TREE_STAGE_ROWS // P) + LV) // B) * unit
    base = lambda b, dr, keep: (1 + dr) * region[b] if keep == 0 else 5 * region[b] + dr * stage
    assert max(5 * r + 4 * stage for r in region) <= buffer_rows, "tree regions exceed the buffer"
    nb = lambda a, dr: ((a // C + (dr == _S) - (dr == _N)) % R) * C + (a % C + (dr == _E) - (dr == _W)) % C
    stream = lambda dr, n: (((n % per_dir) // k_per) * 4 + dr) * k_per + (n % per_dir) % k_per

    edges_of = {}
    for u in np.flatnonzero(remote):
        o, m, x = int(owners[u]), int(remote[u]), int(uniq[u])
        ro, co = divmod(o, C)
        key = (o, m, x & 1)
        if key not in edges_of and dim:
            edges_of[key] = _dim_edges(o, m, n_dev, mesh_shape)
        elif key not in edges_of:
            rel = [(((d // C) - ro) % R, ((d % C) - co) % C) for d in range(n_dev) if m >> d & 1]
            edges_of[key] = [
                (((a[0] + ro) % R) * C + (a[1] + co) % C, dr, ((b[0] + ro) % R) * C + (b[1] + co) % C, h)
                for a, dr, b, h in _tree_edges(rel, _tree_order(x, rel, mesh_shape), mesh_shape)
            ]
    # Group counts per (chip, dir, level, keep) and each (row, edge)'s rank in its group (rows ascending).
    count = np.zeros((n_dev, 4, LV, 2), dtype=np.int64)
    on_edge = []  # (a, dr, h, keep, rank, row id)
    for u in np.flatnonzero(remote):  # uniq is ascending, so ranks come out in row order
        m = int(remote[u])
        for a, dr, b, h in edges_of[(int(owners[u]), m, int(uniq[u]) & 1)]:
            assert h < LV
            keep = 0 if m >> b & 1 else 1
            on_edge.append((a, dr, h, keep, int(count[a, dr, h, keep]), u))
            count[a, dr, h, keep] += 1
    pk = -(-count // P)  # packets per group
    flat = pk.reshape(n_dev, 4, 2 * LV)
    pk_before = (np.cumsum(flat, axis=2) - flat).reshape(n_dev, 4, LV, 2)  # packets of earlier groups on the edge
    lanes_before = np.cumsum(pk, axis=2) - pk  # lanes of earlier levels in the same (keep) region

    slot = np.full((n_dev, len(uniq)), -1, dtype=np.int64)
    arrival = np.zeros((n_dev, len(uniq)), dtype=np.int64)
    for a, dr, h, keep, k, u in on_edge:
        b = nb(a, dr)
        n = pk_before[a, dr, h, keep] + k // P
        slot[b, u] = base(b, dr, keep) + lane_rel(lanes_before[a, dr, h, keep] + k // P) + (k % P) * B
        arrival[b, u] = (stream(dr, n) << 24) | (n // per_dir + 1)
    own = [np.flatnonzero((owners == b) & (masks >> b & 1).astype(bool)) for b in range(n_dev)]
    for b in range(n_dev):
        slot[b, own[b]] = lane_rel(np.arange(len(own[b])) // P) + (np.arange(len(own[b])) % P) * B

    total = pk.sum(axis=(2, 3))  # packets per edge
    incoming = np.zeros((n_dev, senders), dtype=np.int64)
    for a in range(n_dev):
        for dr in range(4):
            for n in range(int(total[a, dr])):
                incoming[nb(a, dr), stream(dr, n)] += 1
    nlocal = -(-np.array([len(o) for o in own]) // P)
    locals_of = lambda a, w: max(0, -(-(int(nlocal[a]) - w) // senders))
    records = int(max(locals_of(a, 0) for a in range(n_dev)) + LV + -(-total.max() // per_dir) + 1)
    host = np.zeros((n_dev, senders, 32 + records * RW), dtype=np.int64)
    rec = lambda a, w, r: host[a, w, 32 + r * RW : 32 + (r + 1) * RW]
    for a in range(n_dev):
        host[a, :, 1 : 1 + senders] = incoming[a]
        for q in range(int(nlocal[a])):
            rows = own[a][q * P : (q + 1) * P]
            r = rec(a, q % senders, q // senders)
            r[:3] = (3, lane_rel(q), len(rows))
            r[8 : 8 + 2 * len(rows) : 2] = local[rows]
        for w in range(senders):
            dr, pick = (w // k_per) % 4, (w // (4 * k_per)) * k_per + w % k_per
            done_before = 0
            for h in range(LV):
                done_before = -(-(pk[a, dr, : h + 1].sum() - pick) // per_dir)
                if h < LV - 1:
                    rec(a, w, locals_of(a, w) + h + done_before)[0] = 6
            rec(a, w, locals_of(a, w) + LV - 1 + done_before)[0] = 0
    for a, dr, h, keep, k, u in on_edge:
        n = pk_before[a, dr, h, keep] + k // P
        w = stream(dr, n)
        first = k - k % P
        r = rec(a, w, locals_of(a, w) + h + n // per_dir)
        if k % P == 0:
            r[:3] = (
                1 if h == 0 else 2,
                base(nb(a, dr), dr, keep) + lane_rel(lanes_before[a, dr, h, keep] + k // P),
                min(P, count[a, dr, h, keep] - first),
            )
        r[8 + 2 * (k % P)] = local[u] if h == 0 else slot[a, u]
        r[9 + 2 * (k % P)] = 0 if h == 0 else arrival[a, u]
    slot_maps = np.full((n_dev, natural.max() + 1), -1, dtype=np.int64)
    for d in range(n_dev):
        ids = np.searchsorted(uniq, needed[d])
        assert (slot[d, ids] >= 0).all(), "requester row without a slot"
        slot_maps[d, needed[d]] = slot[d, ids]
    stats = dict(
        depth=int((count.sum(axis=(0, 1, 3)) > 0).nonzero()[0].max() + 1),
        packets=int(total.sum()),
        stream_words=host.shape[-1],
    )
    return torch.from_numpy(host.reshape(n_dev, 1, senders, -1).astype(np.int32)), slot_maps, stats
