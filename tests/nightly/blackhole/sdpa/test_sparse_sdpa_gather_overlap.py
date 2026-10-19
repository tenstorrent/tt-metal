# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa on a restricted core grid, and sparse_sdpa overlapped with a next-layer KVPE prefetch.

GLM-5.3 prefill shape on the 8x4 Galaxy (SP8 x TP4, 64 heads resharded head->seq, TOPK 2048, latent 512 +
rope 64). The overlap case captures each configuration as a trace and times back-to-back replays on the host:
neither the realtime profiler nor the Tracy device profiler reports traced sub-device-1 gathers correctly.
"""

import os
import time

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


def _sparse_sdpa(q, kv, idx, chunk_local, k_chunk=128, **grid):
    return ttnn.transformer.sparse_sdpa(
        q,
        kv,
        idx,
        V_DIM,
        kv_format=ttnn.transformer.SparseKVFormat.BF16,
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


def _glm_inputs(mesh_device, chunk_global, cache_tokens):
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
    tt_kv = ttnn.from_torch(
        torch.randn(1, 1, cache_tokens, K_DIM, generator=gen).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        **common,
    )
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
def test_sparse_sdpa_glm_core_count_perf(mesh_device, chunk_global):
    """sparse_sdpa alone on 120 / 100 / 80 cores at the GLM shape."""
    grid = mesh_device.compute_with_storage_grid_size()
    tt_q, tt_kv, tt_idx, chunk_local = _glm_inputs(mesh_device, chunk_global, cache_tokens=chunk_global * 51)
    rows = []
    for name, sub in [("full", None), ("100", _cols(0, 9, grid.y)), ("80", _cols(0, 7, grid.y))]:
        kw = {} if sub is None else {"sub_core_grids": sub}
        run = lambda: _sparse_sdpa(tt_q, tt_kv, tt_idx, chunk_local, **kw)
        run()
        ttnn.synchronize_device(mesh_device)
        samples = sorted(_program_us(mesh_device, run, "sparse_sdpa") for _ in range(5))
        rows.append((name, samples[len(samples) // 2]))
    logger.info(f"sparse_sdpa GLM {chunk_global} per-program median us: {rows}")


@run_for_blackhole()
@_PERF
@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@_CHUNKS
@pytest.mark.parametrize("prefix_slabs", [10, 40], ids=["prefix10slabs", "prefix40slabs"])
def test_sparse_sdpa_overlap_with_kvpe_prefetch(mesh_device, chunk_global, prefix_slabs):
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

    tt_q, tt_kv, tt_idx, chunk_local = _glm_inputs(mesh_device, chunk_global, cache_tokens)
    # Two-layer KVPE cache striped over all 32 chips (the gather source) and the second gather buffer.
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    host_cache = torch.randn(2, 1, cache_tokens, K_DIM).to(torch.bfloat16)
    cache = ttnn.from_torch(
        host_cache,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        **common,
    )
    next_buffer = ttnn.from_torch(
        torch.zeros(1, 1, cache_tokens, K_DIM).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        **common,
    )

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
        tt_q, tt_kv, tt_idx, chunk_local, subdevice_id=ttnn.SubDeviceId(0), sub_core_grids=sdpa_grid
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
            f"GLM chunk {chunk_global}, prefix {prefix_slabs} slabs: sdpa {base_sdpa:.0f} + gather {base_gather:.0f} = {base:.0f}us"
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
        # The overlapped prefetch + resume must have gathered layer 1's prefix exactly.
        local_rows = cache_tokens // n_dev
        rows = (prefix_slabs + 1) * (slab_global // n_dev)
        shards = ttnn.get_device_tensors(next_buffer)
        for chip in (0, n_dev - 1):
            actual = ttnn.to_torch(shards[chip])
            for rank in range(n_dev):
                lo = rank * local_rows
                assert torch.equal(actual[0, 0, lo : lo + rows], host_cache[1, 0, lo : lo + rows]), (chip, rank)
        signpost("done")
    finally:
        ttnn.synchronize_device(mesh_device)
        for tid in traces.values():
            ttnn.release_trace(mesh_device, tid)
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)
