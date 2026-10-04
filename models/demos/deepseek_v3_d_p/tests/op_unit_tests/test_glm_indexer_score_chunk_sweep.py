# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced GLM-5.2 indexer score (``ring_indexer_score_dsa``) chunk / KV-prefix sweep on the 8x4 mesh.

Mirrors ``TtIndexer.score`` on the fused full-mesh route (``cluster_axis=None``): per device
q ``[1, 32, chunk/32, 128]`` bfp8, gate weights ``[1, 1, chunk/32, 32]`` bf16, the deduped index-key cache
``[1, 1, T/32, 128]`` bfp8 striped over all 32 devices, and a replicated full-T gathered-K scratch.
T = GLM-5.2's 1M context (rounded to whole chunks), which also sizes the op's K-band schedule.
Scalar bounds: chunk_start_idx = prefix, kv_len = prefix + chunk (chunk-aligned starts).

Flow: warm-up, capture one score into a trace, replay RING_MLA_SWEEP_ITERS times, one realtime-profiler
window per replay. INDEXER_SCORE_COMPUTE_ONLY=1 stubs every read/write, the mcasts and the fused gather.

    MESH_DEVICE=TG pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_glm_indexer_score_chunk_sweep.py -k "chunk_sweep and chunk5120"
"""

import json
import math
import os
import statistics
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import get_indexer_key_chunk
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.test_ring_indexer_score_dsa import (
    _close_full_mesh_ccl,
    _open_full_mesh_ccl,
)
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

SP, TP = 8, 4
RING = SP * TP

# GLM_SP_BATCH=1 keeps the 8x4 mesh and fabric but runs each TP column as an independent request:
# Q/W replicated across columns, KV TP-replicated and sharded only on rows, gather on the SP axis.
# Four requests then share the machine instead of one spanning it.
SP_BATCH = os.environ.get("GLM_SP_BATCH", "0") == "1"
SHARD_N = SP if SP_BATCH else SP * TP  # devices the sequence is split across
CLUSTER_AXIS = 0 if SP_BATCH else None  # SP-axis gather vs full-mesh gather
KV_TP_AXIS = None if SP_BATCH else 1  # None = TP-replicated, 1 = also striped across TP

HEADS = GLM52Config.INDEX_N_HEADS  # 32
HEAD_DIM = GLM52Config.INDEX_HEAD_DIM  # 128
MAX_CONTEXT = GLM52Config.MAX_POSITION_EMBEDDINGS  # 1M
TARGET_PREFIX = 51200
TRACE_REGION_SIZE = 16 * 1024 * 1024
BH_CLOCK_GHZ = 1.35
LOFI_MUL_ADDS_PER_CYCLE_PER_CORE = 4096

ITERS = int(os.environ.get("RING_MLA_SWEEP_ITERS", "10"))
COMPUTE_ONLY = os.environ.get("INDEXER_SCORE_COMPUTE_ONLY", "0") == "1"
RESULTS_DIR = Path(os.environ.get("GLM_INDEXER_SWEEP_OUT", "generated/glm_indexer_score_sweep"))

CHUNKS = [1024, 2048, 3072, 4096, 5120]
if os.environ.get("GLM_CHUNKS"):
    CHUNKS = [int(c) for c in os.environ["GLM_CHUNKS"].split(",") if c.strip()]
PREFIX_TARGETS = [0, 2048, 4096, 8192, 16384, 32768, 51200, 65536, 102400, 131072, 196608, 262144]
# GLM_PREFIX_TARGETS / GLM_CHUNKS override the sweep grids (comma-separated) for long-context runs.
if os.environ.get("GLM_PREFIX_TARGETS"):
    PREFIX_TARGETS = [int(t) for t in os.environ["GLM_PREFIX_TARGETS"].split(",") if t.strip()]


def _prefix_sweep_params():
    params = []
    for chunk in CHUNKS:
        for prefix in sorted({round(t / chunk) * chunk for t in PREFIX_TARGETS}):
            params.append(pytest.param(chunk, prefix, id=f"chunk{chunk}-prefix{prefix}"))
    return params


def _prod_program_config(rows, chunk):
    """indexer.py: q_chunk 64 when the TP-split row count allows it, else 32; k_chunk from DSA_INDEXER_CONFIG."""
    q_chunk = int(os.environ.get("GLM_INDEXER_Q_CHUNK", 64 if rows % 64 == 0 else 32))
    k_chunk = int(os.environ.get("GLM_INDEXER_K_CHUNK", min(get_indexer_key_chunk(HEADS), chunk)))
    return q_chunk, k_chunk


def _largest_divisor_leq(value, cap):
    return max(d for d in range(1, min(value, cap) + 1) if value % d == 0)


def _ideal_cycles(grid, rows, capacity, k_chunk, first_pos, kv_len):
    """Mirror of the nightly _ring_indexer_ideal_compute_cycles (the op's fusion-aware LoFi model) for the
    critical device: its q tiles against every causally visible K tile, over the score cores that are left
    after the 4 fused all-gather workers take their column."""
    q_tiles = rows // 32
    kv_tiles = kv_len // 32
    first_tiles = first_pos // 32
    valid_tiles = sum(min(kv_tiles, first_tiles + row + 1) for row in range(q_tiles))
    k_bands = math.ceil((capacity // 32) / (k_chunk // 32))
    compute_grid_x = grid.x - math.ceil(4 / grid.y)
    group_rows = _largest_divisor_leq(q_tiles, grid.y)
    band_columns = min(k_bands, compute_grid_x)
    row_blocks = max(1, min(grid.y // group_rows, k_bands // band_columns))
    core_count = group_rows * row_blocks * band_columns
    mul_adds = 2 * valid_tiles * HEADS * (32 * 32) * HEAD_DIM
    return math.ceil(mul_adds / (core_count * LOFI_MUL_ADDS_PER_CYCLE_PER_CORE)), core_count


@pytest.mark.parametrize("chunk", CHUNKS, ids=[f"chunk{c}" for c in CHUNKS])
@pytest.mark.timeout(900)
def test_glm_indexer_score_chunk_sweep(chunk):
    _run_indexer_score(chunk, round(TARGET_PREFIX / chunk) * chunk)


@pytest.mark.parametrize("chunk, prefix", _prefix_sweep_params())
@pytest.mark.timeout(900)
def test_glm_indexer_score_prefix_sweep(chunk, prefix):
    _run_indexer_score(chunk, prefix)


def _run_indexer_score(chunk, prefix):
    # Needs at least RING devices, not exactly RING: GLM_SP_BATCH runs the same 32-chip mesh as four
    # independent 8-device columns, so an equality check here would reject a machine that fits fine.
    if ttnn.get_num_devices() < RING:
        pytest.skip(f"needs at least {RING} devices, found {ttnn.get_num_devices()}")
    torch.manual_seed(1234)
    rows = chunk // SHARD_N
    capacity = MAX_CONTEXT // chunk * chunk
    kv_len = prefix + chunk
    assert kv_len <= capacity
    q_chunk, k_chunk = _prod_program_config(rows, chunk)

    mesh, semaphores, subdevice_id, stall_group = _open_full_mesh_ccl((SP, TP), trace_region_size=TRACE_REGION_SIZE)
    try:
        mesh.enable_program_cache()
        grid = mesh.compute_with_storage_grid_size()
        # SP-batch: split the sequence down the rows only, replicating each request across its column.
        shard = (
            ttnn.ShardTensor2dMesh(mesh, mesh_shape=(SP, TP), dims=[2, None])
            if SP_BATCH
            else ttnn.ShardTensorToMesh(mesh, dim=2)
        )
        q_dev = ttnn.from_torch(
            torch.randn(1, HEADS, chunk, HEAD_DIM, dtype=torch.bfloat16),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        w_dev = ttnn.from_torch(
            torch.randn(1, 1, chunk, HEADS, dtype=torch.bfloat16),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        k_local = init_kvpe_cache(
            kvpe_cache_head_dim=HEAD_DIM,
            mesh_device=mesh,
            seq_len=capacity,
            mesh_shape=(SP, TP),
            sp_axis=0,
            num_kvpe_cache_layers=1,
            num_users=1,
            dtype=ttnn.bfloat8_b,
            tp_axis=KV_TP_AXIS,
        )
        k_full = ttnn.from_torch(
            torch.zeros(1, 1, capacity, HEAD_DIM),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        program_config = ttnn.IndexerScoreProgramConfig(q_chunk_size=q_chunk, k_chunk_size=k_chunk, head_group_size=0)

        def run_once():
            return ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_full,
                w_dev,
                k_local,
                semaphores,
                cluster_axis=CLUSTER_AXIS,
                topology=ttnn.Topology.Ring,
                num_links=2,
                ag_sub_device_id=subdevice_id,
                program_config=program_config,
                chunk_start_idx=prefix,
                kv_len=kv_len,
                cache_batch_idx=0,
                index_cache_num_layers=1,
                index_cache_layer_idx=0,
                seq_subshard_axis=None,
                # The op requires sp_axis and chunk_local to be set together; the SP-axis route
                # reaches that check, the full-mesh route does not.
                block_cyclic_sp_axis=0 if SP_BATCH else None,
                block_cyclic_chunk_local=rows,
                block_cyclic_cache_tp_sharded=False,
            )

        try:
            warm = run_once()
        except RuntimeError as e:
            pytest.skip(f"rejected by ring_indexer_score_dsa: {str(e).splitlines()[0][:300]}")
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        assert list(warm.shape) == [1, 1, rows, capacity], f"unexpected logits shape {warm.shape}"
        ttnn.deallocate(warm)

        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_out = run_once()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)

        def replay_once():
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)

        durations_ns = []
        try:
            # Separate windows: replays share a runtime id and would merge into one record.
            profile_realtime_program_merged(mesh, replay_once, record_timeout_seconds=30.0)
            for _ in range(ITERS):
                _, programs = profile_realtime_program_merged(mesh, replay_once, record_timeout_seconds=30.0)
                durations_ns.append(max(p["duration_ns"] for p in programs.values()))
        finally:
            ttnn.release_trace(mesh, trace_id)
            ttnn.deallocate(trace_out)
    finally:
        _close_full_mesh_ccl(mesh)

    median_ns = statistics.median(durations_ns)
    # Chunk-aligned start: device d (row-major) scores positions prefix + d*rows ...; the last one is critical.
    first_pos = prefix + (RING - 1) * rows
    ideal_cycles, score_cores = _ideal_cycles(grid, rows, capacity, k_chunk, first_pos, kv_len)
    row = {
        "chunk": chunk,
        "rows": rows,
        "prefix": prefix,
        "kv_len": kv_len,
        "capacity": capacity,
        "q_chunk": q_chunk,
        "k_chunk": k_chunk,
        "core_grid": f"{grid.x}x{grid.y}",
        "score_cores": score_cores,
        "q_tile_groups": rows // 32,
        "compute_only": COMPUTE_ONLY,
        "median_us": round(median_ns / 1e3, 2),
        "min_us": round(min(durations_ns) / 1e3, 2),
        "max_us": round(max(durations_ns) / 1e3, 2),
        "ideal_us": round(ideal_cycles / BH_CLOCK_GHZ / 1e3, 2),
        "fpu_util_pct": round(ideal_cycles / (median_ns * BH_CLOCK_GHZ) * 100, 2),
    }
    logger.info(f"glm indexer score sweep: {row}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rt_results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
