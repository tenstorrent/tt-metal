# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced GLM-5.2 indexer top-k (``topk_large_indices``) chunk / KV-prefix sweep on the 8x4 mesh.

Mirrors ``TtIndexer.select_local`` on the production trace path: per-device logits
``[1, 1, chunk/32, T]`` bf16 ROW_MAJOR DRAM (full-mesh 32-way query-row striping), k=2048,
``valid_length_tensor`` = actual_start, ``valid_length_offset`` = chunk, ``valid_end_tensor`` = real end.
Production runs top-k on the 80-core sparse-MLA overlap grid (0,0)-(7,9) of an 80/40 sub-device
split; ``full120`` runs it on the whole 12x10 grid for comparison.

Work is split by rows only (whole rows per core, no cross-core merge), and each row costs
ceil(valid_length / 2048) sort+merge steps, so the critical path is ceil(rows / cores) rows.

Flow: warm-up, capture one top-k into a trace, replay RING_MLA_SWEEP_ITERS times, each replay in
its own realtime-profiler window. TOPK_COMPUTE_ONLY=1 stubs the row reads and index writes.

    pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_glm_topk_chunk_sweep.py -k "chunk_sweep and prod80"
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
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

SP, TP = 8, 4
K = 2048  # GLM-5.2 index_topk
LLK_CHUNK = 2048  # elements per sort step (LLK K for k=2048)
TARGET_PREFIX = 51200

ITERS = int(os.environ.get("RING_MLA_SWEEP_ITERS", "10"))
COMPUTE_ONLY = os.environ.get("TOPK_COMPUTE_ONLY", "0") == "1"

# GLM_SP_BATCH=1 keeps the 8x4 mesh but gives each TP column its own request: rows are split over
# the 8 rows only, so each device takes 4x the tokens and four requests share the machine.
# This op is device-local with replicated inputs, so only the row count changes.
SP_BATCH = os.environ.get("GLM_SP_BATCH", "0") == "1"
SHARD_N = SP if SP_BATCH else SP * TP

RESULTS_DIR = Path(os.environ.get("GLM_TOPK_SWEEP_OUT", "generated/glm_topk_sweep"))

CHUNKS = [1024, 2048, 3072, 4096, 5120]
if os.environ.get("GLM_CHUNKS"):
    CHUNKS = [int(c) for c in os.environ["GLM_CHUNKS"].split(",") if c.strip()]
GRIDS = ["prod80", "full120"]
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


def _replicated(mesh_device, torch_tensor, dtype):
    return ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.parametrize("mesh_device", [pytest.param((SP, TP), id="8x4")], indirect=True)
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("chunk", CHUNKS, ids=[f"chunk{c}" for c in CHUNKS])
@pytest.mark.timeout(900)
def test_glm_topk_chunk_sweep(mesh_device, grid, chunk):
    _run_topk(mesh_device, grid, chunk, round(TARGET_PREFIX / chunk) * chunk)


@pytest.mark.parametrize("mesh_device", [pytest.param((SP, TP), id="8x4")], indirect=True)
@pytest.mark.parametrize("chunk, prefix", _prefix_sweep_params())
@pytest.mark.timeout(900)
def test_glm_topk_prefix_sweep(mesh_device, chunk, prefix):
    _run_topk(mesh_device, "prod80", chunk, prefix)


def _run_topk(mesh_device, grid, chunk, prefix):
    torch.manual_seed(1234)
    mesh_device.enable_program_cache()

    rows = chunk // SHARD_N
    valid = prefix + chunk
    # T = model capacity (always >= k); the reader only touches the valid prefix, T sets the row stride.
    width = max(valid, K)
    full = mesh_device.compute_with_storage_grid_size()

    manager = None
    if grid == "prod80":
        topk_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, full.y - 1))])
        gather_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(full.x - 1, full.y - 1))])
        manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([topk_grid]), ttnn.SubDevice([gather_grid])], 0)
        mesh_device.load_sub_device_manager(manager)
        topk_kwargs = {"subdevice_id": ttnn.SubDeviceId(0), "sub_core_grids": topk_grid}
        num_cores = 8 * full.y
        grid_str = f"8x{full.y}"
    else:
        topk_kwargs = {}
        num_cores = full.x * full.y
        grid_str = f"{full.x}x{full.y}"

    try:
        tt_logits = _replicated(mesh_device, torch.randn(1, 1, rows, width), ttnn.bfloat16)
        start_md = _replicated(mesh_device, torch.tensor([prefix], dtype=torch.int64).reshape(1, 1, 1, 1), ttnn.uint32)
        end_md = _replicated(mesh_device, torch.tensor([valid], dtype=torch.int64).reshape(1, 1, 1, 1), ttnn.uint32)

        def run_once():
            return ttnn.experimental.topk_large_indices(
                tt_logits,
                k=K,
                valid_length_tensor=start_md,
                valid_length_offset=chunk,
                valid_end_tensor=end_md,
                **topk_kwargs,
            )

        try:
            warm = run_once()
        except RuntimeError as e:
            pytest.skip(f"rejected by topk_large_indices: {str(e).splitlines()[0][:300]}")
        ttnn.synchronize_device(mesh_device)
        assert list(warm.shape) == [1, 1, rows, K], f"unexpected output shape {warm.shape}"
        ttnn.deallocate(warm)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_out = run_once()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        def replay_once():
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

        durations_ns = []
        try:
            # Separate windows: replays share a runtime id and would merge into one record.
            profile_realtime_program_merged(mesh_device, replay_once, record_timeout_seconds=30.0)
            for _ in range(ITERS):
                _, programs = profile_realtime_program_merged(mesh_device, replay_once, record_timeout_seconds=30.0)
                durations_ns.append(max(p["duration_ns"] for p in programs.values()))
        finally:
            ttnn.release_trace(mesh_device, trace_id)
            ttnn.deallocate(trace_out)
    finally:
        if manager is not None:
            ttnn.synchronize_device(mesh_device)
            mesh_device.clear_loaded_sub_device_manager()
            mesh_device.remove_sub_device_manager(manager)

    median_ns = statistics.median(durations_ns)
    active = min(num_cores, rows)
    rows_per_core = math.ceil(rows / num_cores)
    sort_steps = math.ceil(valid / LLK_CHUNK)
    row = {
        "chunk": chunk,
        "rows": rows,
        "prefix": prefix,
        "valid_length": valid,
        "grid": grid,
        "core_grid": grid_str,
        "cores": num_cores,
        "active_cores": active,
        "rows_per_core": rows_per_core,
        "core_util": round(rows / (num_cores * rows_per_core), 4),
        "sort_steps_per_row": sort_steps,
        "compute_only": COMPUTE_ONLY,
        "median_us": round(median_ns / 1e3, 2),
        "min_us": round(min(durations_ns) / 1e3, 2),
        "max_us": round(max(durations_ns) / 1e3, 2),
        # Critical-path core cost per 2048-element sort step: flat when the op scales cleanly.
        "ns_per_step": round(median_ns / (rows_per_core * sort_steps), 1),
        "logits_read_gbps": round(rows * valid * 2 / median_ns, 2),
    }
    logger.info(f"glm topk sweep: {row}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rt_results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
