# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced GLM-5.2 sparse attention (``ttnn.transformer.sparse_sdpa``) chunk / KV-prefix sweep on 8x4.

Mirrors ``ttMLA._sparse_mla`` after the head->sequence reshard: per device q ``[1, 64, chunk/32, 576]``
bf16 RM, top-k indices ``[1, 1, chunk/32, 2048]`` uint32 RM (logical positions, 0xFFFFFFFF tail),
replicated BF16_RM KVPE buffer ``[1, 1, T, 576]`` in block-cyclic order (tp-sharded, 32 stripes),
v_dim=512, scale=256**-0.5, k_chunk_size=128, default kernel config (HiFi2), full 12x10 grid.

The op splits query rows (tokens) over the 120 cores; each core runs all 64 heads of its tokens
and gathers each selected key row from DRAM by index. Row r gets nv = min(pos + 1, 2048) keys; every
device holds rows of the last stripe here (the slowest device sets the critical path).

Flow: warm-up, capture one sparse_sdpa into a trace, replay RING_MLA_SWEEP_ITERS times, one realtime
profiler window per replay. SPARSE_SDPA_COMPUTE_ONLY=1 stubs the Q reads, KV gather and output writes.

    pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_glm_sparse_sdpa_chunk_sweep.py -k "chunk_sweep and chunk5120-kc128"
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
NUM_HEADS = 64
KVPE_DIM = 576
V_DIM = 512
TOPK = 2048
SCALE = 256**-0.5  # GLM-5.2 qk_head_dim = 192 + 64; rope factor 1.0 -> no mscale
SENTINEL = 0xFFFFFFFF
TARGET_PREFIX = 51200
BH_CLOCK_GHZ = 1.35
FLOPS_PER_CYCLE_PER_CORE = 2048  # HiFi2

ITERS = int(os.environ.get("RING_MLA_SWEEP_ITERS", "10"))
COMPUTE_ONLY = os.environ.get("SPARSE_SDPA_COMPUTE_ONLY", "0") == "1"

# GLM_SP_BATCH=1 keeps the 8x4 mesh but gives each TP column its own request: rows are split over
# the 8 rows only, so each device takes 4x the tokens and four requests share the machine.
# This op is device-local with replicated inputs, so only the row count changes.
SP_BATCH = os.environ.get("GLM_SP_BATCH", "0") == "1"
SHARD_N = SP if SP_BATCH else SP * TP

RESULTS_DIR = Path(os.environ.get("GLM_SPARSE_SDPA_SWEEP_OUT", "generated/glm_sparse_sdpa_sweep"))

CHUNKS = [1024, 2048, 3072, 4096, 5120]
if os.environ.get("GLM_CHUNKS"):
    CHUNKS = [int(c) for c in os.environ["GLM_CHUNKS"].split(",") if c.strip()]
K_CHUNKS = [64, 128, 256]
PROD_K_CHUNK = 128
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


def _make_indices(rows, chunk, prefix):
    """Top-k-like indices for the last stripe's rows: nv = min(pos+1, TOPK) distinct causal keys."""
    idx = torch.full((1, 1, rows, TOPK), SENTINEL, dtype=torch.int64)
    nvs = []
    for r in range(rows):
        pos = prefix + chunk - rows + r
        nv = min(pos + 1, TOPK)
        idx[0, 0, r, :nv] = torch.randperm(pos + 1)[:nv]
        nvs.append(nv)
    return idx, nvs


@pytest.mark.parametrize("mesh_device", [pytest.param((SP, TP), id="8x4")], indirect=True)
@pytest.mark.parametrize("k_chunk", K_CHUNKS, ids=[f"kc{k}" for k in K_CHUNKS])
@pytest.mark.parametrize("chunk", CHUNKS, ids=[f"chunk{c}" for c in CHUNKS])
@pytest.mark.timeout(900)
def test_glm_sparse_sdpa_chunk_sweep(mesh_device, chunk, k_chunk):
    _run_sparse_sdpa(mesh_device, chunk, round(TARGET_PREFIX / chunk) * chunk, k_chunk)


@pytest.mark.parametrize("mesh_device", [pytest.param((SP, TP), id="8x4")], indirect=True)
@pytest.mark.parametrize("chunk, prefix", _prefix_sweep_params())
@pytest.mark.timeout(900)
def test_glm_sparse_sdpa_prefix_sweep(mesh_device, chunk, prefix):
    _run_sparse_sdpa(mesh_device, chunk, prefix, PROD_K_CHUNK)


def _run_sparse_sdpa(mesh_device, chunk, prefix, k_chunk):
    torch.manual_seed(1234)
    mesh_device.enable_program_cache()

    rows = chunk // SHARD_N
    capacity = prefix + chunk  # whole chunks, so the block-cyclic slab tiling holds
    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y

    tt_q = _replicated(mesh_device, torch.randn(1, NUM_HEADS, rows, KVPE_DIM), ttnn.bfloat16)
    tt_kv = _replicated(mesh_device, torch.randn(1, 1, capacity, KVPE_DIM), ttnn.bfloat16)
    indices, nvs = _make_indices(rows, chunk, prefix)
    tt_idx = _replicated(mesh_device, indices, ttnn.uint32)

    def run_once():
        return ttnn.transformer.sparse_sdpa(
            tt_q,
            tt_kv,
            tt_idx,
            v_dim=V_DIM,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=SCALE,
            k_chunk_size=k_chunk,
            block_cyclic_sp_axis=0,
            block_cyclic_chunk_local=chunk // SP,
            block_cyclic_cache_tp_sharded=True,
            cache_batch_idx=None,
        )

    try:
        warm = run_once()
    except RuntimeError as e:
        pytest.skip(f"rejected by sparse_sdpa: {str(e).splitlines()[0][:300]}")
    ttnn.synchronize_device(mesh_device)
    assert list(warm.shape) == [1, NUM_HEADS, rows, V_DIM], f"unexpected output shape {warm.shape}"
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

    median_ns = statistics.median(durations_ns)
    tokens_per_core = math.ceil(rows / num_cores)
    # Critical core: its tokens' key chunks (the last rows carry the most keys).
    crit_chunks = sum(math.ceil(nv / k_chunk) for nv in nvs[-tokens_per_core:])
    flops = 2 * NUM_HEADS * (KVPE_DIM + V_DIM) * sum(nvs)
    util = flops / (num_cores * FLOPS_PER_CYCLE_PER_CORE * BH_CLOCK_GHZ * median_ns) * 100
    row = {
        "chunk": chunk,
        "rows": rows,
        "prefix": prefix,
        "k_chunk": k_chunk,
        "core_grid": f"{grid.x}x{grid.y}",
        "cores": num_cores,
        "active_cores": min(num_cores, rows),
        "tokens_per_core": tokens_per_core,
        "core_util": round(rows / (num_cores * tokens_per_core), 4),
        "mean_nv": round(sum(nvs) / len(nvs), 1),
        "compute_only": COMPUTE_ONLY,
        "median_us": round(median_ns / 1e3, 2),
        "min_us": round(min(durations_ns) / 1e3, 2),
        "max_us": round(max(durations_ns) / 1e3, 2),
        "math_util_pct": round(util, 2),
        # Critical core's cost per (token, 128-key chunk): flat when the op scales cleanly.
        "ns_per_token_chunk": round(median_ns / crit_chunks, 1),
        "kv_gather_gbps": round(sum(nvs) * KVPE_DIM * 2 / median_ns, 2),
    }
    logger.info(f"glm sparse_sdpa sweep: {row}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rt_results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
