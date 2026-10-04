# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced Kimi-K2.7 ring_mla chunk-size sweep on the 8x4 torus.

Mirrors the model's dense chunked-prefill call (mla.py ``_chunked_attn``): per-device Q
``[1, 16, chunk/8, 576]`` bf16 against a bf8 KVPE cache (V = first 512 columns) holding a ~50k
prefix plus the current chunk, fused KV all-gather on SP axis 0, 11x10 SDPA grid, HiFi2.

Flow: one eager warm-up (compile), capture one ``ring_mla`` into a trace, replay it
``RING_MLA_SWEEP_ITERS`` times. Without the device profiler each replay is timed in its own
realtime-profiler window (replays share a runtime id); under tracy the replays sit between
``start``/``stop`` signposts and ``summarize_ring_mla_tracy.py`` reads kernel time + core count.

    # realtime profiler (fast sweep)
    pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_mla_chunk_sweep.py -k "chunk5120-q32-k640"
    # tracy (kernel duration, core count, PM IDEAL)
    python -m tracy -r -p -v -m pytest ...test_ring_mla_chunk_sweep.py -k "chunk5120-q32-k640"

RING_SDPA_COMPUTE_ONLY=1 builds the op with every data movement stubbed out (compute-bound check).
"""

import json
import os
import statistics
from pathlib import Path

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import create_global_semaphores, per_axis_topology
from tests.nightly.sdpa_perf_utils import compute_math_utilization
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

SP, TP = 8, 4
NUM_HEADS = 64  # Kimi-K2.7; 16 per TP device
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
KVPE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
TARGET_PREFIX = 51200  # the 50k+5k production point; rounded to a whole number of chunks
TRACE_REGION_SIZE = 16 * 1024 * 1024

# RING_MLA_FIDELITY selects the matmul fidelity. compute_math_utilization assumes HiFi2
# (2048 FLOP/cycle/core = the 4096 base rate / 2 phases), so its result must be rescaled when the
# phase count changes: true util = reported * phases / 2. LoFi runs one phase, so the roofline
# doubles and the reported figure halves.
_FIDELITY = {
    "LoFi": (ttnn.MathFidelity.LoFi, 1),
    "HiFi2": (ttnn.MathFidelity.HiFi2, 2),
    "HiFi4": (ttnn.MathFidelity.HiFi4, 4),
}
FIDELITY_NAME = os.environ.get("RING_MLA_FIDELITY", "HiFi2")
MATH_FIDELITY, _PHASES = _FIDELITY[FIDELITY_NAME]
UTIL_SCALE = _PHASES / 2

ITERS = int(os.environ.get("RING_MLA_SWEEP_ITERS", "10"))
COMPUTE_ONLY = os.environ.get("RING_SDPA_COMPUTE_ONLY", "0") == "1"
RESULTS_DIR = Path(os.environ.get("RING_MLA_SWEEP_OUT", "generated/ring_mla_chunk_sweep"))

CHUNKS = [1024, 2048, 3072, 4096, 5120]
# RING_MLA_CHUNKS overrides the chunk list (comma-separated global chunk sizes).
if os.environ.get("RING_MLA_CHUNKS"):
    CHUNKS = [int(c) for c in os.environ["RING_MLA_CHUNKS"].split(",") if c.strip()]
Q_CHUNKS = [32, 64, 128]
K_CHUNKS = [128, 256, 320, 512, 640, 1024, 1280]


def _sweep_params():
    params = []
    for chunk in CHUNKS:
        chunk_local = chunk // SP
        for q in Q_CHUNKS:
            if chunk_local % q:
                continue
            for k in K_CHUNKS:
                params.append(pytest.param(chunk, q, k, id=f"chunk{chunk}-q{q}-k{k}"))
    return params


def _prefix_for(chunk):
    return max(1, round(TARGET_PREFIX / chunk)) * chunk


def _occupancy(chunk, q_chunk, num_cores):
    units = (NUM_HEADS // TP) * (chunk // SP) // q_chunk
    per_core = -(-units // num_cores)
    return units, units / (num_cores * per_core)


MESH_8X4 = pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (SP, TP),
            torus_xy_device_params(trace_region_size=TRACE_REGION_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(SP, TP), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)

# KV-prefix sweep per chunk size at its best q/k (from the chunk sweep). Target prefixes are rounded
# to whole chunks and deduplicated.
PREFIX_SWEEP_BEST_QK = {1024: (32, 128), 2048: (32, 512), 3072: (32, 640), 4096: (32, 512), 5120: (32, 640)}
# RING_MLA_BEST_QK adds or replaces entries as "chunk:q:k" pairs, e.g. "1792:32:512".
for _entry in os.environ.get("RING_MLA_BEST_QK", "").split(","):
    if _entry.strip():
        _c, _q, _k = (int(v) for v in _entry.split(":"))
        PREFIX_SWEEP_BEST_QK[_c] = (_q, _k)
PREFIX_TARGETS = [0, 2048, 4096, 8192, 16384, 32768, 51200, 65536, 102400, 131072, 196608, 262144]
# RING_MLA_PREFIX_TARGETS overrides the list (comma-separated tokens) for long-ISL runs, e.g. out to 1M.
if os.environ.get("RING_MLA_PREFIX_TARGETS"):
    PREFIX_TARGETS = [int(t) for t in os.environ["RING_MLA_PREFIX_TARGETS"].split(",") if t.strip()]


def _prefix_sweep_params():
    params = []
    for chunk, (q, k) in PREFIX_SWEEP_BEST_QK.items():
        for prefix in sorted({round(t / chunk) * chunk for t in PREFIX_TARGETS}):
            params.append(pytest.param(chunk, q, k, prefix, id=f"chunk{chunk}-prefix{prefix}"))
    return params


@MESH_8X4
@pytest.mark.parametrize("chunk, q_chunk, k_chunk", _sweep_params())
@pytest.mark.timeout(900)
def test_ring_mla_chunk_sweep(mesh_device, device_params, chunk, q_chunk, k_chunk):
    _run_ring_mla(mesh_device, chunk, q_chunk, k_chunk, _prefix_for(chunk))


@MESH_8X4
@pytest.mark.parametrize("chunk, q_chunk, k_chunk, prefix", _prefix_sweep_params())
@pytest.mark.timeout(900)
def test_ring_mla_prefix_sweep(mesh_device, device_params, chunk, q_chunk, k_chunk, prefix):
    _run_ring_mla(mesh_device, chunk, q_chunk, k_chunk, prefix)


def _run_ring_mla(mesh_device, chunk, q_chunk, k_chunk, prefix):
    torch.manual_seed(1234)
    mesh_device.enable_program_cache()
    profiler_on = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"

    chunk_local = chunk // SP
    heads_local = NUM_HEADS // TP
    assert prefix % chunk == 0, f"prefix {prefix} must be a whole number of {chunk}-token chunks"
    logical_n = prefix + chunk
    # ring_mla needs Q.seq < K.seq (chunked prefill); the model's cache is always sized past the first
    # chunk, so give the empty-prefix case one spare chunk of capacity (unread: logical_n bounds it).
    capacity = logical_n + (chunk if prefix == 0 else 0)
    assert chunk_local % ttnn.TILE_SIZE == 0 and (capacity // SP) % ttnn.TILE_SIZE == 0

    grid = mesh_device.compute_with_storage_grid_size()
    sdpa_grid = ttnn.CoreCoord(grid.x - 1, grid.y)  # last column reserved for the fused CCL
    num_cores = sdpa_grid.x * sdpa_grid.y
    sp_topology, _ = per_axis_topology()

    # Q: seq over SP (axis 0), heads over TP (axis 1).
    tt_q = ttnn.from_torch(
        torch.randn(1, NUM_HEADS, chunk, KVPE_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(SP, TP), dims=[2, 1]),
    )
    # KVPE cache: one slot, seq over SP, replicated over TP (dense, not TP-deduped).
    tt_kv = ttnn.from_torch(
        torch.randn(1, 1, capacity, KVPE_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(SP, TP), dims=[2, None]),
    )
    tt_kv_buf = ttnn.from_torch(
        torch.zeros(1, 1, capacity, KVPE_DIM),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(SP, TP), dims=[None, None]),
    )
    semaphores = create_global_semaphores(
        mesh_device,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}),
        0,
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_grid,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=MATH_FIDELITY,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def run_once():
        out, _ = ttnn.transformer.ring_mla(
            tt_q,
            tt_kv,
            persistent_output_buffer_kv=tt_kv_buf,
            head_dim_v=KV_LORA_RANK,
            logical_n=logical_n,
            program_config=program_config,
            scale=(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5,
            compute_kernel_config=compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=semaphores,
            num_links=2,
            cluster_axis=0,
            mesh_device=mesh_device,
            topology=sp_topology,
            ccl_core_grid_offset=(grid.x - 1, 0),
            use_column_major_ccl=True,
            is_balanced=False,
            kv_cache_batch_idx=0,
            kv_actual_isl=prefix,
        )
        return out

    # Warm-up: compile + program cache.
    try:
        warm = run_once()
    except RuntimeError as e:
        pytest.skip(f"config rejected by ring_mla: {str(e).splitlines()[0][:300]}")
    ttnn.synchronize_device(mesh_device)
    assert list(warm.shape) == [1, heads_local, chunk_local, KV_LORA_RANK], f"unexpected output shape {warm.shape}"
    if not COMPUTE_ONLY:
        host = ttnn.to_torch(ttnn.get_device_tensors(warm)[0])
        assert torch.isfinite(host).all(), "warm-up output has non-finite values"
    ttnn.deallocate(warm)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = run_once()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    def replay_once():
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

    durations_ns = []
    try:
        if profiler_on:
            replay_once()  # warm replay, outside the signposts
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            for _ in range(ITERS):
                replay_once()
            ttnn.synchronize_device(mesh_device)
            signpost("stop")
        else:
            # Separate windows: replays share a runtime id and would merge into one record.
            profile_realtime_program_merged(mesh_device, replay_once, record_timeout_seconds=30.0)
            for _ in range(ITERS):
                _, programs = profile_realtime_program_merged(mesh_device, replay_once, record_timeout_seconds=30.0)
                durations_ns.append(max(p["duration_ns"] for p in programs.values()))
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(trace_out)

    units, occupancy = _occupancy(chunk, q_chunk, num_cores)
    row = {
        "chunk": chunk,
        "chunk_local": chunk_local,
        "prefix": prefix,
        "q_chunk": q_chunk,
        "k_chunk": k_chunk,
        "compute_only": COMPUTE_ONLY,
        "fidelity": FIDELITY_NAME,
        "sdpa_grid": f"{sdpa_grid.x}x{sdpa_grid.y}",
        "sdpa_cores": num_cores,
        "work_units": units,
        # Derived from the op's split (min(cores, heads x Q chunks)), not measured.
        "active_cores": min(num_cores, units),
        "occupancy": round(occupancy, 4),
    }
    if durations_ns:
        median_ns = statistics.median(durations_ns)
        # Rectangle (prefix) + causal triangle folded into an effective KV length, as the nightly check.
        util = compute_math_utilization(
            chunk_local, prefix + chunk // 2, KVPE_DIM, KV_LORA_RANK, heads_local, median_ns, num_cores
        )
        util *= UTIL_SCALE  # rescale off the helper's HiFi2 basis
        row.update(
            median_us=round(median_ns / 1e3, 2),
            min_us=round(min(durations_ns) / 1e3, 2),
            max_us=round(max(durations_ns) / 1e3, 2),
            math_util_pct=round(util, 2),
        )
    logger.info(f"ring_mla chunk sweep: {row}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / ("tracy_runs.jsonl" if profiler_on else "rt_results.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
