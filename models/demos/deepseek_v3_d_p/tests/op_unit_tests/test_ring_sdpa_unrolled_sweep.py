# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced Kimi-K2.7 ring joint SDPA sweep with V unrolled out of the MLA latent space.

The absorbed form (``test_ring_mla_chunk_sweep.py``) attends directly on the compressed cache:
one shared latent row of 576 = 512 + 64 rope, V = its first 512 columns, and the W_UV
up-projection runs *after* attention on the Q rows only. This harness measures the other
arrangement -- the up-projection done *before* attention over the whole KV cache -- so SDPA
sees ordinary per-head K/V: d_q = 128 nope + 64 rope = 192, d_v = 128.

That trades attention FLOPs for cache width:
  absorbed  d_q+d_v = 1088, KV row 1 x 576  per device
  unrolled  d_q+d_v =  320, KV row 16 x 320 = 5120 per device
so 3.40x fewer attention FLOPs against an 8.9x wider KV row. This harness times the SDPA only;
the decompression matmul that produces K/V is a separate O(kv_len) cost and is NOT included.

    pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_sdpa_unrolled_sweep.py \
        -k "chunk_sweep and chunk5120"
    RING_SDPA_UNROLLED_PREFIX_TARGETS=... pytest ...::test_ring_sdpa_unrolled_prefix_sweep

RING_SDPA_COMPUTE_ONLY=1 stubs every transfer (shared with the absorbed harness).
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
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
KV_LORA_RANK = 512
V_HEAD_DIM = 128  # W_UV output: materialised per-head V
LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576, the QK product stays in latent space
TARGET_PREFIX = 51200
TRACE_REGION_SIZE = 16 * 1024 * 1024

ITERS = int(os.environ.get("RING_MLA_SWEEP_ITERS", "10"))
COMPUTE_ONLY = os.environ.get("RING_SDPA_COMPUTE_ONLY", "0") == "1"
RESULTS_DIR = Path(os.environ.get("RING_MLA_SWEEP_OUT", "generated/ring_sdpa_unrolled"))

CHUNKS = [int(c) for c in os.environ.get("RING_SDPA_UNROLLED_CHUNKS", "5120").split(",") if c.strip()]
Q_CHUNKS = [32, 64, 128]
K_CHUNKS = [128, 256, 320, 512, 640]

PREFIX_TARGETS = [0, 2048, 4096, 8192, 16384, 24576, 32768, 40960, 51200, 65536, 81920, 102400, 131072]
if os.environ.get("RING_SDPA_UNROLLED_PREFIX_TARGETS"):
    PREFIX_TARGETS = [int(t) for t in os.environ["RING_SDPA_UNROLLED_PREFIX_TARGETS"].split(",") if t.strip()]
BEST_QK = {}
for _e in os.environ.get("RING_SDPA_UNROLLED_BEST_QK", "").split(","):
    if _e.strip():
        _c, _q, _k = (int(v) for v in _e.split(":"))
        BEST_QK[_c] = (_q, _k)


def _slab_major(logical, chunk, capacity):
    """Permute a logical [b,h,capacity,d] cache into the op's slab-major per-device layout.

    Device d owns strip d of EVERY chunk-group, not a contiguous span: global position p lands on
    device (p % chunk) // chunk_local at that device's row (p // chunk) * chunk_local + p % chunk_local.
    A contiguous dim-2 shard of the returned tensor therefore hands each device the right rows.
    """
    chunk_local = chunk // SP
    rows_per_dev = capacity // SP
    perm = torch.empty(capacity, dtype=torch.long)
    for d in range(SP):
        for r in range(rows_per_dev):
            g, cell = divmod(r, chunk_local)
            perm[d * rows_per_dev + r] = g * chunk + d * chunk_local + cell
    assert sorted(perm.tolist()) == list(range(capacity)), "slab-major permutation is not a bijection"
    return logical[:, :, perm, :]


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


def _prefix_sweep_params():
    params = []
    for chunk, (q, k) in BEST_QK.items():
        for prefix in sorted({max(1, round(t / chunk)) * chunk if t else 0 for t in PREFIX_TARGETS}):
            params.append(pytest.param(chunk, q, k, prefix, id=f"chunk{chunk}-prefix{prefix}"))
    return params


def _prefix_for(chunk):
    return max(1, round(TARGET_PREFIX / chunk)) * chunk


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


@MESH_8X4
@pytest.mark.parametrize("chunk, q_chunk, k_chunk", _sweep_params())
@pytest.mark.timeout(900)
def test_ring_sdpa_unrolled_chunk_sweep(mesh_device, device_params, chunk, q_chunk, k_chunk):
    _run(mesh_device, chunk, q_chunk, k_chunk, _prefix_for(chunk))


@MESH_8X4
@pytest.mark.parametrize("chunk, q_chunk, k_chunk, prefix", _prefix_sweep_params())
@pytest.mark.timeout(900)
def test_ring_sdpa_unrolled_prefix_sweep(mesh_device, device_params, chunk, q_chunk, k_chunk, prefix):
    _run(mesh_device, chunk, q_chunk, k_chunk, prefix)


def _run(mesh_device, chunk, q_chunk, k_chunk, prefix):
    torch.manual_seed(1234)
    mesh_device.enable_program_cache()

    chunk_local = chunk // SP
    heads_local = NUM_HEADS // TP
    assert prefix % chunk == 0 or prefix == 0
    logical_n = prefix + chunk
    capacity = logical_n + (chunk if prefix == 0 else 0)
    assert chunk_local % ttnn.TILE_SIZE == 0 and (capacity // SP) % ttnn.TILE_SIZE == 0

    grid = mesh_device.compute_with_storage_grid_size()
    sdpa_grid = ttnn.CoreCoord(grid.x - 1, grid.y)  # last column reserved for the fused CCL
    num_cores = sdpa_grid.x * sdpa_grid.y
    sp_topology, _ = per_axis_topology()

    def up(t, dtype, dims):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(SP, TP), dims=dims),
        )

    # Q: seq over SP, heads over TP.  K/V caches: seq over SP, heads over TP.
    # Q stays in latent space (576) and K stays the ONE shared latent row (nhk=1), exactly as in
    # the absorbed form. Only V is materialised per head at 128 -- W_UV moved ahead of the SDPA.
    tt_q = up(torch.randn(1, NUM_HEADS, chunk, LATENT_DIM), ttnn.bfloat16, [2, 1])
    tt_k = up(_slab_major(torch.randn(1, 1, capacity, LATENT_DIM), chunk, capacity), ttnn.bfloat8_b, [2, None])
    tt_v = up(_slab_major(torch.randn(1, NUM_HEADS, capacity, V_HEAD_DIM), chunk, capacity), ttnn.bfloat8_b, [2, 1])
    # Gather buffers hold the whole sequence, so seq is replicated and only heads shard.
    buf_k = up(torch.zeros(1, 1, capacity, LATENT_DIM), ttnn.bfloat8_b, [None, None])
    buf_v = up(torch.zeros(1, NUM_HEADS, capacity, V_HEAD_DIM), ttnn.bfloat8_b, [None, 1])

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
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def run_once():
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            None,
            None,
            None,
            persistent_output_buffer_k=buf_k,
            persistent_output_buffer_v=buf_v,
            joint_strategy="rear",
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
            is_causal=True,
            is_balanced=False,
            kv_cache_batch_idx=0,
            kv_actual_isl=prefix,
        )
        return out

    try:
        warm = run_once()
    except RuntimeError as e:
        pytest.skip(f"config rejected: {str(e).splitlines()[0][:300]}")
    ttnn.synchronize_device(mesh_device)
    assert list(warm.shape) == [1, heads_local, chunk_local, V_HEAD_DIM], f"unexpected output shape {warm.shape}"
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

    profiler_on = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
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
    if not durations_ns:
        pytest.skip("device profiler mode collects timings through tracy, not this record path")

    median_ns = statistics.median(durations_ns)
    util = compute_math_utilization(
        chunk_local, prefix + chunk // 2, LATENT_DIM, V_HEAD_DIM, heads_local, median_ns, num_cores
    )
    units = heads_local * (chunk_local // q_chunk)
    rec = {
        "form": "v_unrolled",
        "chunk": chunk,
        "chunk_local": chunk_local,
        "prefix": prefix,
        "q_chunk": q_chunk,
        "k_chunk": k_chunk,
        "compute_only": COMPUTE_ONLY,
        "d_q": LATENT_DIM,
        "d_v": V_HEAD_DIM,
        "sdpa_cores": num_cores,
        "work_units": units,
        "active_cores": min(num_cores, units),
        "median_us": round(median_ns / 1000, 2),
        "min_us": round(min(durations_ns) / 1000, 2),
        "max_us": round(max(durations_ns) / 1000, 2),
        "math_util_pct": round(util, 2),
    }
    logger.info(f"ring sdpa unrolled: {rec}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "rt_results.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")

    for t in (tt_q, tt_k, tt_v, buf_k, buf_v):
        ttnn.deallocate(t)
