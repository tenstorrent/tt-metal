# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced GLM-5.2 sparse-KV prefix gather (``high_bw_all_gather``) chunk / KV-prefix sweep on 8x4.

Mirrors ``ttMLA._gather_kvpe_prefix``: one full-mesh snake gather (``cluster_axis=None``) of the TP-deduped
BF16_RM KVPE cache ``[1, 1, T/32, 576]`` per device into the replicated ``[1, 1, T, 576]`` scratch, with the
extent rounded to whole block-cyclic slabs (slab = chunk). The cache is sized to the gathered extent plus one chunk (the model's is 1M tokens; the gather is bounded).
Standalone on the full grid; production overlaps it with top-k on a 40-core sub-device.

A pure CCL has no compute-only mode, so each run reports achieved bandwidth against the fabric roofline
used by test_sparse_mla_ccl_perf.py (200 Gb/s per link per direction on Galaxy, 2 links, both directions).

    pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_glm_kvpe_gather_chunk_sweep.py -k "chunk_sweep and chunk5120"
"""

import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, MlaKvCacheGeometry, init_mla_kv_cache
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

SP, TP = 8, 4
RING = SP * TP
KV_LORA_RANK, QK_ROPE_HEAD_DIM = 512, 64
# GLM_KV_FORMAT selects the KVPE encoding. sparse_sdpa_format accepts only these two, so they are
# the whole choice for the sparse path: bf16_rm is 576 x 2 = 1152 B/row; scaled_fp8 packs fp8 latent,
# fp32 scales and bf16 rope into 656 B/row, 43% fewer bytes over the same fabric.
KV_GEOMETRY = MlaKvCacheGeometry(latent_dim=KV_LORA_RANK, rope_dim=QK_ROPE_HEAD_DIM)
KV_FORMAT = MlaKvCacheFormat(os.environ.get("GLM_KV_FORMAT", "bf16_rm"))
ROW_BYTES = KV_GEOMETRY.packed_row_bytes if KV_FORMAT is MlaKvCacheFormat.SCALED_FP8 else KV_GEOMETRY.logical_width * 2
MAX_CONTEXT = GLM52Config.MAX_POSITION_EMBEDDINGS
TARGET_PREFIX = 51200
TRACE_REGION_SIZE = 16 * 1024 * 1024
NUM_LINKS = int(os.environ.get("GLM_GATHER_NUM_LINKS", "2"))

# GLM_SP_BATCH=1 keeps the 8x4 mesh and fabric but runs each TP column as an independent request:
# Q/W replicated across columns, KV TP-replicated and sharded only on rows, gather on the SP axis.
# Four requests then share the machine instead of one spanning it.
SP_BATCH = os.environ.get("GLM_SP_BATCH", "0") == "1"
SHARD_N = SP if SP_BATCH else SP * TP  # devices the sequence is split across
CLUSTER_AXIS = 0 if SP_BATCH else None  # SP-axis gather vs full-mesh gather
KV_TP_AXIS = None if SP_BATCH else 1  # None = TP-replicated, 1 = also striped across TP

LINK_GBPS_PER_DIRECTION = float(os.environ.get("MLA_CCL_LINK_GBPS_PER_DIRECTION", "200"))

ITERS = int(os.environ.get("RING_MLA_SWEEP_ITERS", "10"))
RESULTS_DIR = Path(os.environ.get("GLM_KVPE_GATHER_SWEEP_OUT", "generated/glm_kvpe_gather_sweep"))

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
@pytest.mark.parametrize("chunk", CHUNKS, ids=[f"chunk{c}" for c in CHUNKS])
@pytest.mark.timeout(900)
def test_glm_kvpe_gather_chunk_sweep(mesh_device, device_params, chunk):
    _run_gather(mesh_device, chunk, round(TARGET_PREFIX / chunk) * chunk)


@MESH_8X4
@pytest.mark.parametrize("chunk, prefix", _prefix_sweep_params())
@pytest.mark.timeout(900)
def test_glm_kvpe_gather_prefix_sweep(mesh_device, device_params, chunk, prefix):
    _run_gather(mesh_device, chunk, prefix)


def _run_gather(mesh_device, chunk, prefix):
    mesh_device.enable_program_cache()
    populated = prefix + chunk
    slab_global = chunk  # block_cyclic_chunk_local (chunk / sp) * sp
    extent = -(-populated // slab_global) * slab_global
    # The gather is bounded by gathered_dim_size, so the cache only needs to exceed the populated prefix; one
    # spare chunk keeps it larger than the gathered extent, as in the model (whose cache is 1M tokens).
    capacity = min(MAX_CONTEXT // chunk * chunk, extent + chunk)

    cache = init_mla_kv_cache(
        cache_format=KV_FORMAT,
        hf_config=SimpleNamespace(kv_lora_rank=KV_LORA_RANK, qk_rope_head_dim=QK_ROPE_HEAD_DIM),
        mesh_device=mesh_device,
        seq_len=capacity,
        mesh_shape=[SP, TP],
        sp_axis=0,
        num_kvpe_cache_layers=1,
        num_users=1,
        tp_axis=KV_TP_AXIS,
    )
    storage = cache.storage
    # Take the width from the cache itself: scaled_fp8 stores a packed 656-byte row, not the 576
    # logical elements, so a hardcoded width is wrong for every format but bf16_rm.
    # ttnn.zeros goes through full_impl, which rejects FP8_E4M3 outright ("output-only dtype, host-side
    # construction via fill is not supported"). ttnn.empty just allocates, and is what the model's own
    # get_mla_high_bw_all_gather_buffer uses. The gather overwrites every row it reads anyway.
    out_buf = ttnn.empty(
        [1, 1, capacity, int(storage.shape[-1])],
        dtype=storage.dtype,
        layout=storage.layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def run_once():
        return ttnn.experimental.high_bw_all_gather(
            storage,
            dim=2,
            output_tensor=out_buf,
            num_links=NUM_LINKS,
            cluster_axis=CLUSTER_AXIS,
            input_batch_index=0,
            gathered_dim_size=extent,
        )

    try:
        run_once()
    except RuntimeError as e:
        pytest.skip(f"rejected by high_bw_all_gather: {str(e).splitlines()[0][:300]}")
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    run_once()
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
        ttnn.deallocate(out_buf)
        ttnn.deallocate(storage)

    median_ns = statistics.median(durations_ns)
    # SHARD_N, not RING: in SP-batch the sequence is split over the 8 rows only, so each device
    # holds a 4x larger slice and receives 7/8 of the extent instead of 31/32.
    local_bytes = extent // SHARD_N * ROW_BYTES  # each device's slice of the gathered extent
    received_bytes = local_bytes * (SHARD_N - 1)  # what every device must receive
    roofline_gbs = LINK_GBPS_PER_DIRECTION * NUM_LINKS * 2 / 8  # 2 links x 2 ring directions
    ideal_ns = received_bytes / roofline_gbs
    row = {
        "chunk": chunk,
        "prefix": prefix,
        "kv_format": KV_FORMAT.value,
        "row_bytes": ROW_BYTES,
        "extent": extent,
        "capacity": capacity,
        "local_mb": round(local_bytes / 1e6, 3),
        "received_mb": round(received_bytes / 1e6, 2),
        "median_us": round(median_ns / 1e3, 2),
        "min_us": round(min(durations_ns) / 1e3, 2),
        "max_us": round(max(durations_ns) / 1e3, 2),
        "achieved_gbs": round(received_bytes / median_ns, 2),
        "roofline_gbs": roofline_gbs,
        "ideal_us": round(ideal_ns / 1e3, 2),
        "bw_util_pct": round(ideal_ns / median_ns * 100, 2),
        "us_per_1k_tokens": round(median_ns / 1e3 * 1024 / chunk, 2),
    }
    logger.info(f"glm kvpe gather sweep: {row}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rt_results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
