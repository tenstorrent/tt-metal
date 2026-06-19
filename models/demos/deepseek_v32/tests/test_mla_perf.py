# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tracy perf harness for the DeepSeek V3.2 MLA (DSA) chunked-prefill layer.

Scenario (defaults): process one **5k-token chunk** with **50k tokens already cached**, on a
**SP=4 × TP=2** LoudBox mesh (mesh shape (4, 2)) — the §7 worked example of the MLA layer report.
No reference values: this just runs the real device forward and reports per-op device-kernel time.
Multi-chip rows are device-collapsed (compute=max, collectives=avg across chips) via merge_device_rows
so the reported time is per-step critical path, not the ~8× over-count of summing parallel device rows.

Two-test pattern (mirrors tests/nightly/blackhole/sdpa):
  * test_mla_chunked_perf_impl  — the work to profile. Builds the v32 ttMLA, populates the index/KV
    caches directly to stand in for the cached prefix (no warm-up forwards), then wraps a SINGLE
    `chunk`-token forward in signpost("start"/"stop"). Run this under tracy.
  * test_mla_chunked_perf       — the driver. Spawns the impl under tracy via run_device_profiler,
    reads the device ops log for the signposted region, prints a per-op table, and writes a CSV.

Run (LoudBox / 8-chip Blackhole):
    pytest models/demos/deepseek_v32/tests/test_mla_perf.py::test_mla_chunked_perf -s

Knobs (env): DS_PERF_CACHE (default 51200), DS_PERF_CHUNK (default 5120), DS_PERF_CSV.

NOTE: caches are POPULATED directly (random index keys; KVPE left at init) rather than warmed with
real chunks — only op shapes/timing matter here, not values. The indexer rope now scales from the HF
config (mla.py), so `config.max_seq_len = total` is enough for a 50k+ context (no manual rope bump).
"""

import os

import pandas as pd
import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from models.demos.deepseek_v32.tests.test_mla import build_cpu_reference, make_hidden
from models.demos.deepseek_v32.tt.mla import ttMLA

CACHE_TOKENS = int(os.environ.get("DS_PERF_CACHE", 51200))  # 50 * 1024 already cached
CHUNK_TOKENS = int(os.environ.get("DS_PERF_CHUNK", 5120))  # 5 * 1024 processed this step
SUBDIR = "deepseek_v32_mla_perf"
CSV_OUT = os.environ.get("DS_PERF_CSV", "deepseek_v32_mla_perf.csv")


# ============================================================================
# Inner: the work to profile (run under tracy by the driver below)
# ============================================================================
@pytest.mark.parametrize("mesh_device", [(4, 2)], ids=["sp4xtp2"], indirect=True)  # SP=4, TP=2
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        }
    ],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.timeout(0)
def test_mla_chunked_perf_impl(mesh_device, device_params, variant, config_only):
    from tracy import signpost

    cache, chunk = CACHE_TOKENS, CHUNK_TOKENS
    total = cache + chunk
    sp_axis, tp_axis = 0, 1
    sp, tp = mesh_device.shape
    assert cache % chunk == 0, f"cache {cache} must be a whole number of {chunk}-token chunks"
    assert total % sp == 0 and (total // sp) % 32 == 0, f"total {total} must be tile-aligned per SP={sp} chip"

    config = config_only
    config.max_seq_len = total  # rope-table / buffer length (same hack as the correctness tests)
    # Weights only — projections are sequence-length independent, so a small build is fine.
    _, _, weights, _ = build_cpu_reference(2048, seed=42)

    # Indexer rope now scales from config.max_seq_len (set above) — no manual bump needed.
    mla = ttMLA(
        config,
        dict(weights),
        mesh_device,
        layer_idx=0,
        seq_len=total,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        layer_num=1,
    )

    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors_indexed(total, chunk)
    kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=total,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    # Represent `cache` already-processed tokens by POPULATING the caches directly (no warm-up
    # forwards). The indexer K-cache (replicated, natural order, grown by concat) is filled with random
    # keys so the measured chunk scores against a full `cache`-length prefix. The KVPE block-cyclic
    # cache is left at its init: the measured chunk writes its own slab and the gather reads the full
    # prefix, and cache values don't change op shapes/timing.
    mla._index_kbuf = ttnn.from_torch(
        torch.randn(1, 1, cache, mla.index_args.index_head_dim, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    hidden = make_hidden(chunk, config.hidden_size, seed=42)  # only the measured chunk
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2
    tt_x = ttnn.from_torch(
        hidden[:, :chunk].unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    logger.info(f"profiling one {chunk}-token chunk @ {cache}-token cache (end_pos={total}) on SP={sp}×TP={tp} …")
    signpost("start")
    out = mla.forward(tt_x, rope, kvpe_cache, actual_start=cache)
    ttnn.deallocate(out)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")


# ============================================================================
# Outer: drive the impl under tracy, post-process, print + write CSV
# ============================================================================
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test — run locally with tracy")
@pytest.mark.timeout(0)
def test_mla_chunked_perf():
    from tracy.process_model_log import run_device_profiler

    # merge_device_rows: the deepseek_v3_d_p / tt_transformers convention for collapsing the device
    # dimension of a multi-chip Tracy ops log (see models/demos/deepseek_v3_d_p/utils/perf_utils.py).
    from models.tt_transformers.tests.test_utils import merge_device_rows
    from tests.nightly.sdpa_perf_utils import post_process_ops_log

    command = "pytest models/demos/deepseek_v32/tests/test_mla_perf.py::test_mla_chunked_perf_impl"
    run_device_profiler(command, SUBDIR, device_analysis_types=["device_kernel_duration"])

    dur_col = "DEVICE KERNEL DURATION [ns]"
    # Rows between signpost("start") and signpost("stop") = the measured chunk's device ops, with ONE
    # ROW PER (op call × mesh chip). On this SP=4×TP=2 mesh every op runs on all 8 chips IN PARALLEL,
    # so the raw rows must NOT be summed — that over-counts wall-clock by ~num_devices (≈8×). Collapse
    # the device dimension to one row per logical op call with the standard merge_device_rows rule:
    #   * compute ops -> MAX duration across chips (the slowest chip gates the step = critical path)
    #   * collectives -> AVG duration across chips (all chips run the same collective together)
    df = post_process_ops_log(SUBDIR, has_signposts=True)
    df[dur_col] = pd.to_numeric(df[dur_col], errors="coerce")
    df = merge_device_rows(df)  # filters to tt_dnn_device rows internally
    assert len(df), "no device ops in the signposted region — was the impl skipped (wrong device count)?"

    total_ns = df[dur_col].sum()
    by_op = (
        df.groupby("OP CODE")[dur_col]
        .agg(count="count", total_ns="sum", avg_ns="mean")
        .sort_values("total_ns", ascending=False)
    )
    by_op["pct"] = 100.0 * by_op["total_ns"] / total_ns

    # Manual formatting (pandas to_string can truncate long tables) — print every op.
    header = f"{'OP CODE':<44}{'count':>7}{'total_ms':>12}{'avg_us':>12}{'pct':>8}"
    rows = [
        f"{op:<44}{int(r['count']):>7}{r['total_ns']/1e6:>12.3f}{r['avg_ns']/1e3:>12.1f}{r['pct']:>7.1f}%"
        for op, r in by_op.iterrows()
    ]
    table = "\n".join(
        [
            f"DeepSeek V3.2 MLA chunked perf — {CHUNK_TOKENS}-tok chunk @ {CACHE_TOKENS}-tok cache, SP=4×TP=2",
            f"critical-path device-kernel time over the chunk (device-collapsed: compute=max, "
            f"collectives=avg across chips): {total_ns/1e6:.3f} ms across {int(by_op['count'].sum())} op calls",
            header,
            "-" * len(header),
            *rows,
        ]
    )
    logger.info("\n" + table)
    print("\n" + table)  # ensure full table reaches stdout even if logging is filtered

    by_op.reset_index().to_csv(CSV_OUT, index=False)
    logger.info(f"per-op CSV written to {os.path.abspath(CSV_OUT)}")
