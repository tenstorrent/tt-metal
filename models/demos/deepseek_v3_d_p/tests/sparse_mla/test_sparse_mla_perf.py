# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tracy perf harness for the DeepSeek V3.2 MLA (DSA) chunked-prefill layer.

Production scenario (defaults): process one **5k-token chunk** with **50k tokens already cached**,
on the Galaxy **SP=8 × TP=4** mesh.

The test can also run on smaller Blackhole boxes by profiling a per-chip Galaxy slice:
  * Galaxy   (32 chips): SP=8 × TP=4, chunk=5120, heads=128
  * LoudBox  (8 chips):  SP=2 × TP=4, chunk=1280, heads=128
  * QuietBox (4 chips):  SP=1 × TP=4, chunk=640,  heads=128

That keeps the per-chip COMPUTE shapes equal to Galaxy: local query rows/chip (640), MLA heads/chip
(32), indexer heads/chip (16). The indexer K-cache is replicated full-depth (50k) on every chip and the
sparse SDPA is top-k-gated, so kernel timing is a faithful Galaxy proxy. CAVEAT: the KVPE cache is
SP-sharded, so its per-chip depth is 50k/SP (50k on QuietBox, 25k on LoudBox, 6.25k on Galaxy) — per-chip
KVPE footprint and any op that scales with KVPE depth are heavier on smaller boxes.
No reference values: this just runs the real device forward and reports per-op device-kernel time.
Multi-chip rows are device-collapsed (compute=max, collectives=avg across chips) via merge_device_rows
so the reported time is per-step critical path, not the ~8× over-count of summing parallel device rows.

Two-test pattern (mirrors tests/nightly/blackhole/sdpa):
  * test_mla_chunked_perf_impl  — the work to profile. Builds the v32 ttMLA, populates the index/KV
    caches directly to stand in for the cached prefix (no warm-up forwards), then wraps a SINGLE
    `chunk`-token forward in signpost("start"/"stop"). Run this under tracy.
  * test_mla_chunked_perf       — the driver. Spawns the impl under tracy via run_device_profiler,
    reads the device ops log for the signposted region, prints a per-op table, and writes a CSV.

Run (Blackhole Galaxy/LoudBox/QuietBox):
    pytest -m perf models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py::test_mla_chunked_perf -s

Knobs (env): DS_PERF_CACHE (default 51200), DS_PERF_CHUNK (default 5120), DS_PERF_CSV.
DS_PERF_CHUNK is the Galaxy-global target chunk; smaller boxes scale the measured chunk by SP/8.

NOTE: caches are POPULATED directly (random index keys; KVPE left at init) rather than warmed with
real chunks — only op shapes/timing matter here, not values. The indexer rope now scales from the HF
config (mla.py), so `config.max_seq_len = total` is enough for a 50k+ context (no manual rope bump).
"""

import copy
import os
from dataclasses import dataclass
from unittest import mock

import pandas as pd
import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import detect_num_devices
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_plugin import is_marker_explicitly_selected
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import make_hidden
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE

CACHE_TOKENS = int(os.environ.get("DS_PERF_CACHE", 51200))  # 50 * 1024 already cached
CHUNK_TOKENS = int(os.environ.get("DS_PERF_CHUNK", 5120))  # 5 * 1024 processed this step
SUBDIR = "deepseek_v32_sparse_mla_perf"
CSV_OUT = os.environ.get("DS_PERF_CSV", "deepseek_v32_sparse_mla_perf.csv")

pytestmark = pytest.mark.perf

GALAXY_SP = 8
GALAXY_TP = 4
GALAXY_NUM_HEADS = 128
GALAXY_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128


@dataclass(frozen=True)
class PerfWorkload:
    system_name: str
    num_devices: int
    mesh_shape: tuple[int, int]
    cache_tokens: int
    chunk_tokens: int
    num_attention_heads: int
    index_n_heads: int

    @property
    def sp(self) -> int:
        return self.mesh_shape[0]

    @property
    def tp(self) -> int:
        return self.mesh_shape[1]

    @property
    def id(self) -> str:
        return f"{self.system_name.lower()}_sp{self.sp}xtp{self.tp}"


_SYSTEM_BY_DEVICE_COUNT = {
    4: ("QuietBox", (1, 4)),
    8: ("LoudBox", (2, 4)),
    32: ("Galaxy", (8, 4)),
}


def _exact_div(numerator: int, denominator: int, label: str) -> int:
    if numerator % denominator != 0:
        raise ValueError(f"{label}={numerator} must be divisible by {denominator}")
    return numerator // denominator


def _detect_perf_workload() -> tuple[PerfWorkload, str | None]:
    num_devices = detect_num_devices()
    system = _SYSTEM_BY_DEVICE_COUNT.get(num_devices)
    if system is None:
        placeholder = PerfWorkload("unsupported", num_devices, (1, 1), CACHE_TOKENS, CHUNK_TOKENS, 32, 16)
        return placeholder, (
            "DeepSeek V3.2 sparse MLA perf supports Blackhole QuietBox/LoudBox/Galaxy only "
            f"(detected {num_devices} chips)"
        )

    system_name, mesh_shape = system
    sp, tp = mesh_shape
    local_chunk = _exact_div(CHUNK_TOKENS, GALAXY_SP, "DS_PERF_CHUNK")
    local_heads = _exact_div(GALAXY_NUM_HEADS, GALAXY_TP, "GALAXY_NUM_HEADS")
    local_index_heads = _exact_div(GALAXY_INDEX_HEADS, GALAXY_TP, "GALAXY_INDEX_HEADS")
    workload = PerfWorkload(
        system_name=system_name,
        num_devices=num_devices,
        mesh_shape=mesh_shape,
        cache_tokens=CACHE_TOKENS,
        chunk_tokens=local_chunk * sp,
        num_attention_heads=local_heads * tp,
        index_n_heads=local_index_heads * tp,
    )
    return workload, None


PERF_WORKLOAD, PERF_SKIP_REASON = _detect_perf_workload()


@pytest.fixture(autouse=True, scope="module")
def _require_perf(request):
    if is_marker_explicitly_selected(request.config, "perf"):
        return
    pytest.skip("sparse MLA perf tests require explicit marker selection: pytest -m perf")


# ============================================================================
# Inner: the work to profile (run under tracy by the driver below)
# ============================================================================
@pytest.mark.parametrize("mesh_device", [PERF_WORKLOAD.mesh_shape], ids=[PERF_WORKLOAD.id], indirect=True)
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
@pytest.mark.parametrize("variant", ["deepseek_v32"], indirect=True, ids=["deepseek_v32"])
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test — skip on CI")
@pytest.mark.timeout(0)
def test_mla_chunked_perf_impl(mesh_device, device_params, variant, config_only):
    from tracy import signpost

    if PERF_SKIP_REASON:
        pytest.skip(PERF_SKIP_REASON)

    cache, chunk = PERF_WORKLOAD.cache_tokens, PERF_WORKLOAD.chunk_tokens
    total = cache + chunk
    sp_axis, tp_axis = 0, 1
    sp, tp = mesh_device.shape
    assert (sp, tp) == PERF_WORKLOAD.mesh_shape, f"expected mesh {PERF_WORKLOAD.mesh_shape}, got {(sp, tp)}"
    assert cache % chunk == 0, f"cache {cache} must be a whole number of {chunk}-token chunks"
    assert total % sp == 0 and (total // sp) % 32 == 0, f"total {total} must be tile-aligned per SP={sp} chip"
    assert (
        PERF_WORKLOAD.num_attention_heads // tp == GALAXY_NUM_HEADS // GALAXY_TP
    ), "local MLA heads/chip must match Galaxy"

    config = copy.deepcopy(config_only)
    config.max_seq_len = total  # rope-table / buffer length (same hack as the correctness tests)
    config.num_attention_heads = PERF_WORKLOAD.num_attention_heads
    if hasattr(config, "num_key_value_heads"):
        config.num_key_value_heads = PERF_WORKLOAD.num_attention_heads
    config.index_n_heads = PERF_WORKLOAD.index_n_heads
    config.index_head_dim = getattr(config, "index_head_dim", INDEX_HEAD_DIM)
    config.index_topk = getattr(config, "index_topk", 2048)
    config.index_rope_interleave = getattr(config, "index_rope_interleave", False)
    weights = random_mla_weights(config)

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
    mla._indexer._index_kbuf = ttnn.from_torch(
        torch.randn(1, 1, cache, mla._indexer.index_args.index_head_dim, dtype=torch.bfloat16),
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

    logger.info(
        f"profiling {PERF_WORKLOAD.system_name} proxy: one {chunk}-token chunk @ {cache}-token cache "
        f"(end_pos={total}) on SP={sp}×TP={tp}; local chunk={chunk // sp}, "
        f"local MLA heads={config.num_attention_heads // tp}, local indexer heads={config.index_n_heads // tp}"
    )
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

    if PERF_SKIP_REASON:
        pytest.skip(PERF_SKIP_REASON)

    # merge_device_rows: the deepseek_v3_d_p / tt_transformers convention for collapsing the device
    # dimension of a multi-chip Tracy ops log (see models/demos/deepseek_v3_d_p/utils/perf_utils.py).
    from models.tt_transformers.tests.test_utils import merge_device_rows
    from tests.nightly.sdpa_perf_utils import post_process_ops_log

    # The impl is skipif(CI=="true"); CI=false in the subprocess lets it run there (mirrors the
    # tests/nightly/blackhole/sdpa perf pattern). The driver itself opens no device, so when the gate is
    # run by node-id only the tracy subprocess opens the board — no parent CHIP_IN_USE lock contention.
    command = (
        "pytest -m perf models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py"
        "::test_mla_chunked_perf_impl"
    )
    with mock.patch.dict(os.environ, {"CI": "false"}):
        run_device_profiler(command, SUBDIR, device_analysis_types=["device_kernel_duration"])

    dur_col = "DEVICE KERNEL DURATION [ns]"
    # Rows between signpost("start") and signpost("stop") = the measured chunk's device ops, with ONE
    # ROW PER (op call × mesh chip). On the selected SP×TP mesh every op runs across chips in parallel,
    # so the raw rows must NOT be summed — that over-counts wall-clock by ~num_devices. Collapse
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
            f"DeepSeek V3.2 MLA chunked perf — {PERF_WORKLOAD.system_name} proxy "
            f"{PERF_WORKLOAD.chunk_tokens}-tok chunk @ {PERF_WORKLOAD.cache_tokens}-tok cache, "
            f"SP={PERF_WORKLOAD.sp}×TP={PERF_WORKLOAD.tp}",
            f"Galaxy target: {CHUNK_TOKENS}-tok chunk @ {CACHE_TOKENS}-tok cache, SP=8×TP=4; "
            f"local chunk={CHUNK_TOKENS // GALAXY_SP}, local MLA heads={GALAXY_NUM_HEADS // GALAXY_TP}",
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
