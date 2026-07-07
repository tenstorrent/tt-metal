# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tracy perf harness for the DeepSeek V3.2 MLA (DSA) chunked-prefill layer.

Production scenario (defaults): process one **5k-token chunk** with **50k tokens already cached**,
on the Galaxy **SP=8 × TP=4** mesh.

The test can also run on smaller Blackhole boxes by profiling a per-chip Galaxy slice. BOTH the chunk
and the cache scale by SP/8, so smaller boxes run a proportionally shorter sequence — the per-chip
workload mirrors Galaxy rather than a heavier one:
  * Galaxy   (32 chips): SP=8 × TP=4, chunk=5120, cache=50k,   heads=128 (full workload)
  * LoudBox  (8 chips):  SP=2 × TP=4, chunk=1280, cache=12.5k, heads=128 (1/4 sequence length)
  * QuietBox (4 chips):  SP=1 × TP=4, chunk=640,  cache=6.25k, heads=128 (1/8 sequence length)

That keeps the per-chip COMPUTE shapes equal to Galaxy: local query rows/chip (640), MLA heads/chip
(32), indexer heads/chip (16), the per-chip KVPE depth (cache/SP = 6.25k on every box), AND the number
of chunks-to-fill in `cold` (11 on every box, not 41 on LoudBox). CAVEAT: the indexer K-cache is
replicated full-depth (= the box-local cache), so on smaller boxes it holds a proportionally SHORTER
prefix than Galaxy — only Galaxy exercises the true 50k (or 0.5M) indexer/top-k depth; smaller boxes
under-represent any op that scales with the replicated key-cache length.
No reference values: this just runs the real device forward and reports per-op device-kernel time.
Multi-chip rows are device-collapsed (compute=max, collectives=avg across chips) via merge_device_rows
so the reported time is per-step critical path, not the ~8× over-count of summing parallel device rows.

Three scenarios (DS_PERF_SCENARIO; the driver sweeps all three):
  * warm — production step: one `chunk`-token forward at start=cache over a `cache`-length prefix. Both
    block-cyclic caches (indexer index_kv_cache + KVPE) are left at init — no warm-up forwards; for a perf
    proxy only op shapes/timing matter, and those are set by the full `total` prefix width the gather+score
    span, not the cache contents. Measures a single steady-state chunk.
  * cold — full cold prefill: forward chunks start=0,chunk,…,cache with real forwards that grow both
    caches (both by per-chunk block-cyclic slab writes). The signposted region spans ALL chunks = the
    total cold-start prefill cost; the final chunk (start=cache) is exactly the `warm` step. Besides the
    aggregate per-op table, cold also emits a per-cache-fill-iteration breakdown (…_cold_by_iter.csv:
    iteration, cache_depth_tokens, total_ns, op_count) showing how the per-chunk critical path grows as
    the cache fills — recovered by splitting the ops log on each forward's MLA_START marker.
  * long — like `warm` but with a 0.5M-token Galaxy cache (512000 = 100 chunks), to profile a single
    chunk over a long prefix. Like the others the cache scales by SP/8, so per-chip depth stays
    Galaxy-equal on every box (LoudBox=128k, QuietBox=64k box-local cache).

Two-test pattern (mirrors tests/nightly/blackhole/sdpa):
  * test_mla_chunked_perf_impl  — the work to profile. Builds the v32 ttMLA and, per DS_PERF_SCENARIO,
    either measures one forward over the (zero-init) block-cyclic caches (warm/long) or forwards a chunk
    loop that fills them (cold), wrapping the measured forward(s) in signpost("start"/"stop"). Run under tracy.
  * test_mla_chunked_perf       — the driver, parametrized over [warm, cold, long] × [sparse, dense].
    Spawns the impl under tracy via run_device_profiler (passing DS_PERF_SCENARIO + DS_PERF_ATTN_MODE),
    reads the device ops log for the signposted region, prints a per-op table, and writes a per-(scenario,
    mode) CSV under the per-mode profiler dir.

attn_mode axis — a baseline to compare the sparse impl against:
  * sparse — v3.2 DSA: indexer builds top-k index keys, sparse_sdpa attends only the top-k=2048 keys.
  * dense  — v3.1 baseline: has_indexer=False -> NullIndexer + full-prefix ring MLA (ring_joint_sdpa
    over the whole prefix, no indexer/top-k). Needs no cache fill (ring reads the prefix by logical_n).
  Each mode writes its own profiler subdir (deepseek_v32_{sparse,dense}_mla_perf) so the runs never
  clobber and the CSVs stay directly comparable.

Run (Blackhole Galaxy/LoudBox/QuietBox) — all six combos, or narrow via -k:
    pytest -m perf models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py::test_mla_chunked_perf -s
    pytest -m perf ...::test_mla_chunked_perf -k "cold and sparse" -s
    pytest -m perf ...::test_mla_chunked_perf -k dense -s

Knobs (env): DS_PERF_SCENARIO (warm|cold|long, default warm for a standalone impl run),
DS_PERF_CACHE (default 51200), DS_PERF_CHUNK (default 5120), DS_PERF_LONG_CACHE (default 512000),
DS_PERF_CSV (summary filename, per-scenario suffix appended; written under the tracy profiler dir
generated/profiler/deepseek_v32_sparse_mla_perf/). DS_PERF_CHUNK is the Galaxy-global target chunk; smaller
boxes scale BOTH the measured chunk and the cache by SP/8; cache must stay a whole chunk multiple.

NOTE: warm/long leave both block-cyclic caches at zero init rather than warming with real chunks — only
op shapes/timing matter here, not values, and those come from the full `total` prefix width (allocation),
not the cache contents; cold instead runs real chunk forwards that fill the caches. The indexer rope
scales from the HF config (mla.py), so `config.max_seq_len = total` is enough for a 50k+ (and 0.5M)
context (no manual rope bump).
"""

import copy
import os
from dataclasses import dataclass
from unittest import mock

import pandas as pd
import pytest
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
# Long-sequence cache (Galaxy target). 512000 = 100 * 5120 == "0.5M" — a whole multiple of the 5k
# chunk (the rope table requires total = cache + chunk to be a multiple of chunk). Override with
# DS_PERF_LONG_CACHE (must stay a chunk multiple), e.g. 522240 (=102 chunks ≈ 512*1024).
LONG_CACHE_TOKENS = int(os.environ.get("DS_PERF_LONG_CACHE", 512000))
# attn_mode axis: sparse (v3.2 DSA indexer + sparse_sdpa) vs dense (v3.1 full-prefix ring MLA — no
# indexer, no top-k), a baseline to compare the sparse impl against. Each mode writes its own tracy
# profiler subdir + per-scenario CSVs so the two runs never clobber and stay directly comparable.
ATTN_MODES = ("sparse", "dense")
ATTN_MODE = os.environ.get("DS_PERF_ATTN_MODE", "sparse")  # the impl selects its mode from this


def _subdir(mode: str) -> str:
    """Per-mode tracy profiler subdir (holds the raw device reports + the per-scenario summary CSVs)."""
    return f"deepseek_v32_{mode}_mla_perf"


def _csv_name(mode: str) -> str:
    return os.environ.get("DS_PERF_CSV" if mode == "sparse" else "DS_DENSE_PERF_CSV", f"{_subdir(mode)}.csv")


# Three profiling scenarios (select the impl's via DS_PERF_SCENARIO; the driver sweeps all three):
#   warm  — the production step: one chunk over a pre-filled `cache` (indexer K-cache populated
#           directly, no warm-up forwards). Measures a single steady-state chunk.
#   cold  — full cold prefill: iteratively forward chunks start=0,chunk,…,cache (real forwards that
#           grow the caches). The signposted region spans ALL chunks = total cold-start prefill cost;
#           the final chunk (start=cache) is exactly the `warm` case.
#   long  — like `warm` but with a 0.5M-token cache, to profile a single chunk over a long prefix.
SCENARIOS = {
    "warm": {"cache": CACHE_TOKENS, "loop": False},
    "cold": {"cache": CACHE_TOKENS, "loop": True},
    "long": {"cache": LONG_CACHE_TOKENS, "loop": False},
}
SCENARIO = os.environ.get("DS_PERF_SCENARIO", "warm")


def _scenario_csv(out_dir, scenario: str, mode: str) -> str:
    """Per-(scenario, mode) summary CSV path under the tracy profiler dir (next to the raw reports)."""
    root, ext = os.path.splitext(_csv_name(mode))
    return os.path.join(out_dir, f"{root}_{scenario}{ext}")


def _local_cache_tokens(galaxy_cache: int, sp: int) -> int:
    """Box-local cached sequence length: scale the Galaxy-global cache by SP/GALAXY_SP exactly like the
    chunk, so every box profiles the Galaxy per-chip workload rather than a heavier one. This keeps the
    number of chunks-to-fill constant (Galaxy=11, LoudBox=11, QuietBox=11 — NOT 41) and the per-chip
    KVPE depth Galaxy-equal (cache/SP = galaxy_cache/GALAXY_SP on every box). LoudBox runs 1/4 the
    sequence length, QuietBox 1/8. CACHE_TOKENS/LONG_CACHE_TOKENS are multiples of GALAXY_SP and of the
    per-box chunk, so the result stays an exact chunk multiple (required by the indexed rope table)."""
    return galaxy_cache * sp // GALAXY_SP


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
        placeholder = PerfWorkload("unsupported", num_devices, (1, 1), CHUNK_TOKENS, 32, 16)
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

    scenario = SCENARIOS[SCENARIO]
    is_cold = scenario["loop"]
    has_indexer = ATTN_MODE == "sparse"  # dense baseline drops the indexer -> full-prefix ring MLA
    sp_axis, tp_axis = 0, 1
    sp, tp = mesh_device.shape
    # cache scales per box (sp/GALAXY_SP) like the chunk, so every box profiles the Galaxy per-chip
    # workload: constant chunks-to-fill and Galaxy-equal per-chip depth (see _local_cache_tokens).
    cache, chunk = _local_cache_tokens(scenario["cache"], sp), PERF_WORKLOAD.chunk_tokens
    total = cache + chunk
    assert (sp, tp) == PERF_WORKLOAD.mesh_shape, f"expected mesh {PERF_WORKLOAD.mesh_shape}, got {(sp, tp)}"
    # cache must be a whole number of chunks: the cold loop steps by `chunk`, and the indexed rope
    # table (get_rope_tensors_indexed) requires total = cache + chunk to be a multiple of chunk.
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
        has_indexer=has_indexer,  # sparse: DSA indexer + sparse_sdpa; dense: NullIndexer + ring MLA
    )

    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors_indexed(total, chunk)
    # KVPE cache format is mode-specific. sparse: sparse_sdpa reads it natively and requires an
    # uncompressed bf16/fp8_e4m3 ROW_MAJOR cache (mla.py asserts) — NOT the init_kvpe_cache bfloat8_b/TILE
    # default. dense: ring_joint_sdpa wants the default (bfloat8_b TILE) and derives its output dtype from
    # the cache, so leave dense on the default.
    kvpe_dtype_layout = dict(dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) if has_indexer else {}
    kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=total,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        **kvpe_dtype_layout,
    )

    # Block-cyclic indexer key cache (SPARSE only): allocated externally (same ownership as the KVPE cache)
    # and passed into forward. warm/long leave it (and the KVPE cache) at zero init — for a profiling proxy
    # the cache CONTENTS don't affect op shapes/timing (the gather + score always cover the full
    # `total`-length prefix), so representing the `cache` already-processed tokens needs no warm-up write.
    # cold fills both caches for real via its own per-chunk block-cyclic slab writes (start=0,chunk,…,cache).
    # DENSE has no indexer (NullIndexer) — ring MLA reads the prefix by logical_n (= total), not by cached
    # index data, so it needs no index cache (index_kv_cache stays None).
    index_kv_cache = None
    if has_indexer:
        index_kv_cache = init_kvpe_cache(
            kvpe_cache_head_dim=mla._indexer.index_args.index_head_dim,
            mesh_device=mesh_device,
            seq_len=total,
            mesh_shape=list(mesh_device.shape),
            sp_axis=sp_axis,
            num_kvpe_cache_layers=1,
            num_users=1,
            dtype=ttnn.bfloat8_b,
        )

    hidden = make_hidden(chunk, config.hidden_size, seed=42)  # one chunk of input (reused per forward)
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

    # cold: forward every chunk start=0,chunk,…,cache (the last, start=cache, is the warm step) and
    # signpost the WHOLE loop = total cold-start prefill cost. warm/long: one forward at start=cache.
    starts = list(range(0, cache + chunk, chunk)) if is_cold else [cache]
    logger.info(
        f"profiling {PERF_WORKLOAD.system_name} {ATTN_MODE}/{SCENARIO} proxy: {len(starts)} × {chunk}-token "
        f"chunk(s) filling to end_pos={total} on SP={sp}×TP={tp}; local chunk={chunk // sp}, "
        f"local MLA heads={config.num_attention_heads // tp}"
        + (f", local indexer heads={config.index_n_heads // tp}" if has_indexer else " (dense: no indexer)")
    )
    signpost("start")
    for start in starts:
        out = mla.forward(tt_x, rope, kvpe_cache, actual_start=start, index_kv_cache=index_kv_cache)
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")


# ============================================================================
# Outer: drive the impl under tracy, post-process, print + write CSV
# ============================================================================
@pytest.mark.parametrize("attn_mode", list(ATTN_MODES), ids=list(ATTN_MODES))
@pytest.mark.parametrize("scenario", list(SCENARIOS), ids=list(SCENARIOS))
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test — run locally with tracy")
@pytest.mark.timeout(0)
def test_mla_chunked_perf(scenario, attn_mode):
    from tracy.common import PROFILER_ARTIFACTS_DIR
    from tracy.process_model_log import run_device_profiler

    if PERF_SKIP_REASON:
        pytest.skip(PERF_SKIP_REASON)

    subdir = _subdir(attn_mode)  # per-mode profiler dir (sparse/dense never clobber each other)

    # merge_device_rows: the deepseek_v3_d_p / tt_transformers convention for collapsing the device
    # dimension of a multi-chip Tracy ops log (see models/demos/deepseek_v3_d_p/utils/perf_utils.py).
    from models.tt_transformers.tests.test_utils import merge_device_rows
    from tests.nightly.sdpa_perf_utils import post_process_ops_log

    galaxy_cache = SCENARIOS[scenario]["cache"]
    cache = _local_cache_tokens(galaxy_cache, PERF_WORKLOAD.sp)  # box-scaled (matches the impl)
    is_cold = SCENARIOS[scenario]["loop"]

    # The impl is skipif(CI=="true"); CI=false in the subprocess lets it run there (mirrors the
    # tests/nightly/blackhole/sdpa perf pattern). DS_PERF_SCENARIO selects which scenario the impl runs.
    # The driver itself opens no device, so when the gate is run by node-id only the tracy subprocess
    # opens the board — no parent CHIP_IN_USE lock contention.
    command = (
        "pytest -m perf models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py"
        "::test_mla_chunked_perf_impl"
    )
    # run_device_profiler defaults op_support_count to ~1333 ops/device — the tracy device-profiler
    # ring buffer silently drops ops beyond it. cold forwards ~11 chunks (~800 ops); raise the cap so a
    # future op-count bump can't silently truncate the log (which would corrupt the per-op totals).
    with mock.patch.dict(os.environ, {"CI": "false", "DS_PERF_SCENARIO": scenario, "DS_PERF_ATTN_MODE": attn_mode}):
        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"], op_support_count=5000)

    dur_col = "DEVICE KERNEL DURATION [ns]"
    # Rows between signpost("start") and signpost("stop") = the measured chunk's device ops, with ONE
    # ROW PER (op call × mesh chip). On the selected SP×TP mesh every op runs across chips in parallel,
    # so the raw rows must NOT be summed — that over-counts wall-clock by ~num_devices. Collapse
    # the device dimension to one row per logical op call with the standard merge_device_rows rule:
    #   * compute ops -> MAX duration across chips (the slowest chip gates the step = critical path)
    #   * collectives -> AVG duration across chips (all chips run the same collective together)
    # region = the sliced signpost("start")..signpost("stop") window, still holding the per-forward
    # MLA_START/MLA_END signpost rows (used below to split cold into per-iteration segments).
    region = post_process_ops_log(subdir, has_signposts=True)
    region[dur_col] = pd.to_numeric(region[dur_col], errors="coerce")
    df = merge_device_rows(region)  # aggregate; filters to tt_dnn_device rows internally
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
    span = f"full cold prefill 0→{cache}-tok cache" if is_cold else f"one chunk @ {cache}-tok cache"
    table = "\n".join(
        [
            f"DeepSeek V3.2 MLA chunked perf [{attn_mode}/{scenario}] — {PERF_WORKLOAD.system_name} proxy "
            f"{PERF_WORKLOAD.chunk_tokens}-tok chunk, {span}, SP={PERF_WORKLOAD.sp}×TP={PERF_WORKLOAD.tp}",
            f"Galaxy target: {CHUNK_TOKENS}-tok chunk @ {galaxy_cache}-tok cache, SP=8×TP=4; "
            f"local chunk={CHUNK_TOKENS // GALAXY_SP}, local MLA heads={GALAXY_NUM_HEADS // GALAXY_TP}",
            f"critical-path device-kernel time over the {'prefill' if is_cold else 'chunk'} "
            f"(device-collapsed: compute=max, collectives=avg across chips): "
            f"{total_ns/1e6:.3f} ms across {int(by_op['count'].sum())} op calls",
            header,
            "-" * len(header),
            *rows,
        ]
    )
    logger.info("\n" + table)
    print("\n" + table)  # ensure full table reaches stdout even if logging is filtered

    csv_out = _scenario_csv(PROFILER_ARTIFACTS_DIR / subdir, scenario, attn_mode)
    by_op.reset_index().to_csv(csv_out, index=False)
    logger.info(f"per-op CSV written to {os.path.abspath(csv_out)}")

    # cold only: per-cache-fill-iteration breakdown. The aggregate above sums all chunks; this shows
    # how the per-chunk critical path grows as the cache fills (the point of the cold scenario). Each
    # forward emits its own MLA_START marker, so split the region on those and device-collapse each
    # segment independently (merge_device_rows drops the interleaved signpost rows internally).
    if is_cold:
        ri = region.reset_index(drop=True)
        starts = list(ri.index[ri["OP CODE"] == "MLA_START"])
        bounds = starts + [len(ri)]
        chunk = PERF_WORKLOAD.chunk_tokens
        # Per-op × per-iteration: the SAME per-op table as the aggregate above (OP CODE, count,
        # total_ns, avg_ns, pct — sorted desc), but computed for each iteration's segment and tagged
        # with iteration + cache_depth_tokens, so each op's time can be tracked as the cache fills.
        per_op_rows, totals = [], []
        for i in range(len(starts)):
            seg = merge_device_rows(ri.iloc[bounds[i] + 1 : bounds[i + 1]])
            tot = seg[dur_col].sum()
            g = (
                seg.groupby("OP CODE")[dur_col]
                .agg(count="count", total_ns="sum", avg_ns="mean")
                .sort_values("total_ns", ascending=False)
            )
            g["pct"] = 100.0 * g["total_ns"] / tot
            g = g.reset_index()
            g.insert(0, "cache_depth_tokens", i * chunk)  # tokens already cached when this chunk ran
            g.insert(0, "iteration", i)
            per_op_rows.append(g)
            totals.append((i, i * chunk, tot, len(g)))
        by_iter_op = pd.concat(per_op_rows, ignore_index=True)

        # stdout: compact per-iteration totals (the full per-op×iteration detail goes to the CSV — too
        # many rows to print). pivot the CSV to a per-op-by-iteration matrix for charting an op's growth.
        iter_header = f"{'iter':>4}{'cache_depth':>12}{'total_ms':>12}{'ops':>6}"
        iter_table = "\n".join(
            [
                f"cold per-iteration critical path [{PERF_WORKLOAD.system_name}] "
                f"(device-collapsed; last iter == the `warm` step):",
                iter_header,
                "-" * len(iter_header),
                *(f"{i:>4}{d:>12}{t/1e6:>12.3f}{n:>6}" for i, d, t, n in totals),
            ]
        )
        logger.info("\n" + iter_table)
        print("\n" + iter_table)
        iter_csv = _scenario_csv(PROFILER_ARTIFACTS_DIR / subdir, f"{scenario}_by_iter", attn_mode)
        by_iter_op.to_csv(iter_csv, index=False)
        logger.info(f"per-op×iteration CSV written to {os.path.abspath(iter_csv)}")
