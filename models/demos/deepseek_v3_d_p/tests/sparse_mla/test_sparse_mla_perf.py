# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Realtime-profiler perf harness for the DeepSeek V3.2 / GLM-5.1 MLA (DSA) chunked-prefill layer.

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

Profiler — realtime (lightweight), in-process (replaces Tracy)
--------------------------------------------------------------
Timing comes from the in-process realtime device profiler
(``tests/ttnn/profiling/realtime_profiler_utils.profile_realtime_program``), not Tracy. It is
auto-enabled by device open on eligible hardware (real Blackhole, WORKER dispatch, fabric-tensix
datamover off); the test gates on ``ttnn.device.IsProgramRealtimeProfilerActive()``. This removes the
Tracy subprocess, signposts, and ops-CSV re-parse — the same swap PR #49840 made for the sparse-MLA
CCL benchmarks (LoudBox: ~11× faster wall-clock). Two semantic notes versus the Tracy harness:

  * Reporting unit is the device **program** (one ``runtime_id`` per op dispatch). A record carries only
    kernel-source paths, so the ``OP CODE`` column is mapped from those paths to a tracy-style op code
    (``_op_code``: a priority table over the operations dir, verified against the tracy op-code
    counts/durations). One program = one op code, matching Tracy's op names — not per-instance identical
    to a tracy op struct, but the same code space the report/graph pipeline consumes.
  * Multi-chip collapse takes the **max** ``duration_ns`` across chips for every program (the slowest
    chip gates that program's critical path). Tracy used max for compute and avg for collectives, so
    collective-heavy numbers can differ a few percent (see PR #49840 deltas). Both express the same
    per-step critical-path quantity.

Each measured ``forward()`` is profiled as its own region (register callback → run one forward →
drain), because a cached op re-dispatched across the cold loop reuses its ``runtime_id`` — so
per-forward regions are what make the cold per-iteration sum correct (and replace the old MLA_START
signpost split). The run total is the sum of per-forward criticals.

Single test (was a two-test tracy driver+impl split):
  * test_mla_chunked_perf — parametrized over [deepseek_v32, glm_5_1] × [warm, cold, long] ×
    [sparse, dense]. Builds the DSA ttMLA (variant from the ``variant`` fixture) and, per scenario,
    measures one forward over the (zero-init) block-cyclic caches (warm/long) or a chunk loop that
    fills them (cold), profiling each forward under the realtime profiler. Prints a per-op table and
    writes a per-(scenario, variant, mode) CSV under generated/profiler/<variant>_<mode>_mla_perf/.

Three scenarios (the test sweeps all three):
  * warm — production step: one `chunk`-token forward at start=cache over a `cache`-length prefix. Both
    block-cyclic caches (indexer index_kv_cache + KVPE) are left at init — no warm-up forwards; for a perf
    proxy only op shapes/timing matter, and those are set by the full `total` prefix width the gather+score
    span, not the cache contents. Measures a single steady-state chunk.
  * cold — full cold prefill: forward chunks start=0,chunk,…,cache with real forwards that grow both
    caches (both by per-chunk block-cyclic slab writes). The measured region spans ALL chunks = the
    total cold-start prefill cost; the final chunk (start=cache) is exactly the `warm` step. Besides the
    aggregate per-op table, cold also emits a per-cache-fill-iteration breakdown (…_cold_by_iter.csv:
    iteration, cache_depth_tokens, total_ns, op_count) showing how the per-chunk critical path grows as
    the cache fills — recovered by profiling each forward as its own region.
  * long — like `warm` but with a 0.5M-token Galaxy cache (512000 = 100 chunks), to profile a single
    chunk over a long prefix. Like the others the cache scales by SP/8, so per-chip depth stays
    Galaxy-equal on every box (LoudBox=128k, QuietBox=64k box-local cache).

variant axis — deepseek_v32 (128 q-heads / 64 index heads) vs glm_5_1 (64 / 32). Both run the SAME TP=4
  meshes: GLM's thin per-chip head shard (64/4=16 < 32) is handled by the head→sequence reshard in
  ttMLA._sparse_mla (#48727) plus the head-replicated seq-sharded indexer, so GLM is no longer TP-capped.
  All model dims come from the single-source reference config (reference/{deepseek_v3_2,glm_5_1}_config.py).

attn_mode axis — a baseline to compare the sparse impl against:
  * sparse — v3.2 DSA: indexer builds top-k index keys, sparse_sdpa attends only the top-k=2048 keys.
  * dense  — v3.1 baseline: has_indexer=False -> NullIndexer + full-prefix ring MLA (ring_joint_sdpa
    over the whole prefix, no indexer/top-k). Needs no cache fill (ring reads the prefix by logical_n).
  Each (variant, mode) writes its own profiler subdir ({variant}_{sparse,dense}_mla_perf) so the runs
  never clobber and the CSVs stay directly comparable.

Run (Blackhole Galaxy/LoudBox/QuietBox) — all combos (2 variants × 3 scenarios × 2 modes), or narrow via -k:
    pytest -m perf models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py::test_mla_chunked_perf -s
    pytest -m perf ...::test_mla_chunked_perf -k "glm_5_1 and cold and sparse" -s
    pytest -m perf ...::test_mla_chunked_perf -k "deepseek_v32 and dense" -s

Knobs (env): DS_PERF_CACHE (default 51200), DS_PERF_CHUNK (default 5120), DS_PERF_LONG_CACHE (default
512000), DS_PERF_CSV / DS_DENSE_PERF_CSV (summary filename, per-scenario suffix appended; written under
generated/profiler/{variant}_{mode}_mla_perf/), DS_PERF_RT_TIMEOUT (realtime-profiler record drain
ceiling in seconds, default 30). DS_PERF_CHUNK is the Galaxy-global target chunk; smaller boxes scale
BOTH the measured chunk and the cache by SP/8; cache must stay a whole chunk multiple. DS_PERF_VARIANT /
DS_PERF_SCENARIO / DS_PERF_ATTN_MODE remain as the module-level defaults used for mesh-shape detection,
but the test itself sweeps the full matrix via parametrization.

NOTE: warm/long leave both block-cyclic caches at zero init rather than warming with real chunks — only
op shapes/timing matter here, not values, and those come from the full `total` prefix width (allocation),
not the cache contents; cold instead runs real chunk forwards that fill the caches. The indexer rope
scales from the HF config (mla.py), so `config.max_seq_len = total` is enough for a 50k+ (and 0.5M)
context (no manual rope bump).
"""

import copy
import csv
import datetime
import json
import os
from dataclasses import dataclass

import pandas as pd
import pytest
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_2_config import deepseek_v32_hf_config
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import glm_hf_config
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import detect_num_devices
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_plugin import is_marker_explicitly_selected
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import make_hidden
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

CACHE_TOKENS = int(os.environ.get("DS_PERF_CACHE", 51200))  # 50 * 1024 already cached
CHUNK_TOKENS = int(os.environ.get("DS_PERF_CHUNK", 5120))  # 5 * 1024 processed this step
# Long-sequence cache (Galaxy target). 512000 = 100 * 5120 == "0.5M" — a whole multiple of the 5k
# chunk (the rope table requires total = cache + chunk to be a multiple of chunk). Override with
# DS_PERF_LONG_CACHE (must stay a chunk multiple), e.g. 522240 (=102 chunks ≈ 512*1024).
LONG_CACHE_TOKENS = int(os.environ.get("DS_PERF_LONG_CACHE", 512000))
# attn_mode axis: sparse (v3.2 DSA indexer + sparse_sdpa) vs dense (v3.1 full-prefix ring MLA — no
# indexer, no top-k), a baseline to compare the sparse impl against. Each mode writes its own profiler
# subdir + per-scenario CSVs so the two runs never clobber and stay directly comparable.
ATTN_MODES = ("sparse", "dense")
ATTN_MODE = os.environ.get("DS_PERF_ATTN_MODE", "sparse")  # module-level default (mesh-shape detection)
# Model-variant axis: deepseek_v32 (128 q-heads / 64 index heads) vs glm_5_1 (64 / 32). BOTH run the
# SAME TP=4 meshes — GLM's thin per-chip head shard (64/4=16 < 32) is handled by the head→sequence
# reshard in ttMLA._sparse_mla (#48727) plus the head-replicated seq-sharded indexer, so no TP cap
# applies. Every model dimension comes from the single-source reference config
# (reference/{deepseek_v3_2,glm_5_1}_config.py), never hardcoded here.
VARIANTS = ("deepseek_v32", "glm_5_1")
VARIANT = os.environ.get("DS_PERF_VARIANT", "deepseek_v32")
_CONFIG_BUILDERS = {"deepseek_v32": deepseek_v32_hf_config, "glm_5_1": glm_hf_config}

# Fabric transport being profiled — single source for BOTH the device_params and the run manifest, so the
# recorded provenance can never drift from what actually ran (FABRIC_2D is the production transport;
# FABRIC_1D exhibited the multi-hop line-broadcast hang). FABRIC_2D + fabric_router_config leaves the
# fabric-tensix datamover off, so the realtime profiler stays eligible (see PR #49840 CCL benchmarks).
PERF_FABRIC = ttnn.FabricConfig.FABRIC_2D
# Matches the production GLM adapters: enough room for all_gather's global semaphores without reducing
# the static-CB headroom required by the dense MLA path.
L1_SMALL_SIZE = 512

# Realtime-profiler record drain ceiling. The receiver thread delivers records asynchronously; the
# wrapper stops once no new record has landed for its settle window, bounded by this ceiling. A generous
# default covers a many-program forward across up to 32 chips; each forward is profiled as its own
# (volume-bounded) region so this is a safety ceiling, not the expected wait.
RT_RECORD_TIMEOUT_S = float(os.environ.get("DS_PERF_RT_TIMEOUT", 30.0))

# When set, also write an execution-ordered per-call CSV (the RT analog of tracy's ops_perf_results):
# one device-collapsed row per program per forward, ordered by start tick — the input a per-call graph
# attribution (parse_percall) needs. Off by default (the summary CSVs are the normal output).
RT_OPS_DUMP = os.environ.get("DS_PERF_RT_OPS_DUMP", "") not in ("", "0", "false")


def _subdir(variant: str, mode: str) -> str:
    """Per-(variant, mode) profiler subdir (per-scenario summary CSVs + run manifest). Keeps
    deepseek_v32/glm_5_1 × sparse/dense runs from clobbering each other."""
    return f"{variant}_{mode}_mla_perf"


def _csv_name(variant: str, mode: str) -> str:
    return os.environ.get("DS_PERF_CSV" if mode == "sparse" else "DS_DENSE_PERF_CSV", f"{_subdir(variant, mode)}.csv")


# Three profiling scenarios (the test sweeps all three):
#   warm  — the production step: one chunk over a pre-filled `cache` (indexer K-cache populated
#           directly, no warm-up forwards). Measures a single steady-state chunk.
#   cold  — full cold prefill: iteratively forward chunks start=0,chunk,…,cache (real forwards that
#           grow the caches). The measured region spans ALL chunks = total cold-start prefill cost;
#           the final chunk (start=cache) is exactly the `warm` case.
#   long  — like `warm` but with a 0.5M-token cache, to profile a single chunk over a long prefix.
SCENARIOS = {
    "warm": {"cache": CACHE_TOKENS, "loop": False},
    "cold": {"cache": CACHE_TOKENS, "loop": True},
    "long": {"cache": LONG_CACHE_TOKENS, "loop": False},
}
SCENARIO = os.environ.get("DS_PERF_SCENARIO", "warm")


_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo root is five levels up (sparse_mla → tests → deepseek_v3_d_p → demos → models → <root>).
_REPO_ROOT = os.path.normpath(os.path.join(_REPO_DIR, *([os.pardir] * 5)))


def _output_dir(subdir: str) -> str:
    """Per-(variant, mode) output dir for summary CSVs + run manifest. Replaces Tracy's
    PROFILER_ARTIFACTS_DIR now that profiling is in-process (no tracy report tree)."""
    d = os.path.join(_REPO_ROOT, "generated", "profiler", subdir)
    os.makedirs(d, exist_ok=True)
    return d


def _scenario_csv(out_dir, scenario: str, variant: str, mode: str) -> str:
    """Per-(scenario, variant, mode) summary CSV path under the output dir."""
    root, ext = os.path.splitext(_csv_name(variant, mode))
    return os.path.join(out_dir, f"{root}_{scenario}{ext}")


def _git_head() -> dict:
    """Commit + branch of the working tree, read straight from the .git metadata — NO subprocess, NO
    dirty/diff. The workflow makes the recorded commit truthful by construction: the caller commits before
    profiling, and the skill's run wrapper enforces a clean tree when not run by hand — so there is no
    working-tree delta to fingerprint. Handles a linked worktree (`.git` is a file redirecting to a gitdir,
    as the baseline sweep uses) and both HEAD forms: a symbolic ref (resolved via the loose ref, then
    packed-refs in the common dir) or a bare detached SHA. Best-effort — any failure degrades to nulls so
    provenance never breaks the run."""
    try:
        d = _REPO_DIR
        while not os.path.exists(os.path.join(d, ".git")):
            parent = os.path.dirname(d)
            if parent == d:
                return {"commit": None, "branch": None}
            d = parent
        git_entry = os.path.join(d, ".git")
        if os.path.isfile(git_entry):  # worktree: '.git' file → 'gitdir: <path>'
            gitdir = open(git_entry).read().split("gitdir:", 1)[1].strip()
            gitdir = os.path.normpath(os.path.join(d, gitdir))
        else:
            gitdir = git_entry
        commondir_file = os.path.join(gitdir, "commondir")  # refs/packed-refs live in the common dir
        common = (
            os.path.normpath(os.path.join(gitdir, open(commondir_file).read().strip()))
            if os.path.exists(commondir_file)
            else gitdir
        )
        head = open(os.path.join(gitdir, "HEAD")).read().strip()
        if not head.startswith("ref:"):
            return {"commit": head, "branch": None}  # detached HEAD → bare SHA
        ref = head.split("ref:", 1)[1].strip()
        branch = ref[len("refs/heads/") :] if ref.startswith("refs/heads/") else ref
        loose = os.path.join(common, ref)
        if os.path.exists(loose):
            return {"commit": open(loose).read().strip(), "branch": branch}
        packed = os.path.join(common, "packed-refs")  # ref may only exist packed
        if os.path.exists(packed):
            for line in open(packed):
                line = line.strip()
                if line and not line.startswith(("#", "^")) and line.endswith(" " + ref):
                    return {"commit": line.split(" ", 1)[0], "branch": branch}
        return {"commit": None, "branch": branch}
    except Exception as e:  # noqa: BLE001 — provenance must never break the run
        logger.warning(f"run manifest: could not read git HEAD ({e}); commit/branch will be null")
        return {"commit": None, "branch": None}


def _write_run_manifest(report_dir, *, variant, scenario, attn_mode, command, workload) -> None:
    """Drop a lean run_manifest_<scenario>.json into the output dir. Records ONLY what cannot be
    reconstructed from git (given the commit) or from the co-located ops CSV:
      * commit / branch — the code-state anchor (read subprocess-free from .git; no dirty flag — the
        workflow commits before profiling);
      * device / mesh / fabric — where and how it ran (runtime facts, not in source);
      * build.so_mtime — the stale-build guard (not in git, not in the CSV);
      * command — a copy-paste reproducer (env-prefixed).
    Deliberately omitted because they are recoverable: commit subject/time & the full model config (from
    ``git show <commit>`` + reference/{variant}_config.py), the workload sizes (derived from config + mesh
    + scenario), and per-op measurements (the summary CSV sitting in this same dir). Never raises."""
    try:
        so = os.path.join(_REPO_ROOT, "ttnn", "ttnn", "_ttnn.so")
        so_mtime = (
            datetime.datetime.fromtimestamp(os.path.getmtime(so), datetime.timezone.utc).isoformat()
            if os.path.exists(so)
            else None
        )
        reproducer = (
            f"DS_PERF_CACHE={CACHE_TOKENS} DS_PERF_CHUNK={CHUNK_TOKENS} DS_PERF_LONG_CACHE={LONG_CACHE_TOKENS} "
            f"{command} -k '{variant} and {scenario} and {attn_mode}'"
        )
        head = _git_head()
        manifest = {
            "schema_version": 3,
            "profiler": "realtime",
            "variant": variant,
            "scenario": scenario,
            "attn_mode": attn_mode,
            "commit": head["commit"],
            "branch": head["branch"],
            "device": {
                "num_devices": workload.num_devices,
                "box": workload.system_name,
                "mesh_sp": workload.sp,
                "mesh_tp": workload.tp,
                "fabric": getattr(PERF_FABRIC, "name", str(PERF_FABRIC)),
            },
            "build": {"so_mtime": so_mtime},
            "command": reproducer,
        }
        path = os.path.join(report_dir, f"run_manifest_{scenario}.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"run manifest written to {path}")
    except Exception as e:  # noqa: BLE001 — provenance must never break the run
        logger.warning(f"run manifest: failed to write ({e}); this dump will lack provenance")


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
# Head counts / index dims are NOT constants here — they come from the reference config per variant
# (deepseek_v32: 128/64, glm_5_1: 64/32; see _detect_perf_workload). GALAXY_SP/GALAXY_TP are the
# production mesh topology (shared by both variants), not model dims, so they stay in the harness.


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


def _detect_perf_workload(variant_name: str) -> tuple[PerfWorkload, str | None]:
    num_devices = detect_num_devices()
    system = _SYSTEM_BY_DEVICE_COUNT.get(num_devices)
    if system is None:
        placeholder = PerfWorkload("unsupported", num_devices, (1, 1), CHUNK_TOKENS, 32, 16)
        return placeholder, (
            "sparse MLA perf supports Blackhole QuietBox/LoudBox/Galaxy only " f"(detected {num_devices} chips)"
        )

    system_name, mesh_shape = system
    sp, tp = mesh_shape
    # Head counts come from the single-source reference config for the variant (deepseek_v32: 128/64,
    # glm_5_1: 64/32) — the same builder the config_only fixture resolves, so device and harness agree.
    cfg = _CONFIG_BUILDERS[variant_name]()
    local_chunk = _exact_div(CHUNK_TOKENS, GALAXY_SP, "DS_PERF_CHUNK")
    local_heads = _exact_div(cfg.num_attention_heads, GALAXY_TP, f"{variant_name}.num_attention_heads")
    local_index_heads = _exact_div(cfg.index_n_heads, GALAXY_TP, f"{variant_name}.index_n_heads")
    workload = PerfWorkload(
        system_name=system_name,
        num_devices=num_devices,
        mesh_shape=mesh_shape,
        chunk_tokens=local_chunk * sp,
        num_attention_heads=local_heads * tp,
        index_n_heads=local_index_heads * tp,
    )
    return workload, None


PERF_WORKLOAD, PERF_SKIP_REASON = _detect_perf_workload(VARIANT)


@pytest.fixture(autouse=True, scope="module")
def _require_perf(request):
    if is_marker_explicitly_selected(request.config, "perf"):
        return
    pytest.skip("sparse MLA perf tests require explicit marker selection: pytest -m perf")


# ============================================================================
# Realtime-profiler helpers (mirror PR #49840 CCL harness + SDPA nightly suite)
# ============================================================================
def _require_rt_profiler() -> None:
    """The realtime profiler is auto-enabled on eligible hardware; a False here means the current
    dispatch/fabric config disabled it (see realtime_profiler_manager eligibility) — fail loudly rather
    than silently mis-measure with an empty record set."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for sparse MLA perf checks (eligible Blackhole HW required)")


def _op_label(kernel_sources) -> str:
    """Best-effort human label for a device program from its kernel-source paths. RT records carry no
    Tracy OP CODE, so derive the op kind from the ttnn operations path (``.../operations/<name>/...``,
    unwrapping an ``experimental`` prefix); fall back to the kernel basenames. Programs of the same op
    kind share kernels, so this groups like the old per-OP-CODE table — kernel-derived, not exact."""
    names = set()
    for src in kernel_sources:
        parts = src.replace("\\", "/").split("/")
        if "operations" in parts:
            i = parts.index("operations") + 1
            if i < len(parts):
                name = parts[i]
                if name == "experimental" and i + 1 < len(parts):
                    name = parts[i + 1]
                names.add(name)
    if names:
        return "+".join(sorted(names))
    basenames = {os.path.splitext(os.path.basename(s))[0] for s in kernel_sources}
    return "+".join(sorted(basenames)) if basenames else "unknown"


# kernel-path substring -> tracy-style OP CODE, in PRIORITY order (first match wins). Ordering resolves
# programs whose kernel set spans dirs: the op-defining kernel must precede its helpers/epilogue —
# untilize_with_unpadding before untilize, tilize_with_val_padding before tilize, typecast before copy,
# the indexer rope (rotary_embedding_indexed) before its shared llama rope kernel, and every real op
# before the trailing eltwise/unary epilogue. Codes chosen to match the tracy device-op names so the
# existing per-call graph attribution (parse_percall + its alias sets) consumes this dump unchanged.
# Verified against the tracy op-code counts/durations for deepseek_v32 warm/sparse.
_OP_CODE_RULES = (
    # ring_mla (dense) fuses a ring all-gather with the joint-SDPA compute; its compute kernel lives under
    # transformer/sdpa/ (ring_joint_sdpa.cpp), so it must be matched BEFORE the generic sparse-SDPA rule.
    ("/ring_joint_sdpa", "RingJointSDPA"),
    ("/ring_attention_all_gather", "RingJointSDPA"),
    ("/transformer/sdpa/", "SDPA"),
    ("/indexer_score/", "IndexerScore"),
    ("/topk_large_indices/", "TopkLargeIndices"),
    ("/ccl/all_gather_async/", "AllGatherAsync"),
    ("/ccl/reduce_scatter_minimal_async/", "ReduceScatterMinimalAsync"),
    ("/ccl/broadcast/", "AllBroadcast"),
    ("/nlp_create_qkv_heads", "NlpCreateHeads"),
    ("/nlp_concat_heads", "NlpConcatHeads"),
    ("/rotary_embedding_indexed/", "RotaryEmbeddingIndexed"),
    ("/rotary_embedding_llama/", "RotaryEmbeddingLlama"),
    ("/update_padded_kv_cache/", "UpdateCache"),
    ("/fast_reduce_nc/", "FastReduceNC"),
    ("/matmul/", "Matmul"),
    ("/layernorm/", "LayerNorm"),
    ("/untilize_with_unpadding/", "UntilizeWithUnpadding"),
    ("/tilize_with_val_padding/", "TilizeWithValPadding"),
    ("/untilize/", "Untilize"),
    ("/tilize/", "Tilize"),
    ("/concat/", "Concat"),
    ("/permute/", "Permute"),
    ("/slice/", "Slice"),
    ("/fill_pad/", "FillPad"),
    ("/typecast/", "Typecast"),
    ("/copy/", "Copy"),
    ("/binary", "BinaryNg"),
    ("/unary", "UnaryEltwise"),
)


def _op_code(kernel_sources) -> str:
    """Translate a program's kernel-source paths to a tracy-style OP CODE (one program = one op), so the
    per-call dump — and the summary table — carry op identity the tracy graph pipeline already understands.
    Priority match over _OP_CODE_RULES (op-defining kernel wins over helpers/epilogue); falls back to the
    coarse operations-dir label for any op not in the table."""
    paths = "\n".join(src.replace("\\", "/") for src in kernel_sources)
    for needle, code in _OP_CODE_RULES:
        if needle in paths:
            return code
    return _op_label(kernel_sources)


def _write_ops_dump(out_dir: str, name_root: str, forwards: list) -> str:
    """Execution-ordered per-call CSV — the RT analog of tracy's ops_perf_results. One device-collapsed
    row per program (duration = max across chips) per forward, in program order (the per_program dict is
    keyed by first arrival, which equals device execution order — see _profile_forward). Also carries the
    raw kernel_sources so the op-code translation can be authored/verified from real data."""
    path = os.path.join(out_dir, f"{name_root}_ops.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["forward", "seq", "runtime_id", "OP CODE", "DEVICE KERNEL DURATION [ns]", "kernel_sources"])
        for forward_index, per_program in enumerate(forwards):
            for seq, (runtime_id, info) in enumerate(per_program.items()):
                w.writerow(
                    [
                        forward_index,
                        seq,
                        runtime_id,
                        _op_code(info["kernel_sources"]),
                        f"{info['duration_ns']:.3f}",
                        "|".join(info["kernel_sources"]),
                    ]
                )
    return path


def _profile_forward(mesh_device, run_fn) -> dict:
    """Profile one region and collapse the device dimension: return {runtime_id -> {"duration_ns",
    "kernel_sources"}} where duration_ns is the MAX across chips for that program (slowest chip = that
    program's critical path). Mirrors PR #49840's _profile_programs. runtime_id 0 is the profiler's
    sentinel and is skipped. The dict preserves first-arrival order, which equals device execution
    (dispatch) order — the profiler delivers records in program order; verified equal to the device
    start-tick order across every case/forward — so callers get the ops in program order for free."""
    _, records = profile_realtime_program(
        mesh_device, run_fn, collect_all=True, record_timeout_seconds=RT_RECORD_TIMEOUT_S
    )
    per_program: dict = {}
    for record in records:
        runtime_id = record["runtime_id"]
        if not runtime_id:
            continue
        duration_ns = float(record["duration_ns"])
        current = per_program.get(runtime_id)
        if current is None:
            per_program[runtime_id] = {"duration_ns": duration_ns, "kernel_sources": record["kernel_sources"]}
        else:
            current["duration_ns"] = max(current["duration_ns"], duration_ns)
    assert per_program, "real-time profiler returned no valid program records for the measured forward"
    return per_program


def _programs_to_frame(per_program: dict, dur_col: str) -> pd.DataFrame:
    """One row per device program: (OP CODE label, critical-path duration)."""
    return pd.DataFrame(
        [{"OP CODE": _op_code(info["kernel_sources"]), dur_col: info["duration_ns"]} for info in per_program.values()]
    )


def _by_op(frame: pd.DataFrame, dur_col: str) -> pd.DataFrame:
    """Aggregate per-op-label: count, total_ns, avg_ns, pct — sorted by total desc (old table shape)."""
    total_ns = frame[dur_col].sum()
    by_op = (
        frame.groupby("OP CODE")[dur_col]
        .agg(count="count", total_ns="sum", avg_ns="mean")
        .sort_values("total_ns", ascending=False)
    )
    by_op["pct"] = 100.0 * by_op["total_ns"] / total_ns
    return by_op


# ============================================================================
# The perf test — build the DSA ttMLA, profile the measured forward(s), report
# ============================================================================
@pytest.mark.parametrize("mesh_device", [PERF_WORKLOAD.mesh_shape], ids=[PERF_WORKLOAD.id], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": PERF_FABRIC,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": L1_SMALL_SIZE,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        }
    ],
    ids=["fabric2d"],
    indirect=True,
)
@pytest.mark.parametrize("attn_mode", list(ATTN_MODES), ids=list(ATTN_MODES))
@pytest.mark.parametrize("scenario", list(SCENARIOS), ids=list(SCENARIOS))
@pytest.mark.parametrize("variant", list(VARIANTS), indirect=True, ids=list(VARIANTS))
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test — skip on CI")
@pytest.mark.timeout(0)
def test_mla_chunked_perf(mesh_device, variant, scenario, attn_mode, config_only):
    if PERF_SKIP_REASON:
        pytest.skip(PERF_SKIP_REASON)
    _require_rt_profiler()

    # Workload is variant-specific (head counts differ); the mesh/SP is shared. Resolve per parametrized
    # variant so labels + head counts match the variant under test (module-level VARIANT may differ).
    workload, skip_reason = _detect_perf_workload(variant.name)
    if skip_reason:
        pytest.skip(skip_reason)

    scenario_cfg = SCENARIOS[scenario]
    is_cold = scenario_cfg["loop"]
    has_indexer = attn_mode == "sparse"  # dense baseline drops the indexer -> full-prefix ring MLA
    subdir = _subdir(variant.name, attn_mode)  # per-(variant, mode) dir: runs never clobber each other
    sp_axis, tp_axis = 0, 1
    sp, tp = mesh_device.shape
    # cache scales per box (sp/GALAXY_SP) like the chunk, so every box profiles the Galaxy per-chip
    # workload: constant chunks-to-fill and Galaxy-equal per-chip depth (see _local_cache_tokens).
    galaxy_cache = scenario_cfg["cache"]
    cache, chunk = _local_cache_tokens(galaxy_cache, sp), workload.chunk_tokens
    total = cache + chunk
    assert (sp, tp) == workload.mesh_shape, f"expected mesh {workload.mesh_shape}, got {(sp, tp)}"
    # cache must be a whole number of chunks: the cold loop steps by `chunk`, and the indexed rope
    # table (get_rope_tensors_indexed) requires total = cache + chunk to be a multiple of chunk.
    assert cache % chunk == 0, f"cache {cache} must be a whole number of {chunk}-token chunks"
    assert total % sp == 0 and (total // sp) % 32 == 0, f"total {total} must be tile-aligned per SP={sp} chip"
    assert config_only.num_attention_heads % tp == 0 and config_only.index_n_heads % tp == 0, (
        f"{variant.name} heads (MLA={config_only.num_attention_heads}, index={config_only.index_n_heads}) "
        f"must be divisible by TP={tp}"
    )

    # Every model dimension (heads, index_*, rope interleave) comes from the single-source reference
    # config (config_only → reference/{deepseek_v3_2,glm_5_1}_config.py). Mirror the correctness tests:
    # set only max_seq_len and let ttMLA read all dims from the config — no per-variant overrides here.
    config = copy.deepcopy(config_only)
    config.max_seq_len = total  # rope-table / buffer length (same hack as the correctness tests)
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

    # cold: forward every chunk start=0,chunk,…,cache (the last, start=cache, is the warm step). warm/long:
    # one forward at start=cache. Each forward is profiled as its OWN realtime-profiler region: a cached op
    # reuses its runtime_id across forwards, so per-forward regions are what make the cold per-iteration
    # sum correct (and replace the old signposted-region split).
    starts = list(range(0, cache + chunk, chunk)) if is_cold else [cache]
    logger.info(
        f"profiling {workload.system_name} {attn_mode}/{scenario} proxy: {len(starts)} × {chunk}-token "
        f"chunk(s) filling to end_pos={total} on SP={sp}×TP={tp}; local chunk={chunk // sp}, "
        f"local MLA heads={config.num_attention_heads // tp}"
        + (f", local indexer heads={config.index_n_heads // tp}" if has_indexer else " (dense: no indexer)")
    )

    def _one_forward(start):
        out = mla.forward(tt_x, rope, kvpe_cache, actual_start=start, index_kv_cache=index_kv_cache)
        ttnn.deallocate(out)

    forwards = []  # one {runtime_id -> {...}} per measured forward (device-collapsed to critical path)
    for start in starts:
        ttnn.synchronize_device(mesh_device)  # drain prior programs so only this forward contributes records
        forwards.append(_profile_forward(mesh_device, lambda start=start: _one_forward(start)))

    dur_col = "DEVICE KERNEL DURATION [ns]"  # kept for downstream compatibility (holds RT critical-path ns)
    frame = pd.concat([_programs_to_frame(pp, dur_col) for pp in forwards], ignore_index=True)
    assert len(frame), "no device programs in the measured region — was the impl skipped (wrong device count)?"

    total_ns = frame[dur_col].sum()
    by_op = _by_op(frame, dur_col)

    # Manual formatting (pandas to_string can truncate long tables) — print every op.
    header = f"{'OP CODE':<44}{'count':>7}{'total_ms':>12}{'avg_us':>12}{'pct':>8}"
    rows = [
        f"{op:<44}{int(r['count']):>7}{r['total_ns']/1e6:>12.3f}{r['avg_ns']/1e3:>12.1f}{r['pct']:>7.1f}%"
        for op, r in by_op.iterrows()
    ]
    span = f"full cold prefill 0→{cache}-tok cache" if is_cold else f"one chunk @ {cache}-tok cache"
    table = "\n".join(
        [
            f"{variant.name} MLA chunked perf [{attn_mode}/{scenario}] — {workload.system_name} proxy "
            f"{workload.chunk_tokens}-tok chunk, {span}, SP={workload.sp}×TP={workload.tp}",
            f"Galaxy target: {CHUNK_TOKENS}-tok chunk @ {galaxy_cache}-tok cache, SP={GALAXY_SP}×TP={GALAXY_TP}; "
            f"local chunk={CHUNK_TOKENS // GALAXY_SP}, local MLA heads={workload.num_attention_heads // GALAXY_TP}",
            f"critical-path device-kernel time over the {'prefill' if is_cold else 'chunk'} "
            f"(realtime profiler; per-program max across chips): "
            f"{total_ns/1e6:.3f} ms across {int(by_op['count'].sum())} device programs",
            "(OP CODE = tracy-style op code mapped from kernel sources; see module docstring)",
            header,
            "-" * len(header),
            *rows,
        ]
    )
    logger.info("\n" + table)
    print("\n" + table)  # ensure full table reaches stdout even if logging is filtered

    out_dir = _output_dir(subdir)
    csv_out = _scenario_csv(out_dir, scenario, variant.name, attn_mode)
    by_op.reset_index().to_csv(csv_out, index=False)
    logger.info(f"per-op CSV written to {os.path.abspath(csv_out)}")

    if RT_OPS_DUMP:
        ops_csv = _write_ops_dump(out_dir, os.path.splitext(os.path.basename(csv_out))[0], forwards)
        logger.info(f"per-call ops dump written to {os.path.abspath(ops_csv)}")

    # Provenance: drop run_manifest_<scenario>.json next to the summary CSV so every dump self-documents
    # the commit, command, and device/mesh/fabric that produced it (config and per-op measurements are
    # deliberately omitted — recoverable from git + reference config + the co-located ops CSV).
    command = (
        "pytest -m perf models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py::test_mla_chunked_perf -s"
    )
    _write_run_manifest(
        out_dir, variant=variant.name, scenario=scenario, attn_mode=attn_mode, command=command, workload=workload
    )

    # cold only: per-cache-fill-iteration breakdown. The aggregate above sums all chunks; this shows how
    # the per-chunk critical path grows as the cache fills (the point of the cold scenario). Each forward
    # was profiled as its own region, so iteration i is simply forwards[i] — no signpost splitting.
    if not is_cold:
        return
    per_op_rows, totals = [], []
    for i, per_program in enumerate(forwards):
        seg = _programs_to_frame(per_program, dur_col)
        tot = seg[dur_col].sum()
        g = _by_op(seg, dur_col).reset_index()
        g.insert(0, "cache_depth_tokens", i * chunk)  # tokens already cached when this chunk ran
        g.insert(0, "iteration", i)
        per_op_rows.append(g)
        totals.append((i, i * chunk, tot, len(g)))
    by_iter_op = pd.concat(per_op_rows, ignore_index=True)

    # stdout: compact per-iteration totals (the full per-op×iteration detail goes to the CSV — too many
    # rows to print). The CSV pivots to a per-op-by-iteration matrix for charting an op's growth.
    iter_header = f"{'iter':>4}{'cache_depth':>12}{'total_ms':>12}{'ops':>6}"
    iter_table = "\n".join(
        [
            f"cold per-iteration critical path [{variant.name}/{workload.system_name}] "
            f"(realtime profiler; last iter == the `warm` step):",
            iter_header,
            "-" * len(iter_header),
            *(f"{i:>4}{d:>12}{t/1e6:>12.3f}{n:>6}" for i, d, t, n in totals),
        ]
    )
    logger.info("\n" + iter_table)
    print("\n" + iter_table)
    iter_csv = _scenario_csv(out_dir, f"{scenario}_by_iter", variant.name, attn_mode)
    by_iter_op.to_csv(iter_csv, index=False)
    logger.info(f"per-op×iteration CSV written to {os.path.abspath(iter_csv)}")
