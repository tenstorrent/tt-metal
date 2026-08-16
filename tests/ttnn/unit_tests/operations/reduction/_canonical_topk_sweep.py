# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Canonical device-kernel-runtime sweep for the top-k / sorting op family.

WHY THIS EXISTS
---------------
The branch has kernel-level (MATH_ISOLATE, cyc/vector) numbers from the tt-llk
perf drivers and a single op-level Tracy baseline for ttnn.topk. Neither is a
table you can put in front of a reviewer: the LLK numbers exclude unpack/pack/
dispatch, and the op-level baseline covers one op at a handful of shapes with
no A/B arm. This script produces ONE table across ops x (N, K, dtype) x arms,
at both layers:

  - ttnn layer : DEVICE KERNEL DURATION [ns] from Tracy, per (op, N, K, dtype).
  - llk layer  : cyc/vector from tt_metal/tt-llk/perf_data/*.post.csv
                 (perf_topk_rebuild_xl.py carries baseline and ours as paired
                 arms in one run, so it needs no header flip).

MEASUREMENT DISCIPLINE (inherited from _topk_sort_bench.py, and extended)
-------------------------------------------------------------------------
- Device Kernel Duration from Tracy, never time.perf_counter.
- Every cell runs in a FRESH subprocess under a watchdog: a hang in one
  (shape, k) cannot kill the sweep, and each Tracy report contains exactly
  one cell, which makes CSV attribution unambiguous.
- Per-config attribution joins on OP CODE and takes the LAST `iters` rows by
  GLOBAL CALL COUNT, dropping the cache-miss/warmup invocations.
- mean +/- std across `--trials` independent subprocess runs; a speedup is
  printed only when |delta| > 2 * pooled_std, otherwise "~1.00 (noise)".
- Unsupported configurations are RECORDED WITH THEIR ERROR, not silently
  dropped. Cells the constraint model predicts as unsupported are still
  attempted once in the baseline arm: the real error message is the datum,
  and a prediction/reality mismatch is itself a finding.
- The A/B arms flip `#define TOPK_REPLAY_STEP_{LOAD,STORE}` at the top of
  tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h. There is no
  env-var -> kernel-define mechanism in tt-metal (checked: the topk program
  factories pass no compute defines; jit_build injects defines only from
  rtoptions), so the header edit IS the mechanism. It is guarded behind
  --allow-header-edit, asserted against `git diff`, restored on exit, and the
  JIT kernel cache is cleared between arms as the hard guarantee (the dephash
  sidecar would catch the header change anyway, but a guarantee beats a
  mechanism you have to trust).

USAGE
-----
  # (a) baseline-only, ttnn layer
  python tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py \
      --arms baseline --layers ttnn --out generated/canonical_sweep/run1

  # (b) full three-arm A/B, both layers
  python tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py \
      --arms baseline,disable_replay --layers ttnn,llk \
      --allow-header-edit --out generated/canonical_sweep/run1

  # (c) report-only (rebuild CSV+markdown from an existing out dir)
  python tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py \
      --report --out generated/canonical_sweep/run1

  # resume an interrupted run (skips cells already MEASURED/UNSUPPORTED)
  ... --resume --out generated/canonical_sweep/run1

The child measurement process is this same file: the orchestrator launches
`python -m tracy -r -v <this file>` with CANONICAL_SWEEP_CHILD_SPEC pointing
at a JSON cell spec. Args ride an env var, not the command line, because
tracy -r re-invokes via shell=True and mangles anything with spaces
(docs/profiling.md).

COMPETITION MODE (--competition): HOW TO RERUN
----------------------------------------------
Deterministic K x W "competition table" across four measured layers plus the
llm_perf roofline model. K in {512, 1024, 2048} x W in {2048 ... 262144} by
default (override with --ks / --ns, where --ns is W). Layers, in the FIXED
run order (header-editing layer last):

  op        ttnn.experimental.topk_large_indices alone, single row
  opstock   topk_large_indices AS SHIPPED (pre-branch): measured via the
            rows=2 row-parallel proxy -- rows=1 now auto-selects our
            column-parallel factory, but two independent rows select the
            row-parallel factory whose kernels are byte-identical to
            pre-branch and run concurrently on 2 cores, so op device time
            equals the single-row single-core as-shipped time
  routed    ttnn.topk largest=True  (routes to topk_large_indices; composite)
  stocknow  ttnn.topk largest=False (stock single-core path, header
            as-committed: replay ON by default since the branch landed it)
  prebranch stocknow with `#define TOPK_DISABLE_REPLAY_STEP 1` temporarily
            armed in ckernel_sfpu_topk.h (pre-branch stock behavior);
            needs --allow-header-edit, restored+cache-cleared in `finally`
  blaze     OPT-IN via --with-blaze: the tt-blaze GLM indexer bench as an
            external pytest under our Tracy; single cell k=2048 W=65536 (their
            bench is fixed-shape). Needs /home/nachiket/tt-blaze with a
            tt-metal symlink into this repo. Runs between stocknow and
            prebranch. Its row carries the fused-program caveat.
  roofline  constants from tenstorrent/llm_perf (aspirational; see table)

  # full competition run (three non-editing layers + prebranch)
  python tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py \
      --competition --allow-header-edit --out generated/canonical_sweep/comp1

  # without --allow-header-edit the prebranch layer is SKIPPED, the rest run
  python ..._canonical_topk_sweep.py --competition --out .../comp1

  # single-layer rerun (e.g. just the routed layer), same out dir
  python ..._canonical_topk_sweep.py --competition --layers-competition routed \
      --out .../comp1

  # resume an interrupted run (skips MEASURED/UNSUPPORTED/WRONG cells)
  python ..._canonical_topk_sweep.py --competition --allow-header-edit \
      --resume --out .../comp1

Every competition cell record is stamped with the git HEAD sha, an md5 of
`git diff --stat` (working-tree fingerprint) and the mtime+md5 of
ttnn/ttnn/_ttnn.so, so a mid-run rebuild or a dirtied tree is visible in the
output instead of silently corrupting the A/B (we got burned by exactly this).
Correctness is verified BEFORE timing; a failing cell reports status=WRONG and
its timing never enters the table. Output: competition_table.csv +
competition_table.md in --out.

Underscore-prefixed so routine `pytest tests/...` does not collect it.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

REPO = os.environ.get("TT_METAL_HOME", "/home/nachiket/tt-metal")
SCRIPT_PATH = os.path.abspath(__file__)

CHILD_SPEC_ENV = "CANONICAL_SWEEP_CHILD_SPEC"


def _script_md5():
    """md5 of THIS script's bytes on disk right now. The child stamps it into
    its manifest and the orchestrator compares per cell: the two processes
    import the file at different times, so a mid-run edit can make them run
    DIFFERENT code while looking like one harness (see the HARNESS_BUG note
    in _build_cell_callable)."""
    import hashlib

    with open(SCRIPT_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# A/B arms: the LLK header edit.
#
# The #ifdef sites live inside _bitonic_topk_phases_steps, which is the
# function ttnn.topk's topk_local_sort actually executes; a JIT header change
# needs no host rebuild (tt_metal/tt-llk/common is on the JIT include path,
# jit_build/build.cpp:348).
#
# HISTORY: the branch has since COMMITTED the replay optimization default-ON
# (ckernel_sfpu_topk.h: `#if !defined(TOPK_DISABLE_REPLAY_STEP) &&
# !defined(TOPK_REPLAY_STEP_STORE)` arms STORE, which implies LOAD). So the
# arms today are:
#   baseline        header as-committed = replay ON (the old "replay_store")
#   disable_replay  TOPK_DISABLE_REPLAY_STEP armed  = pre-branch stock kernel
# The old replay_load / replay_store arm names are retired: their effect is
# now the committed default, and inserting their defines would be a no-op
# that MEASURES AS baseline while claiming to be an arm.
# ---------------------------------------------------------------------------
HEADER_RELPATH = "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h"
MARKER_BEGIN = "// BEGIN CANONICAL_SWEEP_ARM (auto-managed by _canonical_topk_sweep.py;"
MARKER_BEGIN_FULL = MARKER_BEGIN + " a stray copy means a sweep died mid-arm -- safe to delete this block)"
MARKER_END = "// END CANONICAL_SWEEP_ARM"
ARM_DEFINES = {
    "baseline": None,
    "disable_replay": "#define TOPK_DISABLE_REPLAY_STEP 1",
}
KERNEL_CACHE_DIR = os.path.expanduser("~/.cache/tt-metal-cache")

# Blackhole user-visible L1 per core; feeds the a-priori single-core cost model.
BH_L1_BYTES = 1_572_864
TILE_BYTES = {"bf16": 2048, "fp32": 4096, "uint16": 2048, "uint32": 4096}


# Tracy OP CODE is the device-operation struct name. Verified against the
# 2026_08_16 baseline CSV (TopKDeviceOperation) and the C++ struct names
# (SortDeviceOperation, TopkLargeIndicesDeviceOperation,
# GeneralizedMoeGateDeviceOperation). Matched loosely so a rename does not
# silently zero the table; FillPad/Pad rows must never match.
def _op_code_matches(op, op_code):
    s = op_code.lower().replace("_", "")
    if "pad" in s:
        return False
    if op == "topk":
        return s.startswith("topk") and "large" not in s
    if op == "sort":
        return "sort" in s
    if op == "topk_large_indices":
        return "topklargeindices" in s
    if op == "moe_gate":
        return "moegate" in s
    return False


# ---------------------------------------------------------------------------
# Grid + a-priori constraint model.
#
# Sources (audited on this tree, quoted by file:line so drift is checkable):
#   ttnn.topk   ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp
#               multi-core iff W>=8192 (:66) AND W<65535 strict (:70) AND
#               pow2(W) (:72) AND k<=64 (:75) AND find_topk_core_config fits.
#               Single-core L1 wall: verify_single_core_cost, topk_utils.cpp:220.
#   ttnn.sort   ttnn/cpp/ttnn/operations/data_movement/sort/device/sort_device_operation.cpp
#               host pads W to next pow2 (min 64); Wt<=64 -> single core,
#               else hybrid/DRAM multi-core. dtype in {bf16, uint16, fp32}.
#   topk_large_indices  .../experimental/topk_large_indices/device/..._device_operation.cpp
#               Blackhole only, ROW_MAJOR BFLOAT16 interleaved,
#               k in [16, 2048] multiples of 16, N in [k, 2^30].
#   moe_gate    .../experimental/deepseek/moe/generalized_moe_gate/device/...
#               topk in {4, 6, 8}, bf16, sharded, 256 experts fixed geometry.
# ---------------------------------------------------------------------------
DEFAULT_NS = "1024,8192,32768,65534,65536,131072"
DEFAULT_KS = "8,16,32,64,128,256,512,1024,2048"
DEFAULT_DTYPES = "bf16,fp32"

# The two shipped production perf anchors for topk_large_indices
# (tests/ttnn/nightly/.../test_topk_large_indices.py::TOPK_LARGE_INDICES_PRODUCTION_PERF_CONFIGS).
LARGE_INDICES_ANCHORS = [
    # (tag, rows, n, valid_length, k)
    ("prod_prefill", 640, 51200, None, 1536),
    ("prod_bounded_cache", 2, 102400, 56320, 1536),
]

# ---------------------------------------------------------------------------
# Competition mode constants.
# ---------------------------------------------------------------------------
COMPETITION_KS = [512, 1024, 2048]
COMPETITION_WS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

# Roofline: the llm_perf performance model for a hypothetical 128-core top-k.
# PROVENANCE: tenstorrent/llm_perf main, PR 671 + 676. ASPIRATIONAL -- no such
# 128-core kernel exists in tt-metal; this row is the model's claim, not a
# measurement, and the gap columns quantify distance to that claim.
# Table: (candidates W, us at K=2048); piecewise-linear interpolation in W,
# then scaled by the K multiplier.
ROOFLINE_CANDIDATES_US = [
    (2048, 0.526),
    (4096, 0.747),
    (8192, 0.968),
    (16384, 1.189),
    (32768, 1.432),
    (65536, 1.839),
    (131072, 2.558),
    (262144, 3.901),
]
ROOFLINE_K_MULT = {512: 0.612, 1024: 0.850, 2048: 1.000}

# Iteration-count rule for competition cells: default 5 measured iterations,
# dropped to 3 when a cell is predicted to exceed 100 ms/iter. The prediction
# is W*k >= 2**24 on the stock single-core layers ONLY: the 2026-08-16
# baseline measured the stock kernel at ~158 ms for W*k = 33.5M (65536x512)
# and ~20 ms at 4.2M, i.e. ~100 ms lands near W*k ~ 21M; 2**24 = 16.8M is the
# conservative power-of-two threshold below that. The op/routed layers are
# multi-core and sit in the tens-of-us range -- never slow-classified.
COMPETITION_SLOW_WK = 1 << 24
COMPETITION_DEFAULT_ITERS = 5
COMPETITION_SLOW_ITERS = 3

# Fixed layer run order. The header-editing layer runs LAST so an abort mid-
# sweep leaves the maximum number of layers measured on the committed header.
#
# seed_index is PINNED per layer (not list position): the 2026-08-16 silicon
# run derived seeds from enumerate() order op=0/routed=1/stocknow=2/
# prebranch=3, and inserting a layer must never silently reseed the others --
# a rerun has to reproduce the recorded cells bit-for-bit.
#
# opstock = ttnn.experimental.topk_large_indices AS SHIPPED (pre-branch).
# rows=1 now auto-selects our column-parallel factory, so the honest proxy is
# num_rows=2 (same W, same k, two independent rows): that selects the
# row-parallel factory whose kernels are byte-identical to pre-branch, both
# rows process concurrently on 2 cores, so the op's device time equals the
# single-row single-core (as-shipped) time. Child reuses the plain
# topk_large_indices harness with batch=2; correctness gathers per row.
#
# (layer, child op, composite parse, arm to check out or None, seed_index)
COMPETITION_LAYERS = [
    ("op", "topk_large_indices", False, None, 0),
    ("opstock", "topk_large_indices", False, None, 4),
    ("routed", "topk_routed", True, None, 1),
    ("stocknow", "topk_stock", True, None, 2),
    ("prebranch", "topk_stock", True, "disable_replay", 3),
]
# Verbatim caveat carried by the .md preamble whenever opstock appears:
OPSTOCK_CAVEAT = (
    "opstock measured via the rows=2 row-parallel proxy: byte-identical "
    "pre-branch kernels, per-row single-core wall time"
)

# Optional fifth layer (--with-blaze): the tt-blaze GLM indexer bench, run as
# an EXTERNAL pytest under our Tracy. Single cell only -- their bench is
# fixed-shape (k=2048, 64K global context), so it maps to the k=2048 W=65536
# table row and nowhere else; fabricating other cells would be inventing data.
# The bench's own printout falls back to host wall-clock on this machine (no
# IOMMU -> no realtime profiler), so it is IGNORED: the Tracy CSV is the datum
# (median DEVICE KERNEL DURATION of the GenericOpDeviceOperation rows;
# validated 24.5 us median of 9 on 2026-08-16).
BLAZE_ROOT = "/home/nachiket/tt-blaze"
BLAZE_TEST = (
    BLAZE_ROOT
    + "/tests/blaze/micro_ops/dsa/test_indexer_sdpa_local_topk.py"
    + "::test_glm52_indexer_sdpa_streaming_local_topk[64k]"
)
# Verbatim caveat carried by the table row and the .md whenever blaze appears:
BLAZE_CAVEAT = (
    "fused SDPA+localTopK program — includes SDPA work the other layers don't; "
    "blaze-vendored kernels, FusedProgram-locked, not a callable op"
)
BLAZE_K, BLAZE_W = 2048, 65536

# ---------------------------------------------------------------------------
# Model-scenario mode constants (--model-scenarios).
# Engines reuse the competition child ops verbatim; seed_index values are the
# SAME pinned indices as COMPETITION_LAYERS so a scenario that coincides with
# a competition cell reproduces the identical input tensor.
#
# SEMANTICS: the `routed` engine is honestly "ttnn.topk largest=True on this
# branch, canonical call form (no indices_tensor / sub_core_grids / stable)".
# For shapes where routing does not engage (e.g. pow2 W in [8192, 65535) with
# k<=64, which stays on the stock multi-core bitonic) it measures that
# bitonic -- exactly what a model calling ttnn.topk gets post-branch with
# zero code change. The per-cell attrs/cores in the result JSON disambiguate
# which factory actually ran. `stocknow` is ttnn.topk largest=False = the
# stock factory on the committed header (the largest flag is only the router;
# pre-branch, both largest values hit the same stock factory).
#
# engine -> (child op, composite, strict, seed_index)
# ---------------------------------------------------------------------------
SCENARIO_ENGINES = {
    "op": ("topk_large_indices", False, True, 0),  # our op, called directly
    "routed": ("topk_routed", True, True, 1),  # ttnn.topk largest=True (this branch)
    "stocknow": ("topk_stock", True, True, 2),  # ttnn.topk largest=False = stock factory
}
SCENARIO_ENGINE_ORDER = ["op", "routed", "stocknow"]  # fixed run order, cheap engines first
# Ledger-measured linear single-core stock rate at small k (k~32: 9.49 ms at
# N=65536 -> ~145 ns/elem; 137 is the conservative fit). SIZING ONLY -- feeds
# the tier-C/D bounding of stock cells, never a reported number. NOTE: the
# rate scales ~linearly with k (k=2048 measures ~9,600 ns/elem); this constant
# is calibrated for the small-k (k<=64) stock cells the built-in scenarios
# actually run. A future large-k stocknow scenario must scale it by k/32.
SCENARIO_STOCK_NS_PER_ELEM = 137
SCENARIO_DEFAULT_ITERS = COMPETITION_DEFAULT_ITERS  # 5
SCENARIO_SLOW_ITERS = COMPETITION_SLOW_ITERS  # 3

# Built-in scenario grid: shapes from real call sites (see the ledger's
# MODEL_SCENARIOS region for the two structural findings these rows carry).
# These are DATA -- override with --scenarios-file, subset with --scenarios.
# `today_engine` = what the model gets pre-branch. `engines` lists are
# deliberately permissive (unsupported => recorded error => em-dash), except:
# the DSA/MSA rows omit stocknow (those models never call ttnn.topk, and the
# k=2048 stock single-core cell would cost ~100 s/iter for a comparison no
# call site makes), and the MoE-gate control rows are STOCK-ONLY by design
# (k=4/k=10 violate the op's k%16==0/k>=16 gate and N=128/512 sit below every
# routing threshold, so routing provably cannot fire -- the rows exist purely
# as no-change proof).
MODEL_SCENARIOS = [
    {
        "name": "sampling_qwen36_tp4",
        "model": "Qwen3 decode sampling, BH p150 TP=4 (vocab 151936 -> 37984/dev -> pad 65536)",
        "callsite": "models/common/sampling/tt_sampling.py:847 (shard: models/demos/blackhole/qwen36/tt/model.py:43-61)",
        "rows": 32,
        "n": 65536,
        "k": 32,
        "dtype": "bf16",
        "engines": ["op", "routed", "stocknow"],
        "today_engine": "stocknow",
        "calls_note": "1x/token/device",
        "notes": "65536 is the one pow2 width that fails the bitonic W<65535 gate -> linear "
        "single-core in production today; prod call passes indices_tensor+sub_core_grids+"
        "stable=True, so the routed column is the CANONICAL form (args dropped), not a free win",
    },
    {
        "name": "sampling_tp8_pow2",
        "model": "Qwen2.5-72B / gpt-oss / MiniMax-M3 decode sampling, TP=8 (pad 32768)",
        "callsite": "models/common/sampling/tt_sampling.py:847 / models/common/modules/sampling/sampling_1d.py:568",
        "rows": 32,
        "n": 32768,
        "k": 32,
        "dtype": "bf16",
        "engines": ["op", "routed", "stocknow"],
        "today_engine": "stocknow",
        "calls_note": "1x/token/device",
        "notes": "today IS the multi-core bitonic (pow2, 8192<=W<65535, k<=64) -- the honest "
        "already-fast row; routing does not engage here, so routed measures the same bitonic",
    },
    {
        "name": "sampling_1chip_split",
        "model": "Llama-3.2-1B/3B single chip, split-vocab sampling (128256/2, deliberately unpadded)",
        "callsite": "models/common/modules/sampling/sampling_1d.py:530 / models/common/sampling/tt_sampling.py:807",
        "rows": 32,
        "n": 64128,
        "k": 32,
        "dtype": "bf16",
        "engines": ["op", "routed", "stocknow"],
        "today_engine": "stocknow",
        "calls_note": "2x/token (vocab halved)",
        "notes": "non-pow2 -> linear single-core today; routed small-k arm engages (non-pow2, >=4096); "
        "canonical-form caveat as above (prod passes indices_tensor/stable)",
    },
    {
        "name": "dsa_indexer_k2048",
        "model": "DeepSeek-V3.2 / GLM-5.x / Kimi-K2.6 DSA indexer top-2048 (Galaxy SP8xTP4, chunk 5120)",
        "callsite": "models/demos/deepseek_v3_d_p/tt/mla/indexer.py:737",
        "rows": 160,
        "n": 65536,
        "k": 2048,
        "dtype": "bf16",
        "engines": ["op", "routed"],
        "today_engine": "op",
        "calls_note": "per sparse-MLA layer per prefill chunk (61 layers DS-v3.2)",
        "notes": "model already calls topk_large_indices by name -- today IS the op; rows=160 "
        "exercises the row-parallel path with the chunk skip live; stocknow omitted (no call "
        "site, ~100 s/iter at k=2048)",
    },
    {
        "name": "dsa_indexer_v4_k512",
        "model": "DeepSeek-V4 DSA indexer top-512 (same geometry, index_topk=512)",
        "callsite": "models/demos/deepseek_v3_d_p/tt/mla/indexer.py:737 (k: reference/deepseek_v4/configuration_deepseek_v4.py:173)",
        "rows": 160,
        "n": 65536,
        "k": 512,
        "dtype": "bf16",
        "engines": ["op", "routed"],
        "today_engine": "op",
        "calls_note": "per sparse-MLA layer per prefill chunk",
        "notes": "today IS the op (row-parallel, chunk skip live); stocknow omitted (no call site)",
    },
    {
        "name": "msa_blocks_k16",
        "model": "MiniMax-M3 MSA block selection, top-16 of 8192 blocks (1M ctx / block 128)",
        "callsite": "models/demos/minimax_m3/tt/attention/msa.py:147",
        "rows": 1,
        "n": 8192,
        "k": 16,
        "dtype": "bf16",
        "engines": ["op", "routed", "stocknow"],
        "today_engine": "op",
        "calls_note": "57 sparse layers per prefill chunk",
        "notes": "the op's floor-k corner; model already ships the op. routed/stocknow both land "
        "on the stock multi-core bitonic here (pow2 8192, k<=64) -- context columns, not wins",
    },
    {
        "name": "gate_gptoss_k4",
        "model": "gpt-oss MoE expert gate, top-4 of 128 experts [NO-CHANGE CONTROL]",
        "callsite": "models/demos/gpt_oss/tt/topk.py:26",
        "rows": 32,
        "n": 128,
        "k": 4,
        "dtype": "bf16",
        "engines": ["stocknow"],
        "today_engine": "stocknow",
        "calls_note": "per MoE layer per forward (36 layers)",
        "notes": "no-change control: k=4 violates the op's k>=16/k%16 gate and N=128 is below "
        "every routing threshold -- routing provably cannot fire; expected tiny and unchanged. "
        "(decode B=32 actually uses the fused topk_router_gpt kernel)",
    },
    {
        "name": "gate_qwen35_k10",
        "model": "Qwen3.5 MoE gate fallback, top-10 of 512 experts [NO-CHANGE CONTROL]",
        "callsite": "models/common/modules/moe/tt_moe_gate.py:639 (fallback fires for k not in {4,6,8} or N>512)",
        "rows": 32,
        "n": 512,
        "k": 10,
        "dtype": "bf16",
        "engines": ["stocknow"],
        "today_engine": "stocknow",
        "calls_note": "per MoE layer per forward",
        "notes": "no-change control: k=10 violates the op gate, N=512 below every routing "
        "threshold; expected tiny and unchanged",
    },
]


def roofline_us(k, w):
    """Piecewise-linear in W over the candidates table (clamped at the ends),
    scaled by the exact-K multiplier. K outside {512, 1024, 2048} has no
    model -- return None rather than invent one."""
    if k not in ROOFLINE_K_MULT:
        return None
    pts = ROOFLINE_CANDIDATES_US
    if w <= pts[0][0]:
        base = pts[0][1]
    elif w >= pts[-1][0]:
        base = pts[-1][1]
    else:
        base = None
        for (w0, u0), (w1, u1) in zip(pts, pts[1:]):
            if w0 <= w <= w1:
                base = u0 + (u1 - u0) * (w - w0) / (w1 - w0)
                break
    return base * ROOFLINE_K_MULT[k]


def competition_seed(k, w, layer_index):
    """Deterministic per-cell seed: same (k, W, layer) always sees the same
    input tensor, across reruns and across machines."""
    return (k * 1_000_003 + w * 97 + layer_index * 7_919) % (2**31)


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def _adjusted_k(k):
    # ttnn.topk host wrapper rounds k up to a tile multiple (topk.cpp:39-41);
    # the device (and the multi-core k<=64 gate) sees this value.
    return 32 * math.ceil(k / 32)


def topk_single_core_l1_cost(n, k, dtype):
    """Mirror of verify_single_core_cost (topk_utils.cpp:220-262). Approximate
    (raw tile payload sizes, no CB page headers) -- used only for a-priori
    labeling; the baseline arm still attempts the cell to get the real answer."""
    ktiles = math.ceil(_adjusted_k(k) / 32)
    value = TILE_BYTES[dtype]
    compute = value  # bfp8/bfp4 would upcast; bf16/fp32 stay as-is
    uint16_output = (n <= 65535) and dtype != "fp32"
    index = TILE_BYTES["uint16" if uint16_output else "uint32"]
    return 4 * (value + index) + (4 + 2 * ktiles) * (compute + index) + ktiles * (value + index)


def topk_predicted_factory(n, k, dtype):
    ak = _adjusted_k(k)
    if n >= 8192 and n < 65535 and _is_pow2(n) and ak <= 64:
        # find_topk_core_config can still demote (exact-division + core
        # rectangle + L1 fit), so this is a prediction, not a promise.
        return "multi_core(pred)"
    return "single_core(pred)"


def sort_predicted_factory(n):
    w = max(n, 64)
    if not _is_pow2(w):
        w = 1 << (w - 1).bit_length()  # host pads to next pow2 (sort.cpp:110-129)
    wt = w // 32
    if wt <= 64:
        return "single_row_single_core(pred)"
    # Hybrid capacity depends on the live core grid; don't guess the grid here.
    return "hybrid_or_dram_multi_core(pred)"


def build_grid(args):
    """Every cell gets status PLANNED or UNSUPPORTED_APRIORI(reason). A-priori
    unsupported cells are attempted once in the baseline arm anyway."""
    ops = args.ops.split(",")
    ns = [int(x) for x in args.ns.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    dtypes = args.dtypes.split(",")
    cells = []

    op_num_slices = getattr(args, "op_num_slices", None)

    def add(op, batch, n, k, dtype, anchor="", valid_length=None, apriori=None, factory=""):
        kpart = f"k{k}" if k is not None else "knone"
        # Column-parallel override only applies to single-row cells; stamping
        # multi-row anchors would make the op reject them loudly by design.
        num_slices = op_num_slices if (op == "topk_large_indices" and batch == 1) else None
        cid = f"{op}_b{batch}xN{n}_{kpart}_{dtype}" + (f"_{anchor}" if anchor else "")
        if num_slices is not None:
            cid += f"_p{num_slices}"
        cells.append(
            {
                "id": cid,
                "num_slices": num_slices,
                "op": op,
                "batch": batch,
                "n": n,
                "k": k,
                "dtype": dtype,
                "dim": -1,
                "anchor": anchor,
                "valid_length": valid_length,
                "apriori": apriori or "",
                "expected_factory": factory,
            }
        )

    if "topk" in ops:
        for n in ns:
            for k in ks:
                for dt in dtypes:
                    apriori = None
                    if k > n:
                        apriori = f"k={k} > dim size N={n} (topk.cpp:222-226)"
                    elif (
                        topk_predicted_factory(n, k, dt) == "single_core(pred)"
                        and topk_single_core_l1_cost(n, k, dt) >= BH_L1_BYTES
                    ):
                        apriori = (
                            f"predicted single-core L1 overflow: "
                            f"cost {topk_single_core_l1_cost(n, k, dt)} >= {BH_L1_BYTES} "
                            f"(verify_single_core_cost, topk_utils.cpp:220)"
                        )
                    add("topk", 1, n, k, dt, apriori=apriori, factory=topk_predicted_factory(n, k, dt))

    if "sort" in ops:
        for n in ns:
            for dt in dtypes:
                add("sort", 1, n, None, dt, factory=sort_predicted_factory(n))

    if "topk_large_indices" in ops:
        for n in ns:
            for k in ks:
                for dt in dtypes:
                    apriori = None
                    if dt != "bf16":
                        apriori = "input must be BFLOAT16 (topk_large_indices_device_operation.cpp:33)"
                    elif k < 16 or k > 2048 or k % 16 != 0:
                        apriori = f"k must be in [16, 2048] in multiples of 16, got {k} (:24-28)"
                    elif n < k:
                        apriori = f"N={n} < k={k} (:49-50)"
                    add("topk_large_indices", 1, n, k, dt, apriori=apriori, factory="row_major_multi_core")
        for tag, rows, n, vl, k in LARGE_INDICES_ANCHORS:
            add(
                "topk_large_indices",
                rows,
                n,
                k,
                "bf16",
                anchor=tag,
                valid_length=vl,
                factory="row_major_multi_core",
            )

    if "moe_gate" in ops:
        # Fixed 256-expert DeepSeek geometry (8 groups x 32); k in {4,6,8} is the
        # kernel's entire supported range -- these are anchor cells, not a grid axis.
        for k in (4, 6, 8):
            add("moe_gate", 32, 256, k, "bf16", anchor="moe_anchor", factory="single_tile_sharded")

    return cells


# ---------------------------------------------------------------------------
# Child mode: runs INSIDE `python -m tracy -r -v`. One process per cell.
# Philosophy from _topk_sort_bench.py: warm the cache with a correctness-checked
# call, warmup iters, measured iters, sync between phases; every exception is a
# recorded datum. This child additionally checks sort values (the old bench
# never did) and covers topk_large_indices and the MoE gate.
# ---------------------------------------------------------------------------
def run_child(spec_path):
    import torch

    import ttnn

    with open(spec_path) as f:
        spec = json.load(f)
    iters = spec["iters"]
    warmup = spec["warmup"]

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    arch = ttnn.get_arch_name()
    manifest = {"arch": arch, "iters": iters, "cells": [], "child_script_md5": _script_md5()}
    torch.manual_seed(0)

    for cell in spec["cells"]:
        entry = dict(cell)
        entry.update({"status": "", "error": "", "phase": ""})
        # Determinism: competition cells carry a seed derived from
        # (k, W, layer); the classic grid keeps the historical fixed 0.
        torch.manual_seed(cell.get("seed", 0))
        cell_iters = cell.get("iters", iters)
        cell_warmup = cell.get("warmup", warmup)  # additive; absent key == old behavior
        try:
            entry["phase"] = "setup"
            call, correctness = _build_cell_callable(ttnn, torch, device, cell)

            # First call warms the program cache AND is correctness-checked:
            # a table row that silently reports garbage timing is worse than
            # no row. An exception here (validate() fires before dispatch)
            # means UNSUPPORTED; an exception later means FAILED.
            entry["phase"] = "first_call"
            out = call()
            entry.update(correctness(out))
            ttnn.synchronize_device(device)

            # Correctness gates timing: a WRONG cell's duration is a number
            # about the wrong computation and must never enter the table.
            if entry.get("wrong"):
                entry["status"] = "WRONG"
                print(f"SWEEP_WRONG {cell['id']} :: {entry.get('wrong_detail', '')}", flush=True)
                manifest["cells"].append(entry)
                continue

            entry["phase"] = "warmup"
            for _ in range(cell_warmup):
                call()
            ttnn.synchronize_device(device)

            entry["phase"] = "measure"
            for _ in range(cell_iters):
                call()
            ttnn.synchronize_device(device)

            entry["status"] = "RAN"
            print(f"SWEEP_OK   {cell['id']}", flush=True)
        except Exception as e:  # noqa: BLE001 - the message IS the result
            entry["error"] = f"{type(e).__name__}: {e}".split("\n")[0][:400]
            if "HARNESS_BUG" in entry["error"]:
                # Not a property of the op or the shape: the harness itself
                # misfired (e.g. script edited mid-run). FAILED is retryable
                # under --resume; UNSUPPORTED would wrongly stick forever.
                entry["status"] = "FAILED"
            else:
                entry["status"] = "UNSUPPORTED" if entry["phase"] in ("setup", "first_call") else "FAILED"
            print(f"SWEEP_FAIL {cell['id']} [{entry['phase']}] :: {entry['error']}", flush=True)
        manifest["cells"].append(entry)

    ttnn.close_device(device)
    with open(spec["manifest"], "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"SWEEP: manifest -> {spec['manifest']}", flush=True)


def _build_cell_callable(ttnn, torch, device, cell):
    """Return (call, correctness) for one cell. correctness(out) -> dict of
    extra manifest fields; it runs on the first call's output only."""
    op, n, k, batch, dtype = cell["op"], cell["n"], cell["k"], cell["batch"], cell["dtype"]
    torch_dt = torch.bfloat16 if dtype == "bf16" else torch.float32
    ttnn_dt = ttnn.bfloat16 if dtype == "bf16" else ttnn.float32

    if op == "topk":
        t = torch.randn((1, 1, batch, n), dtype=torch_dt)
        x = ttnn.from_torch(
            t, dtype=ttnn_dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        def call():
            return ttnn.topk(x, k=k, dim=-1, largest=True, sorted=True)

        def correctness(out):
            vals = ttnn.to_torch(out[0])[..., :k].float()
            ref = torch.topk(t.float(), k=k, dim=-1).values
            return {"max_abs_err": (vals - ref).abs().max().item()}

        return call, correctness

    if op in ("topk_routed", "topk_stock"):
        # Competition layers. Both are ttnn.topk on a single TILE row; the
        # `largest` flag is the router: largest=True takes the routed
        # composite (untilize + TopkLargeIndices + gather + tilize + ... after
        # the return_values rewire), largest=False keeps the stock single-core
        # factory on the same shape. Correctness gates timing: exact value
        # multiset vs torch.topk (bf16 selection is exact, no arithmetic) AND
        # indices self-consistency (input gathered at the returned indices
        # must reproduce the returned values).
        largest = op == "topk_routed"
        t = torch.randn((1, 1, batch, n), dtype=torch_dt)
        x = ttnn.from_torch(
            t, dtype=ttnn_dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        def call():
            return ttnn.topk(x, k=k, dim=-1, largest=largest, sorted=True)

        def correctness(out):
            vals = ttnn.to_torch(out[0])[..., :k].float()
            idx = ttnn.to_torch(out[1]).to(torch.int64)[..., :k]
            ref = torch.topk(t.float(), k=k, dim=-1, largest=largest).values
            v_sorted = torch.sort(vals, dim=-1, descending=largest).values
            r_sorted = torch.sort(ref, dim=-1, descending=largest).values
            val_err = (v_sorted - r_sorted).abs().max().item()
            gathered = t.float().gather(-1, idx)
            idx_err = (gathered - vals).abs().max().item()
            wrong = val_err > 0 or idx_err > 0
            return {
                "max_abs_err": val_err,
                "wrong": wrong,
                "wrong_detail": f"val_err={val_err:g} idx_selfconsistency_err={idx_err:g}" if wrong else "",
            }

        return call, correctness

    if op == "sort":
        t = torch.randn((1, 1, batch, n), dtype=torch_dt)
        x = ttnn.from_torch(
            t, dtype=ttnn_dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        def call():
            return ttnn.sort(x, dim=-1, descending=True)

        def correctness(out):
            # Output carries the host's pow2 padding; descending puts the -inf
            # pads last, so the first n columns are the real sorted values.
            vals = ttnn.to_torch(out[0]).float()[..., :n]
            ref = torch.sort(t.float(), dim=-1, descending=True).values
            return {"max_abs_err": (vals - ref).abs().max().item()}

        return call, correctness

    if op == "topk_large_indices":
        vl = cell.get("valid_length")
        t = torch.randn((batch, n), dtype=torch.bfloat16)
        x = ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        op_kwargs = {}
        if vl is not None:
            op_kwargs["valid_length"] = vl
        if cell.get("num_slices") is not None:
            # --op-num-slices passthrough (column-parallel core-count override).
            op_kwargs["num_slices"] = cell["num_slices"]

        def call():
            return ttnn.experimental.topk_large_indices(x, k=k, **op_kwargs)

        def correctness(out):
            # The op returns uint32 INDICES; validate them by gathering the
            # input at those indices and comparing (order-insensitively) to
            # torch.topk over the valid_length-masked input.
            idx = ttnn.to_torch(out).to(torch.int64)[..., :k]
            gathered = t.float().gather(-1, idx)
            masked = t.float().clone()
            if vl is not None:
                masked[..., vl:] = float("-inf")
            ref = torch.topk(masked, k=k, dim=-1).values
            g_sorted = torch.sort(gathered, dim=-1, descending=True).values
            err = (g_sorted - ref).abs().max().item()
            res = {"max_abs_err": err}
            if cell.get("strict"):
                # Competition mode: selection must be exact, and a wrong cell
                # never gets timed.
                res["wrong"] = err > 0
                res["wrong_detail"] = f"gathered-vs-torch.topk max_abs_err={err:g}" if err > 0 else ""
            return res

        return call, correctness

    if op == "moe_gate":
        # Fixed DeepSeek geometry mirrored from
        # models/common/tests/modules/moe/test_generalized_moe_gate.py:
        # (batch, 8, 32) logits reshaped to (batch, 16, 16), one 32x32 shard
        # per batch row, transposed bias and arange indices, preallocated
        # outputs. Linear-renorm path (enable_sigmoid=True, output_softmax=False).
        input_shape = (batch, 8, 32)
        reshaped = (batch, 16, 16)
        out_shape = (batch, 1, 16)
        tile32 = ttnn.Tile((32, 32))
        t_in = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1
        t_bias = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1

        grid = device.compute_with_storage_grid_size()
        core_grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        )

        def up(torch_t, dt):
            return ttnn.from_torch(
                torch_t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem, tile=tile32
            )

        x = up(t_in.reshape(reshaped), ttnn.bfloat16)
        bias = up(t_bias.reshape(reshaped).transpose(-2, -1), ttnn.bfloat16)
        t_idx = torch.arange(256, dtype=torch.int32).unsqueeze(0).expand(batch, -1).reshape(reshaped)
        idx = up(t_idx.transpose(-2, -1).to(torch.uint16), ttnn.uint16)
        out_v = up(torch.zeros(out_shape, dtype=torch.bfloat16), ttnn.bfloat16)
        out_i = up(torch.zeros(out_shape, dtype=torch.uint16), ttnn.uint16)

        def call():
            return ttnn.experimental.deepseek.moe.generalized_moe_gate(
                x,
                bias_tensor=bias,
                input_indices_tensor=idx,
                output_tensor=out_v,
                output_indices_tensor=out_i,
                eps=1e-20,
                scaling_factor=2.5,
                enable_sigmoid=True,
                topk=k,
                output_softmax=False,
            )

        def correctness(out):
            # Selection golden: the gate ranks by sigmoid(logit) + bias
            # (DeepSeek noaux). Ties in bf16 make exact index equality too
            # strict for a perf harness, so record the tie-tolerant index
            # match fraction; the full golden lives in
            # test_generalized_moe_gate.py and is not re-derived here.
            dev_idx = ttnn.to_torch(out[1])[:, 0, :k].to(torch.int64)
            sel = torch.sigmoid(t_in.float()) + t_bias.float()
            ref_idx = torch.topk(sel.reshape(batch, 256), k=k, dim=-1).indices
            match = 0
            for b in range(batch):
                match += len(set(dev_idx[b].tolist()) & set(ref_idx[b].tolist()))
            frac = match / (batch * k)
            return {"max_abs_err": 1.0 - frac, "index_match_frac": frac}

        return call, correctness

    # Dispatch fall-through is a HARNESS bug, not an op validation result.
    # Post-mortem (competition run 2026-08-16): cell routed k=2048 W=262144
    # reported "unknown op topk_routed" while every sibling measured -- the
    # child re-imports this script from DISK per cell, and a mid-run commit
    # workflow briefly put a pre-topk_routed version of the file on disk
    # (routed W=131072 measured 11:18:16, this cell failed 11:18:19, the
    # commit carrying the branch landed 11:18:46). Two defenses: the
    # HARNESS_BUG prefix classifies this as FAILED (so --resume RETRIES it
    # instead of treating it as a deterministic UNSUPPORTED), and the child
    # stamps its own script md5 into the manifest so the orchestrator can
    # flag CHILD_SCRIPT_DRIFT explicitly.
    raise RuntimeError(
        f"HARNESS_BUG: unknown op {op} -- the child script on disk lacks this op's "
        "branch (script edited mid-run?); retryable, rerun with --resume"
    )


# ---------------------------------------------------------------------------
# Header arm management (orchestrator side).
# ---------------------------------------------------------------------------
def _header_path():
    return os.path.join(REPO, HEADER_RELPATH)


# Recognize BOTH marker dialects: ours, and the "SWEEP_ARM_BEGIN/END" variant
# found on this tree (inserted by a parallel session). Stripping only our own
# markers would let a foreign block survive into the "baseline" arm and
# silently turn every baseline number into a replay_load number.
_MARKER_BEGINS = (MARKER_BEGIN, "// SWEEP_ARM_BEGIN")
_MARKER_ENDS = (MARKER_END, "// SWEEP_ARM_END")


def _strip_arm_block(text):
    if not any(m in text for m in _MARKER_BEGINS):
        return text, False
    lines = text.splitlines(keepends=True)
    out, skipping, stripped = [], False, False
    for line in lines:
        if any(m in line for m in _MARKER_BEGINS):
            skipping, stripped = True, True
            continue
        if skipping and any(m in line for m in _MARKER_ENDS):
            skipping = False
            continue
        if not skipping:
            out.append(line)
    return "".join(out), stripped


def _verify_baseline_header(text):
    """The committed header contains exactly ONE guarded
    `#define TOPK_REPLAY_STEP_STORE 1` (the default-ON block behind
    `!defined(TOPK_DISABLE_REPLAY_STEP)`), ONE `#define TOPK_REPLAY_STEP_LOAD 1`
    (the STORE->LOAD implication), and NO `#define TOPK_DISABLE_REPLAY_STEP`
    (the token appears only in comments and `!defined(...)` guards). Anything
    else means an edit outside our markers is flipping the replay path -- a
    'baseline' measured on top of that is a lie."""
    loads = text.count("#define TOPK_REPLAY_STEP_LOAD 1")
    stores = text.count("#define TOPK_REPLAY_STEP_STORE 1")
    disables = sum(1 for line in text.splitlines() if line.strip().startswith("#define TOPK_DISABLE_REPLAY_STEP"))
    return loads == 1 and stores == 1 and disables == 0


def checkout_arm(arm, allow_edit, git_baseline_dirty):
    """Flip the header to `arm`. baseline strips the marker block; the other
    arms insert it at the very top of the file. Asserts via git that nothing
    else changed. Refuses to write anything without --allow-header-edit."""
    path = _header_path()
    with open(path) as f:
        text = f.read()
    stripped, had_block = _strip_arm_block(text)

    if not _verify_baseline_header(stripped):
        sys.exit(
            f"REFUSING to run: after stripping the marker block, {HEADER_RELPATH} still "
            "arms TOPK_REPLAY_STEP_* somewhere outside the markers. A 'baseline' on top "
            "of that would be a lie. Diff the header against git and clean it by hand."
        )

    define = ARM_DEFINES[arm]
    if define is None:
        new_text = stripped
    else:
        new_text = f"{MARKER_BEGIN_FULL}\n{define}\n{MARKER_END}\n{stripped}"

    if new_text == text:
        return  # already on this arm; no write, no mtime churn

    if not allow_edit:
        if had_block:
            sys.exit(
                f"REFUSING to run: {HEADER_RELPATH} carries a leftover CANONICAL_SWEEP_ARM "
                "block from a previous sweep, so 'baseline' would not be baseline. "
                "Re-run with --allow-header-edit to let the sweep clean it, or remove it by hand."
            )
        sys.exit(
            f"arm '{arm}' requires editing {HEADER_RELPATH}; pass --allow-header-edit "
            "to permit the (git-verified, auto-restored) edit."
        )

    with open(path, "w") as f:
        f.write(new_text)

    # The only file allowed to differ from the pre-sweep state is the header.
    dirty = set(
        subprocess.run(
            ["git", "-C", REPO, "diff", "--name-only"], capture_output=True, text=True, check=True
        ).stdout.split()
    )
    unexpected = dirty - git_baseline_dirty - {HEADER_RELPATH}
    if unexpected:
        # Restore before dying: never leave the tree in a mystery state.
        with open(path, "w") as f:
            f.write(_strip_arm_block(new_text)[0])
        sys.exit(f"checkout_arm({arm}): unexpected modified files {sorted(unexpected)}; aborting.")
    print(f"ARM: {arm} ({'block removed' if define is None else define})", flush=True)


def clear_kernel_cache():
    # The JIT dephash sidecar (jit_build/build.cpp:563-568) would invalidate
    # stale ELFs on a header change, but between A/B arms we clear the cache
    # outright: a guarantee beats a mechanism the measurement has to trust.
    shutil.rmtree(KERNEL_CACHE_DIR, ignore_errors=True)
    print(f"CACHE: cleared {KERNEL_CACHE_DIR}", flush=True)


# ---------------------------------------------------------------------------
# Orchestrator: one Tracy subprocess per cell per trial.
# ---------------------------------------------------------------------------
def _reports_glob():
    return os.path.join(REPO, "generated/profiler/reports/*/ops_perf_results_*.csv")


def _newest_report_after(t0):
    candidates = [p for p in glob.glob(_reports_glob()) if os.path.getmtime(p) >= t0]
    return max(candidates, key=os.path.getmtime) if candidates else None


def parse_tracy_for_cell(csv_path, cell, iters):
    """Per-config attribution: filter to the cell's device op by OP CODE, order
    by GLOBAL CALL COUNT, take the LAST `iters` rows (drops the correctness
    call, the warmup iters, and any cache-miss compile row). The subprocess ran
    exactly one cell, so this is exact, and the ATTRIBUTES/W/CORE COUNT columns
    are recorded as provenance rather than used as a fragile join key (note:
    ttnn.topk rounds k up to a tile multiple host-side, so ATTRIBUTES shows the
    ADJUSTED k, not the requested one)."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    dur_col = next((c for c in rows[0] if "DEVICE KERNEL DURATION [ns]" == c.strip()), None)
    if dur_col is None:
        dur_col = next((c for c in rows[0] if "DEVICE KERNEL DURATION" in c.upper()), None)
    if dur_col is None:
        return None
    matched = []
    for r in rows:
        name = (r.get("OP CODE") or "").strip()
        raw = (r.get(dur_col) or "").strip()
        if not name or not raw or not _op_code_matches(cell["op"], name):
            continue
        try:
            matched.append(
                {
                    "call": int(float(r.get("GLOBAL CALL COUNT") or 0)),
                    "ns": float(raw),
                    "cores": int(float(r.get("CORE COUNT") or 0)),
                    "attrs": (r.get("ATTRIBUTES") or "")[:160],
                }
            )
        except ValueError:
            continue
    if not matched:
        return None
    matched.sort(key=lambda m: m["call"])
    measured = matched[-iters:] if len(matched) >= iters else matched
    return {
        "ns_median": statistics.median(m["ns"] for m in measured),
        "ns_samples": [m["ns"] for m in measured],
        "cores": max(m["cores"] for m in measured),
        "n_rows_total": len(matched),
        "n_rows_used": len(measured),
        "attrs": measured[-1]["attrs"],
        "csv": csv_path,
    }


def parse_tracy_composite(csv_path, cell, iters):
    """Composite-layer attribution (competition mode). The routed ttnn.topk
    path is a chain (untilize + TopkLargeIndices + gather + 2x tilize + eq +
    where + typecast + slice); the stock path is (FillPad + TopK). One
    iteration's cost is the sum over ALL device ops, anchored on the top-k
    op's row count -- the logic proven in the ad-hoc layers_grid.sh, made
    warmup-aware here: each opcode appears an integer number of times per
    iteration, so per opcode we keep only its last (multiplicity * iters)
    occurrences. That drops the correctness call and the warmup iterations
    exactly, opcode by opcode, with no iteration-boundary bookkeeping; ops
    whose count is below one-per-iteration (JIT-time-only setup ops) are
    excluded from the steady-state figure."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    dur_col = next((c for c in rows[0] if "DEVICE KERNEL DURATION [ns]" == c.strip()), None)
    if dur_col is None:
        dur_col = next((c for c in rows[0] if "DEVICE KERNEL DURATION" in c.upper()), None)
    if dur_col is None:
        return None
    per_op = {}
    for r in rows:
        name = (r.get("OP CODE") or "").strip()
        raw = (r.get(dur_col) or "").strip()
        if not name or not raw:
            continue
        try:
            per_op.setdefault(name, []).append(
                {
                    "call": int(float(r.get("GLOBAL CALL COUNT") or 0)),
                    "ns": float(raw),
                    "cores": int(float(r.get("CORE COUNT") or 0)),
                }
            )
        except ValueError:
            continue
    if not per_op:
        return None

    def _norm(s):
        return s.lower().replace("_", "")

    anchor = next((o for o in per_op if "topklargeindices" in _norm(o)), None)
    if anchor is None:
        anchor = next((o for o in per_op if _norm(o).startswith("topk")), None)
    if anchor is None:
        return None
    anchor_occ = sorted(per_op[anchor], key=lambda m: m["call"])
    n_anchor = len(anchor_occ)
    used_iters = min(iters, n_anchor)

    total_ns = 0.0
    parts = []
    for opcode, occ in sorted(per_op.items()):
        occ = sorted(occ, key=lambda m: m["call"])
        mult = len(occ) // n_anchor  # per-iteration multiplicity; 0 = setup-only
        if mult == 0:
            continue
        kept = occ[-mult * used_iters :]
        op_ns = sum(m["ns"] for m in kept) / used_iters
        total_ns += op_ns
        parts.append(f"{opcode}x{mult}={op_ns:.0f}")
    anchor_meas = anchor_occ[-used_iters:]
    return {
        "ns_median": total_ns,  # composite ns per iteration (the layer's number)
        "ns_samples": [m["ns"] for m in anchor_meas],
        "cores": max(m["cores"] for m in anchor_meas),
        "n_rows_total": sum(len(v) for v in per_op.values()),
        "n_rows_used": used_iters,
        "attrs": f"anchor={anchor} anchor_ns={statistics.median(m['ns'] for m in anchor_meas):.0f} "
        + " ".join(parts)[:200],
        "csv": csv_path,
    }


def result_path(outdir, cell_id, arm, trial):
    return os.path.join(outdir, "results", f"{cell_id}.{arm}.t{trial}.json")


# Provenance stamps: a mid-run `./build_metal.sh` or a dirtied working tree
# invalidates every subsequent number, and nothing in the CSV would show it.
# Stamp each cell record so the corruption is DETECTABLE in the output.
# The .so md5 is cached by mtime (hashing ~100 MB once, not once per cell).
_SO_MD5_CACHE = {}


def provenance_stamp():
    import hashlib

    head = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    diff_stat = subprocess.run(
        ["git", "-C", REPO, "diff", "--stat"], capture_output=True, text=True, check=False
    ).stdout
    tree_md5 = hashlib.md5(diff_stat.encode()).hexdigest()
    so_path = os.path.join(REPO, "ttnn/ttnn/_ttnn.so")
    so_mtime, so_md5 = None, None
    if os.path.exists(so_path):
        so_mtime = os.path.getmtime(so_path)
        if so_mtime not in _SO_MD5_CACHE:
            h = hashlib.md5()
            with open(so_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            _SO_MD5_CACHE[so_mtime] = h.hexdigest()
        so_md5 = _SO_MD5_CACHE[so_mtime]
    return {"head_sha": head, "tree_diff_md5": tree_md5, "so_mtime": so_mtime, "so_md5": so_md5}


def run_cell(cell, arm, trial, args):
    """Fresh `python -m tracy -r -v <this file>` subprocess under a watchdog.
    Writes results/<cell>.<arm>.t<trial>.json and returns its dict."""
    outdir = args.out
    workdir = os.path.join(outdir, "work")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "results"), exist_ok=True)
    tag = f"{cell['id']}.{arm}.t{trial}"
    spec_path = os.path.join(workdir, f"{tag}.spec.json")
    manifest_path = os.path.join(workdir, f"{tag}.manifest.json")
    log_path = os.path.join(workdir, f"{tag}.log")
    effective_iters = cell.get("iters", args.iters)
    with open(spec_path, "w") as f:
        json.dump({"cells": [cell], "iters": effective_iters, "warmup": args.warmup, "manifest": manifest_path}, f)

    env = dict(os.environ)
    env[CHILD_SPEC_ENV] = spec_path
    # DPRINT/Watcher and the device profiler share on-device SRAM and cannot
    # coexist (docs/profiling.md); a leaked env var here poisons every number.
    for var in list(env):
        if var.startswith("TT_METAL_DPRINT") or var.startswith("TT_METAL_WATCHER"):
            env.pop(var)

    result = {
        "cell": cell,
        "arm": arm,
        "trial": trial,
        "status": "",
        "error": "",
        "ns_median": None,
        "cores": None,
        "max_abs_err": None,
        "notes": "",
    }
    result.update(provenance_stamp())
    script_md5_at_launch = _script_md5()
    t0 = time.time()
    try:
        with open(log_path, "w") as log:
            subprocess.run(
                [sys.executable, "-m", "tracy", "-r", "-v", SCRIPT_PATH],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
    except subprocess.TimeoutExpired:
        result["status"] = "FAILED"
        result["error"] = f"watchdog timeout {args.timeout}s -- device may need tt-smi -r before continuing"
        _write_result(outdir, tag, result)
        return result

    if not os.path.exists(manifest_path):
        result["status"] = "FAILED"
        result["error"] = f"child produced no manifest (crash before close_device?); see {log_path}"
        _write_result(outdir, tag, result)
        return result

    with open(manifest_path) as f:
        child_manifest = json.load(f)
    entry = child_manifest["cells"][0]
    # Mid-run-edit tripwire: the child imported the script AFTER we launched
    # it; if its md5 differs from what was on disk at launch, orchestrator and
    # child ran different code and the record says so out loud.
    child_md5 = child_manifest.get("child_script_md5")
    if child_md5 and child_md5 != script_md5_at_launch:
        result["notes"] = (
            f"CHILD_SCRIPT_DRIFT(child={child_md5[:8]}, launch={script_md5_at_launch[:8]}) " + result["notes"]
        ).strip()
    result["max_abs_err"] = entry.get("max_abs_err")
    if "index_match_frac" in entry:
        result["notes"] = (result["notes"] + f" index_match_frac={entry['index_match_frac']:.3f}").strip()
    if entry["status"] != "RAN":
        result["status"] = entry["status"]  # UNSUPPORTED, FAILED, or WRONG -- verbatim
        result["error"] = entry.get("error") or entry.get("wrong_detail", "")
        _write_result(outdir, tag, result)
        return result

    csv_path = _newest_report_after(t0)
    if cell.get("composite"):
        parsed = parse_tracy_composite(csv_path, cell, effective_iters) if csv_path else None
    else:
        parsed = parse_tracy_for_cell(csv_path, cell, effective_iters) if csv_path else None
    if parsed is None:
        result["status"] = "FAILED"
        result["error"] = (
            "cell RAN but no Tracy rows matched -- was the run under `python -m tracy -r`? " f"csv={csv_path}"
        )
    else:
        result["status"] = "MEASURED"
        result.update(
            {
                "ns_median": parsed["ns_median"],
                "cores": parsed["cores"],
                "notes": (result["notes"] + f" attrs={parsed['attrs'][:80]}").strip(),
            }
        )
    _write_result(outdir, tag, result)
    return result


def _write_result(outdir, tag, result):
    path = os.path.join(outdir, "results", f"{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)


# ---------------------------------------------------------------------------
# LLK layer: ingest tt_metal/tt-llk/perf_data/perf_topk_rebuild_xl (the file
# that carries baseline and ours as paired arms in one run). Cyc/vector via
# the same two-point slope the driver documents: slope over rebuild_iter_count
# cancels the ~30-cycle marker pair and every one-time cost inside the zone.
#
# CALIBRATION HONESTY: validated against the on-disk post.csv, the RATIOS here
# reproduce the driver's published speedups to the third decimal (1.069x /
# 1.120x / 1.165x rebuild; 1.978x / 2.192x / 2.331x merge; 1.136x / 1.208x /
# 1.255x step), but the ABSOLUTE cyc/vec differs from the driver's published
# cyc/call by a constant normalization inside the post-processing. Treat the
# speedup column as the measurement and the absolute columns as relative scale.
# ---------------------------------------------------------------------------
LLK_PAIRS = [  # (row name, baseline arm, ours arm)
    ("rebuild", "RbCall", "RbCallFull"),
    ("merge", "XlMerge", "MacroMerge"),
    ("step", "XlStep", "FullStep"),
]


def ingest_llk_rows():
    path = os.path.join(REPO, "tt_metal/tt-llk/perf_data/perf_topk_rebuild_xl/perf_topk_rebuild_xl.post.csv")
    if not os.path.exists(path):
        return [], f"llk: {path} not found -- run the driver first (see --run-llk)"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # mean(MATH_ISOLATE) per (arm, k, iter_count), averaged over the 5 runs.
    acc = {}
    for r in rows:
        try:
            key = (
                r["rebuild_arm"].split(".")[-1],
                int(r["rebuild_k"]),
                int(r["rebuild_iter_count"]),
            )
            acc.setdefault(key, []).append(float(r["mean(MATH_ISOLATE)"]))
        except (KeyError, ValueError):
            continue

    def cyc_per_vec(arm, k):
        pts = sorted((ic, statistics.mean(v)) for (a, kk, ic), v in acc.items() if a == arm and kk == k)
        if len(pts) < 2:
            return None
        (lo_i, lo_m), (hi_i, hi_m) = pts[0], pts[-1]
        slope = (hi_m - lo_m) / (hi_i - lo_i)
        vecs = 2 if arm in ("CtrlLoad", "CtrlSwap") else k // 32
        return slope / vecs

    out, note = [], ""
    ks = sorted({k for (_, k, _) in acc})
    for k in ks:
        # Tripwire from the driver: CtrlSwap must be 2.00x CtrlLoad, or the
        # profiler measured the RISC-V push rate and the whole run is invalid.
        cl, cs = cyc_per_vec("CtrlLoad", k), cyc_per_vec("CtrlSwap", k)
        suspect = ""
        if cl and cs and not (1.9 <= cs / cl <= 2.1):
            suspect = f"TRIPWIRE: CtrlSwap/CtrlLoad = {cs / cl:.3f}, expected 2.00 -- run invalid"
        for name, base_arm, ours_arm in LLK_PAIRS:
            b, o = cyc_per_vec(base_arm, k), cyc_per_vec(ours_arm, k)
            if b is None or o is None:
                continue
            out.append(
                {
                    "layer": "llk",
                    "op": f"llk:topk_xl:{name}",
                    "batch": "",
                    "n": "",
                    "k": k,
                    "dtype": "bf16",
                    "dim": "",
                    "anchor": "MATH_ISOLATE cyc/vec",
                    "expected_factory": "",
                    "cores": 1,
                    "llk_cyc_per_vec_base": round(b, 3),
                    "llk_cyc_per_vec_ours": round(o, 3),
                    "speedup": f"{b / o:.3f}x" if o else "",
                    "status": "MEASURED" if not suspect else "FAILED",
                    "notes": suspect or f"{base_arm} vs {ours_arm}; slope cancels marker overhead",
                }
            )
    return out, note


def run_llk_driver():
    """Run producer+consumer inside ONE flock: a concurrent producer wipes
    /tmp/tt-llk-build between phases (HANDOFF.md). Uses the LLK venv, never
    tt-metal's python_env, and never scripts/run_safe_pytest.sh (SORTING.md D6).
    The serial producer races on artefact dirs -- -n 8 is the fix (D2)."""
    tests_dir = os.path.join(REPO, "tt_metal/tt-llk/tests")
    venv_py = os.path.join(tests_dir, ".venv/bin/python")
    if not os.path.exists(venv_py):
        sys.exit(f"llk venv not found at {venv_py}; run `source setup_external_testing_env.sh` there first")
    inner = (
        f"cd {tests_dir}/python_tests && "
        f"{venv_py} -m pytest perf_topk_rebuild_xl.py --compile-producer -n 8 && "
        f"{venv_py} -m pytest perf_topk_rebuild_xl.py --compile-consumer"
    )
    cmd = ["flock", "/tmp/tt-device.lock", "bash", "-c", inner]
    print(f"LLK: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Aggregation + report.
# ---------------------------------------------------------------------------
def _load_results(outdir):
    results = {}
    for path in glob.glob(os.path.join(outdir, "results", "*.json")):
        with open(path) as f:
            r = json.load(f)
        results.setdefault(r["cell"]["id"], {}).setdefault(r["arm"], []).append(r)
    return results


def _arm_stats(trials):
    measured = [t["ns_median"] for t in trials if t["status"] == "MEASURED" and t["ns_median"]]
    if not measured:
        return None, None, len(measured)
    mean = statistics.mean(measured)
    std = statistics.stdev(measured) if len(measured) > 1 else 0.0
    return mean, std, len(measured)


def _speedup(base_mean, base_std, ours_mean, ours_std):
    if not base_mean or not ours_mean:
        return ""
    pooled = math.sqrt(base_std**2 + ours_std**2)
    if abs(base_mean - ours_mean) <= 2 * pooled:
        return "~1.00 (noise)"
    return f"{base_mean / ours_mean:.3f}x"


def aggregate(cells, results, arms, llk_rows):
    """One row per cell. Status precedence: MEASURED > UNSUPPORTED > FAILED >
    PLANNED (never attempted). Errors ride along verbatim -- half the point."""
    rows = []
    for cell in cells:
        per_arm = results.get(cell["id"], {})
        row = {
            "layer": "ttnn",
            "op": cell["op"],
            "batch": cell["batch"],
            "n": cell["n"],
            "k": cell["k"] if cell["k"] is not None else "-",
            "dtype": cell["dtype"],
            "dim": cell["dim"],
            "anchor": cell["anchor"],
            "expected_factory": cell["expected_factory"],
            "cores": "",
            "llk_cyc_per_vec_base": "",
            "llk_cyc_per_vec_ours": "",
            "max_abs_err": "",
            "status": "PLANNED",
            "notes": "",
        }
        base_mean = base_std = None
        for arm in arms:
            trials = per_arm.get(arm, [])
            mean, std, n_ok = _arm_stats(trials)
            row[f"{arm}_ns_mean"] = round(mean, 1) if mean else ""
            row[f"{arm}_ns_std"] = round(std, 1) if mean else ""
            if arm == "baseline":
                base_mean, base_std = mean, std
            elif mean and base_mean:
                row[f"speedup_{arm}"] = _speedup(base_mean, base_std, mean, std)
            else:
                row[f"speedup_{arm}"] = ""
            for t in trials:
                if t.get("cores"):
                    row["cores"] = t["cores"]
                if t.get("max_abs_err") is not None:
                    row["max_abs_err"] = round(t["max_abs_err"], 6)
                if t["status"] == "MEASURED":
                    row["status"] = "MEASURED"
                elif t["status"] in ("UNSUPPORTED", "FAILED") and row["status"] != "MEASURED":
                    row["status"] = f"{t['status']}({t['error']})"
        if row["status"] == "PLANNED" and cell["apriori"]:
            row["status"] = f"UNSUPPORTED_APRIORI({cell['apriori']})"
        elif cell["apriori"] and row["status"].startswith("MEASURED"):
            row["notes"] = f"PREDICTION MISMATCH: a-priori said unsupported ({cell['apriori']})"
        rows.append(row)

    for lrow in llk_rows:
        full = {**{k: "" for k in rows[0]}, **lrow} if rows else lrow
        rows.append(full)
    return rows


def write_reports(rows, arms, outdir):
    if not rows:
        print("REPORT: nothing to write")
        return
    arm_cols = []
    for arm in arms:
        arm_cols += [f"{arm}_ns_mean", f"{arm}_ns_std"]
        if arm != "baseline":
            arm_cols.append(f"speedup_{arm}")
    columns = (
        ["layer", "op", "batch", "n", "k", "dtype", "dim", "anchor", "expected_factory", "cores"]
        + arm_cols
        + ["speedup", "llk_cyc_per_vec_base", "llk_cyc_per_vec_ours", "max_abs_err", "status", "notes"]
    )
    csv_path = os.path.join(outdir, "canonical_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in columns})

    md_path = os.path.join(outdir, "canonical_sweep.md")
    with open(md_path, "w") as f:
        f.write("# Canonical top-k / sort sweep\n\n")
        f.write(f"Arms: {', '.join(arms)}. ns = DEVICE KERNEL DURATION median-per-trial, ")
        f.write("mean±std across trials. Speedups shown only when |delta| > 2·pooled_std.\n\n")
        for layer, title in (("ttnn", "ttnn layer (op-level, Tracy)"), ("llk", "llk layer (MATH_ISOLATE cyc/vec)")):
            sub = [r for r in rows if r.get("layer") == layer]
            if not sub:
                continue
            f.write(f"## {title}\n\n")
            hdr = [c for c in columns if any(str(r.get(c, "")) != "" for r in sub)]
            f.write("| " + " | ".join(hdr) + " |\n")
            f.write("|" + "|".join("---" for _ in hdr) + "|\n")
            for r in sub:
                cells_out = []
                for c in hdr:
                    v = str(r.get(c, ""))
                    if c == "status" and len(v) > 60:
                        v = v[:57] + "..."
                    cells_out.append(v.replace("|", "/"))
                f.write("| " + " | ".join(cells_out) + " |\n")
            f.write("\n")
        bad = [r for r in rows if r.get("status", "").startswith(("UNSUPPORTED", "FAILED"))]
        if bad:
            f.write("## Unsupported / failed cells (verbatim -- the message is the datum)\n\n")
            for r in bad:
                f.write(f"- `{r.get('op')}` N={r.get('n')} K={r.get('k')} {r.get('dtype')}: {r.get('status')}\n")
    print(f"REPORT: {csv_path}")
    print(f"REPORT: {md_path}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Competition mode: deterministic K x W table across the four measured layers
# plus the roofline model. Folds the ad-hoc scratchpad loops (layers_grid.sh,
# kw_grid.sh) into a rerunnable, provenance-stamped mode of this script.
# ---------------------------------------------------------------------------
def _blaze_preflight():
    """Return an error string if the tt-blaze checkout is not usable, else None.
    The bench imports tt-metal through a symlink INSIDE the blaze tree, so both
    the checkout and the link must exist and the link must point at THIS repo
    (a link to a different tt-metal would silently measure someone else's
    kernels)."""
    if not os.path.isdir(BLAZE_ROOT):
        return f"tt-blaze checkout not found at {BLAZE_ROOT}; clone it there (or drop --with-blaze)"
    link = os.path.join(BLAZE_ROOT, "tt-metal")
    if not os.path.exists(link):
        return f"missing symlink {link}; create it with: ln -s {REPO} {link}"
    if os.path.realpath(link) != os.path.realpath(REPO):
        return (
            f"{link} points at {os.path.realpath(link)}, not this repo ({os.path.realpath(REPO)}); "
            f"fix with: ln -sfn {REPO} {link}"
        )
    return None


def parse_tracy_blaze(csv_path):
    """Blaze harvest: median DEVICE KERNEL DURATION over the
    GenericOpDeviceOperation rows (the FusedProgram dispatches as GenericOp).
    No warmup-drop -- the validated recipe takes the median over all rows,
    which is robust to the one cache-miss row (24.5 us median of 9 measured
    2026-08-16)."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    dur_col = next((c for c in rows[0] if "DEVICE KERNEL DURATION [ns]" == c.strip()), None)
    if dur_col is None:
        dur_col = next((c for c in rows[0] if "DEVICE KERNEL DURATION" in c.upper()), None)
    if dur_col is None:
        return None
    matched = []
    for r in rows:
        name = (r.get("OP CODE") or "").strip()
        raw = (r.get(dur_col) or "").strip()
        if "genericop" not in name.lower().replace("_", "") or not raw:
            continue
        try:
            matched.append({"ns": float(raw), "cores": int(float(r.get("CORE COUNT") or 0))})
        except ValueError:
            continue
    if not matched:
        return None
    return {
        "ns_median": statistics.median(m["ns"] for m in matched),
        "ns_samples": [m["ns"] for m in matched],
        "cores": max(m["cores"] for m in matched),
        "n_rows_total": len(matched),
        "n_rows_used": len(matched),
        "attrs": f"GenericOpDeviceOperation median of {len(matched)}",
        "csv": csv_path,
    }


def run_blaze_cell(cell, args):
    """The blaze layer: launch their fixed-shape bench as an external pytest
    under OUR Tracy and harvest the GenericOp rows. Ignore the bench's own
    printed number -- on this host it falls back to wall-clock (no IOMMU, no
    realtime profiler); the Tracy CSV is the datum."""
    outdir = args.out
    os.makedirs(os.path.join(outdir, "results"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "work"), exist_ok=True)
    tag = f"{cell['id']}.blaze.t0"
    log_path = os.path.join(outdir, "work", f"{tag}.log")

    result = {
        "cell": cell,
        "arm": "blaze",
        "trial": 0,
        "status": "",
        "error": "",
        "ns_median": None,
        "cores": None,
        "max_abs_err": None,
        "notes": BLAZE_CAVEAT,
    }
    result.update(provenance_stamp())

    err = _blaze_preflight()
    if err:
        result["status"] = "FAILED"
        result["error"] = err
        _write_result(outdir, tag, result)
        return result

    env = dict(os.environ)
    for var in list(env):
        if var.startswith("TT_METAL_DPRINT") or var.startswith("TT_METAL_WATCHER"):
            env.pop(var)
    env.update(
        {
            "BLAZE_BENCH_INDEXER_LOCAL_TOPK": "1",
            "TT_METAL_HOME": REPO,
            "TT_METAL_RUNTIME_ROOT": REPO,
            "PYTHONPATH": f"{BLAZE_ROOT}:{REPO}",
        }
    )
    cmd = [
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "-v",
        "--op-support-count",
        "4000",
        "-m",
        "pytest",
        BLAZE_TEST,
        "-x",
        "-s",
        f"--rootdir={BLAZE_ROOT}",
        "-c",
        os.path.join(BLAZE_ROOT, "pyproject.toml"),
    ]
    t0 = time.time()
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(
                cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False
            )
    except subprocess.TimeoutExpired:
        result["status"] = "FAILED"
        result["error"] = f"watchdog timeout {args.timeout}s -- device may need tt-smi -r before continuing"
        _write_result(outdir, tag, result)
        return result

    csv_path = _newest_report_after(t0)
    parsed = parse_tracy_blaze(csv_path) if csv_path else None
    if parsed is None:
        result["status"] = "FAILED"
        result["error"] = (
            f"no GenericOpDeviceOperation rows harvested (pytest rc={proc.returncode}, "
            f"csv={csv_path}); see {log_path}"
        )
    else:
        result["status"] = "MEASURED"
        result["ns_median"] = parsed["ns_median"]
        result["cores"] = parsed["cores"]
        result["notes"] = f"{parsed['attrs']}; {BLAZE_CAVEAT}"
    _write_result(outdir, tag, result)
    return result


def build_competition_cells(args):
    ks = sorted(int(x) for x in (args.ks if args.ks != DEFAULT_KS else ",".join(map(str, COMPETITION_KS))).split(","))
    ws = sorted(int(x) for x in (args.ns if args.ns != DEFAULT_NS else ",".join(map(str, COMPETITION_WS))).split(","))
    base_iters = args.iters if args.iters is not None else COMPETITION_DEFAULT_ITERS
    wanted_layers = args.layers_competition.split(",")
    with_blaze = getattr(args, "with_blaze", False)
    if "blaze" in wanted_layers and not with_blaze:
        sys.exit("layer 'blaze' needs --with-blaze (it runs the external tt-blaze checkout)")
    if with_blaze and "blaze" not in wanted_layers:
        wanted_layers.append("blaze")
    cells = []
    if "blaze" in wanted_layers:
        # ONE cell, by design -- see the BLAZE_* constants block.
        cells.append(
            {
                "id": f"comp_blaze_k{BLAZE_K}_w{BLAZE_W}",
                "op": "blaze_external",
                "layer": "blaze",
                "batch": 1,
                "n": BLAZE_W,
                "k": BLAZE_K,
                "dtype": "bf16",
                "dim": -1,
                "anchor": "competition",
                "valid_length": None,
                "apriori": "",
                "expected_factory": "",
                "composite": False,
                "strict": False,
                "seed": None,  # their bench seeds itself; we do not control it
                "iters": 1,
                "arm": None,
            }
        )
    for layer, child_op, composite, arm, seed_index in COMPETITION_LAYERS:
        if layer not in wanted_layers:
            continue
        for k in ks:
            for w in ws:
                if w < k:
                    continue  # same skip as the ad-hoc grids: k cannot exceed the row
                iters = base_iters
                if layer in ("stocknow", "prebranch") and w * k >= COMPETITION_SLOW_WK:
                    iters = min(base_iters, COMPETITION_SLOW_ITERS)
                # --op-num-slices passthrough: the 'op' LAYER only. opstock
                # shares the topk_large_indices child but is the as-shipped
                # proxy -- a slicing override there would measure something
                # that never shipped. Composite/checkout layers keep their
                # own routing untouched.
                num_slices = getattr(args, "op_num_slices", None) if layer == "op" else None
                cid = f"comp_{layer}_k{k}_w{w}" + (f"_p{num_slices}" if num_slices is not None else "")
                cells.append(
                    {
                        "id": cid,
                        "num_slices": num_slices,
                        "op": child_op,
                        "layer": layer,
                        # opstock: 2 independent rows -> the row-parallel
                        # factory whose kernels are byte-identical to
                        # pre-branch; both rows run concurrently on 2 cores,
                        # so op device time == single-row single-core
                        # (as-shipped) time. rows=1 would auto-select our
                        # column-parallel factory and measure the wrong thing.
                        "batch": 2 if layer == "opstock" else 1,
                        "n": w,
                        "k": k,
                        "dtype": "bf16",
                        "dim": -1,
                        "anchor": "competition",
                        "valid_length": None,
                        "apriori": "",
                        "expected_factory": "",
                        "composite": composite,
                        "strict": True,
                        "seed": competition_seed(k, w, seed_index),
                        "iters": iters,
                        "arm": arm,
                    }
                )
    return cells


def run_competition(args):
    cells = build_competition_cells(args)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "competition_grid.json"), "w") as f:
        json.dump(cells, f, indent=1)

    if args.report:
        write_competition_reports(build_competition_table(cells, args.out), args.out)
        return 0

    git_dirty = set(
        subprocess.run(
            ["git", "-C", REPO, "diff", "--name-only"], capture_output=True, text=True, check=True
        ).stdout.split()
    )
    if HEADER_RELPATH in git_dirty and not args.allow_header_edit:
        sys.exit(f"{HEADER_RELPATH} is already modified; refusing to measure a mystery arm.")

    def _done(cell):
        path = result_path(args.out, cell["id"], cell["layer"], 0)
        if not (args.resume and os.path.exists(path)):
            return False
        with open(path) as f:
            status = json.load(f)["status"]
        # WRONG and UNSUPPORTED are deterministic under a fixed seed; only
        # FAILED (transient hang / timeout) earns a retry.
        return status in ("MEASURED", "UNSUPPORTED", "WRONG")

    header_edited = False
    # FIXED layer order: op, routed, stocknow, then blaze (external, no header
    # involvement), then prebranch last (the only one that edits the header).
    run_order = list(COMPETITION_LAYERS)
    if any(c["layer"] == "blaze" for c in cells):
        # After stocknow, before the header-editing prebranch.
        run_order.insert(4, ("blaze", "blaze_external", False, None, None))
    try:
        for layer, _child_op, _composite, arm, _seed_index in run_order:
            layer_cells = sorted((c for c in cells if c["layer"] == layer), key=lambda c: (c["k"], c["n"]))
            if not layer_cells:
                continue
            if arm is not None:
                if not args.allow_header_edit:
                    print(
                        f"LAYER {layer}: SKIPPED (needs --allow-header-edit to arm {ARM_DEFINES[arm]})",
                        flush=True,
                    )
                    continue
                checkout_arm(arm, True, git_dirty)
                header_edited = True
                clear_kernel_cache()
            for cell in layer_cells:
                if _done(cell):
                    print(f"SKIP (resume) {cell['id']}", flush=True)
                    continue
                if layer == "blaze":
                    r = run_blaze_cell(cell, args)
                else:
                    r = run_cell(cell, cell["layer"], 0, args)
                print(
                    f"{r['status']:<11} {cell['id']} iters={cell['iters']} "
                    f"{r['ns_median'] or ''} {r['error'][:80]}",
                    flush=True,
                )
                # Rewrite the table after every cell: early stop keeps data.
                write_competition_reports(build_competition_table(cells, args.out), args.out)
    finally:
        if header_edited:
            checkout_arm("baseline", True, git_dirty)
            clear_kernel_cache()  # never leave prebranch binaries for the next run

    write_competition_reports(build_competition_table(cells, args.out), args.out)
    return 0


def build_competition_table(cells, outdir):
    """One row per (k, W), layers pivoted into columns, roofline + gap +
    speedup columns. WRONG/FAILED/UNSUPPORTED timings never enter the numeric
    columns; the per-layer status is carried alongside."""
    results = {}
    for cell in cells:
        path = result_path(outdir, cell["id"], cell["layer"], 0)
        if os.path.exists(path):
            with open(path) as f:
                results[(cell["k"], cell["n"], cell["layer"])] = json.load(f)

    provenance_seen = set()
    grid_keys = sorted({(c["k"], c["n"]) for c in cells})
    layer_names = [t[0] for t in COMPETITION_LAYERS]
    if any(c["layer"] == "blaze" for c in cells):
        layer_names.append("blaze")
    # A layer only participates in a row if it HAS a cell there (blaze is a
    # single fixed-shape cell; a "PENDING" on every other row would be noise).
    cell_keys = {(c["k"], c["n"], c["layer"]) for c in cells}
    rows = []
    for k, w in grid_keys:
        row = {"k": k, "W": w}
        statuses = []
        us = {}
        for layer in layer_names:
            if (k, w, layer) not in cell_keys:
                row[f"{layer}_us"], row[f"{layer}_cores"] = "", ""
                continue
            r = results.get((k, w, layer))
            if r is None:
                row[f"{layer}_us"], row[f"{layer}_cores"] = "", ""
                statuses.append(f"{layer}=PENDING")
                continue
            provenance_seen.add((r.get("head_sha"), r.get("tree_diff_md5"), r.get("so_md5")))
            if r["status"] == "MEASURED" and r["ns_median"]:
                us[layer] = r["ns_median"] / 1000.0
                row[f"{layer}_us"] = round(us[layer], 2)
                # opstock is the rows=2 row-parallel proxy: the CSV shows 2
                # cores (one per row), but the number being reported is the
                # PER-ROW single-core time -- display accordingly.
                row[f"{layer}_cores"] = "1/row" if layer == "opstock" else (r.get("cores") or "")
                statuses.append(f"{layer}=MEASURED")
            else:
                row[f"{layer}_us"], row[f"{layer}_cores"] = "", ""
                statuses.append(f"{layer}={r['status']}({r['error'][:60]})" if r["error"] else f"{layer}={r['status']}")
        rl = roofline_us(k, w)
        row["roofline_us"] = round(rl, 3) if rl else ""
        row["gap_op_vs_roofline"] = round(us["op"] / rl, 2) if rl and "op" in us else ""
        row["gap_routed_vs_roofline"] = round(us["routed"] / rl, 2) if rl and "routed" in us else ""
        row["speedup_prebranch_over_routed"] = (
            f"{us['prebranch'] / us['routed']:.2f}x" if "prebranch" in us and "routed" in us else ""
        )
        row["speedup_prebranch_over_op"] = (
            f"{us['prebranch'] / us['op']:.2f}x" if "prebranch" in us and "op" in us else ""
        )
        # as-shipped op / our op: what the branch bought at the op level.
        row["speedup_opstock_over_op"] = f"{us['opstock'] / us['op']:.2f}x" if "opstock" in us and "op" in us else ""
        row["status"] = " ".join(statuses)
        rows.append(row)

    drift = ""
    if len(provenance_seen) > 1:
        drift = (
            f"PROVENANCE DRIFT: {len(provenance_seen)} distinct (head_sha, tree_diff_md5, so_md5) "
            f"combinations across cells -- the tree or _ttnn.so changed MID-RUN; cross-layer "
            f"comparisons are suspect. Combos: {sorted(provenance_seen)}"
        )
    elif provenance_seen:
        sha, tree, so = next(iter(provenance_seen))
        drift = f"provenance: head={str(sha)[:12]} tree_diff_md5={str(tree)[:8]} so_md5={str(so)[:8]} (uniform)"
    return {"rows": rows, "provenance_note": drift, "layer_names": layer_names}


def write_competition_reports(table, outdir):
    rows, note = table["rows"], table["provenance_note"]
    if not rows:
        return
    # Without --with-blaze, layer_names is exactly the four classic layers, so
    # the CSV/md schema of existing runs is byte-identical.
    layer_names = table.get("layer_names") or [t[0] for t in COMPETITION_LAYERS]
    columns = ["k", "W"]
    for layer in layer_names:
        columns += [f"{layer}_us", f"{layer}_cores"]
    columns += [
        "roofline_us",
        "gap_op_vs_roofline",
        "gap_routed_vs_roofline",
        "speedup_prebranch_over_routed",
        "speedup_prebranch_over_op",
        "speedup_opstock_over_op",
        "status",
    ]
    csv_path = os.path.join(outdir, "competition_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in columns})

    md_path = os.path.join(outdir, "competition_table.md")
    with open(md_path, "w") as f:
        f.write("# Competition table: top-k layers vs the llm_perf roofline\n\n")
        f.write(
            "Layers: op = topk_large_indices alone; opstock = topk_large_indices as "
            "shipped (pre-branch, rows=2 proxy); routed = ttnn.topk largest=True "
            "(composite, sums ALL ops per iteration); stocknow = ttnn.topk largest=False "
            "on the committed header (replay ON); prebranch = same with "
            "TOPK_DISABLE_REPLAY_STEP armed. roofline = llm_perf model "
            "(tenstorrent/llm_perf PR 671+676) -- ASPIRATIONAL, no such kernel exists; "
            "gap columns = measured/roofline.\n\n"
        )
        if "opstock" in layer_names:
            f.write(f"> {OPSTOCK_CAVEAT}\n\n")
        if "blaze" in layer_names:
            f.write(f"> blaze: {BLAZE_CAVEAT}\n\n")
        if note:
            f.write(f"> {note}\n\n")
        # Exec summary: the two anchor rows every discussion keeps returning to.
        anchors = [r for r in rows if (r["k"], r["W"]) in ((512, 65536), (2048, 65536))]
        if anchors:
            f.write("## Executive summary (anchor rows)\n\n")
            for r in anchors:
                bits = [f"k={r['k']} W={r['W']}"]
                for layer in layer_names:
                    if r.get(f"{layer}_us") != "":
                        bits.append(f"{layer}={r[f'{layer}_us']}us/{r.get(f'{layer}_cores', '')}c")
                if r.get("roofline_us") != "":
                    bits.append(f"roofline={r['roofline_us']}us")
                for col in ("gap_op_vs_roofline", "speedup_prebranch_over_op", "speedup_opstock_over_op"):
                    if r.get(col) != "":
                        bits.append(f"{col}={r[col]}")
                f.write("- " + "  ".join(str(b) for b in bits) + "\n")
            f.write("\n")
        f.write("## Full grid\n\n")
        hdr = [c for c in columns if any(str(r.get(c, "")) != "" for r in rows)]
        f.write("| " + " | ".join(hdr) + " |\n")
        f.write("|" + "|".join("---" for _ in hdr) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in hdr) + " |\n")
    print(f"REPORT: {csv_path}")
    print(f"REPORT: {md_path}")


# ---------------------------------------------------------------------------
# Model-scenario mode (--model-scenarios): named LLM call-site shapes through
# the same per-cell subprocess + watchdog + correctness gate as the
# competition, on the COMMITTED header only (no engine edits the header, so
# no --allow-header-edit, no finally-restore). Output: scenarios_table.csv/.md
# and per-cell result JSONs named scen_{scenario}_{engine}.{engine}.t0.json --
# the scen_ prefix keeps the namespace disjoint from comp_* and classic ids.
# ---------------------------------------------------------------------------
_SCENARIO_RESERVED_PREFIXES = ("comp_", "topk_", "sort_", "moe_")


def load_scenario_specs(args):
    if args.scenarios_file:
        with open(args.scenarios_file) as f:
            specs = json.load(f)["scenarios"]
    else:
        specs = [dict(s) for s in MODEL_SCENARIOS]
    if args.scenarios:  # comma-list subset, like --layers-competition
        wanted = set(args.scenarios.split(","))
        missing = wanted - {s["name"] for s in specs}
        if missing:
            sys.exit(f"unknown scenario(s) {sorted(missing)}")
        specs = [s for s in specs if s["name"] in wanted]
    for s in specs:
        if not re.fullmatch(r"[a-z0-9_]+", s["name"]):
            sys.exit(f"scenario name {s['name']!r} must be [a-z0-9_]+ (it becomes a filename/glob key)")
        if s["name"].startswith(_SCENARIO_RESERVED_PREFIXES):
            sys.exit(f"scenario name {s['name']!r} must not begin with {_SCENARIO_RESERVED_PREFIXES}")
        unknown = set(s["engines"]) - set(SCENARIO_ENGINES)
        if unknown:
            sys.exit(
                f"scenario {s['name']!r}: unknown engine(s) {sorted(unknown)}; choose from {SCENARIO_ENGINE_ORDER}"
            )
        if s["today_engine"] not in s["engines"]:
            sys.exit(f"scenario {s['name']!r}: today_engine {s['today_engine']!r} not in its engines list")
    return specs


def _scenario_iters(engine, rows, n, k, base_iters, warmup, timeout):
    """Bounding tiers (extends the competition's stock-cell rule to rows>1).
    Tier A: default iters / given warmup.
    Tier B: stock engine with rows*n*k >= COMPETITION_SLOW_WK -> 3 iters, warmup 1
            (the competition rule with rows folded into the work term).
    Tier C: predicted stock iter time (rows*n*SCENARIO_STOCK_NS_PER_ELEM) puts
            first_call+warmup+iters over timeout/2 -> iters=1, warmup=0
            (single-sample, still a REAL measurement; iters rides in the record).
    Tier D: even ONE predicted iter > timeout -> (0 iters, est ms) -- the
            orchestrator writes the result JSON itself with the linear-model
            estimate in notes; the estimate renders as an estimate, never as a
            measurement.
    Returns (iters, warmup, est_ms_or_None)."""
    if engine != "stocknow":
        return base_iters, warmup, None
    iters, wu = base_iters, warmup
    if rows * n * k >= COMPETITION_SLOW_WK:
        iters, wu = min(base_iters, SCENARIO_SLOW_ITERS), 1
    est_iter_s = rows * n * SCENARIO_STOCK_NS_PER_ELEM * 1e-9
    if est_iter_s > timeout:
        return 0, 0, est_iter_s * 1e3  # tier D: skip, carry the estimate (ms)
    if est_iter_s * (1 + wu + iters) > timeout / 2:
        iters, wu = 1, 0  # tier C
    return iters, wu, None


def build_scenario_cells(specs, args):
    base_iters = args.iters if args.iters is not None else SCENARIO_DEFAULT_ITERS
    cells = []
    for s in specs:
        rows, n, k, dt = s.get("rows", 1), s["n"], s["k"], s.get("dtype", "bf16")
        for engine in SCENARIO_ENGINE_ORDER:
            if engine not in s["engines"]:
                continue
            child_op, composite, strict, seed_index = SCENARIO_ENGINES[engine]
            iters, wu, est_ms = _scenario_iters(engine, rows, n, k, base_iters, args.warmup, args.timeout)
            num_slices = s.get("num_slices") if (engine == "op" and rows == 1) else None
            cells.append(
                {
                    "id": f"scen_{s['name']}_{engine}",
                    "scenario": s["name"],
                    "layer": engine,
                    "op": child_op,
                    "num_slices": num_slices,
                    "batch": rows,
                    "n": n,
                    "k": k,
                    "dtype": dt,
                    "dim": -1,
                    "anchor": "model_scenario",
                    "valid_length": s.get("valid_length"),
                    "apriori": "",
                    "expected_factory": "",
                    "composite": composite,
                    "strict": strict,
                    "seed": competition_seed(k, n, seed_index),
                    "iters": iters,
                    "warmup": wu,
                    "est_ms": est_ms,
                    "arm": None,
                }
            )
    return cells


def run_scenarios(args):
    specs = load_scenario_specs(args)
    cells = build_scenario_cells(specs, args)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "scenario_grid.json"), "w") as f:
        json.dump({"specs": specs, "cells": cells}, f, indent=1)
    if args.report:
        write_scenario_reports(build_scenario_table(specs, cells, args.out), args.out)
        return 0

    def _done(cell):  # same resume semantics as competition + SKIPPED_SLOW
        path = result_path(args.out, cell["id"], cell["layer"], 0)
        if not (args.resume and os.path.exists(path)):
            return False
        with open(path) as f:
            return json.load(f)["status"] in ("MEASURED", "UNSUPPORTED", "WRONG", "SKIPPED_SLOW")

    for engine in SCENARIO_ENGINE_ORDER:  # engine-major, like competition
        for cell in [c for c in cells if c["layer"] == engine]:
            if _done(cell):
                print(f"SKIP (resume) {cell['id']}", flush=True)
                continue
            if cell["iters"] == 0:  # tier D: never dispatch
                r = {
                    "cell": cell,
                    "arm": engine,
                    "trial": 0,
                    "status": "SKIPPED_SLOW",
                    "error": "",
                    "ns_median": None,
                    "cores": None,
                    "max_abs_err": None,
                    "notes": (
                        f"est~{cell['est_ms']:.0f}ms/iter (linear {SCENARIO_STOCK_NS_PER_ELEM}ns/elem "
                        "model, NOT measured) exceeds --timeout"
                    ),
                }
                r.update(provenance_stamp())
                _write_result(args.out, f"{cell['id']}.{engine}.t0", r)
            else:
                r = run_cell(cell, cell["layer"], 0, args)
            print(
                f"{r['status']:<12} {cell['id']} iters={cell['iters']} " f"{r['ns_median'] or ''} {r['error'][:80]}",
                flush=True,
            )
            # Rewrite the table after every cell: early stop keeps data.
            write_scenario_reports(build_scenario_table(specs, cells, args.out), args.out)
    write_scenario_reports(build_scenario_table(specs, cells, args.out), args.out)
    return 0


def build_scenario_table(specs, cells, outdir):
    """One row per scenario, engines pivoted into columns. {e}_us is filled
    ONLY from status == MEASURED; anything else leaves it blank and carries
    the verbatim status+error in {e}_status -- WRONG/FAILED/UNSUPPORTED/
    SKIPPED_SLOW timings never enter numeric columns."""
    results = {}
    for cell in cells:
        path = result_path(outdir, cell["id"], cell["layer"], 0)
        if os.path.exists(path):
            with open(path) as f:
                results[(cell["scenario"], cell["layer"])] = json.load(f)

    provenance_seen = set()
    rows = []
    for s in specs:
        row = {
            "scenario": s["name"],
            "model": s.get("model", ""),
            "callsite": s.get("callsite", ""),
            "rows": s.get("rows", 1),
            "n": s["n"],
            "k": s["k"],
            "dtype": s.get("dtype", "bf16"),
            "today_engine": s["today_engine"],
            "calls_note": s.get("calls_note", ""),
            "notes": s.get("notes", ""),
        }
        us = {}
        for engine in SCENARIO_ENGINE_ORDER:
            row[f"{engine}_us"] = row[f"{engine}_cores"] = row[f"{engine}_iters"] = row[f"{engine}_status"] = ""
            if engine not in s["engines"]:
                continue
            r = results.get((s["name"], engine))
            if r is None:
                row[f"{engine}_status"] = "PENDING"
                continue
            provenance_seen.add((r.get("head_sha"), r.get("tree_diff_md5"), r.get("so_md5")))
            row[f"{engine}_iters"] = r["cell"].get("iters", "")
            if r["status"] == "MEASURED" and r["ns_median"]:
                us[engine] = r["ns_median"] / 1000.0
                row[f"{engine}_us"] = round(us[engine], 2)
                row[f"{engine}_cores"] = r.get("cores") or ""
                row[f"{engine}_status"] = "MEASURED"
            elif r["status"] == "SKIPPED_SLOW":
                est_ms = r["cell"].get("est_ms")
                row[f"{engine}_status"] = f"SKIPPED_SLOW(est~{est_ms:.0f}ms)" if est_ms else "SKIPPED_SLOW"
            else:
                row[f"{engine}_status"] = f"{r['status']}({r['error'][:60]})" if r.get("error") else r["status"]
        today = us.get(s["today_engine"])
        row["today_us"] = round(today, 2) if today is not None else ""
        row["speedup_today_over_routed"] = f"{today / us['routed']:.2f}x" if today and us.get("routed") else ""
        row["speedup_today_over_op"] = f"{today / us['op']:.2f}x" if today and us.get("op") else ""
        rows.append(row)

    drift = ""
    if len(provenance_seen) > 1:
        drift = (
            f"PROVENANCE DRIFT: {len(provenance_seen)} distinct (head_sha, tree_diff_md5, so_md5) "
            f"combinations across cells -- the tree or _ttnn.so changed MID-RUN; cross-engine "
            f"comparisons are suspect. Combos: {sorted(provenance_seen)}"
        )
    elif provenance_seen:
        sha, tree, so = next(iter(provenance_seen))
        drift = f"provenance: head={str(sha)[:12]} tree_diff_md5={str(tree)[:8]} so_md5={str(so)[:8]} (uniform)"
    for row in rows:
        row["provenance"] = "DRIFT -- see .md" if drift.startswith("PROVENANCE DRIFT") else drift
    return {"rows": rows, "provenance_note": drift}


SCENARIO_CSV_COLUMNS = (
    ["scenario", "model", "callsite", "rows", "n", "k", "dtype", "today_engine", "calls_note"]
    + [f"{e}_{c}" for e in SCENARIO_ENGINE_ORDER for c in ("us", "cores", "iters", "status")]
    + ["today_us", "speedup_today_over_routed", "speedup_today_over_op", "notes", "provenance"]
)


def write_scenario_reports(table, outdir):
    rows, note = table["rows"], table["provenance_note"]
    if not rows:
        return
    csv_path = os.path.join(outdir, "scenarios_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCENARIO_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in SCENARIO_CSV_COLUMNS})

    md_path = os.path.join(outdir, "scenarios_table.md")
    with open(md_path, "w") as f:
        f.write("# Model scenarios: real LLM call-site shapes through the canonical pipeline\n\n")
        f.write(
            "Engines: op = topk_large_indices called directly; routed = ttnn.topk largest=True "
            "on this branch, CANONICAL call form (no indices_tensor/sub_core_grids/stable -- "
            "production sampling call sites pass those and stay on stock, topk.cpp:271-279); "
            "stocknow = ttnn.topk largest=False = the stock factory on the committed header "
            "(largest is only the router; pre-branch both largest values hit the same factory). "
            "today = the engine the model gets pre-branch. us columns filled only from MEASURED "
            "cells; SKIPPED_SLOW carries a linear-model estimate that is NOT a measurement.\n\n"
        )
        if note:
            f.write(f"> {note}\n\n")
        hdr = [c for c in SCENARIO_CSV_COLUMNS if any(str(r.get(c, "")) != "" for r in rows)]
        f.write("| " + " | ".join(hdr) + " |\n")
        f.write("|" + "|".join("---" for _ in hdr) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in hdr) + " |\n")
    print(f"REPORT: {csv_path}")
    print(f"REPORT: {md_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", default="ttnn", help="ttnn,llk")
    p.add_argument("--arms", default="baseline", help="baseline,disable_replay (replay is the committed default)")
    p.add_argument("--ops", default="topk,sort,topk_large_indices,moe_gate")
    p.add_argument("--ns", default=DEFAULT_NS, help="N values; in --competition mode this is W")
    p.add_argument("--ks", default=DEFAULT_KS)
    p.add_argument("--dtypes", default=DEFAULT_DTYPES)
    p.add_argument(
        "--iters",
        type=int,
        default=None,
        help="measured iterations per cell (classic default 10; competition default 5, "
        "auto-3 on stock cells with W*k >= 2^24 -- see COMPETITION_SLOW_WK)",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument(
        "--op-num-slices",
        type=int,
        default=None,
        help="pass num_slices=N (column-parallel core-count override) to every op-layer "
        "topk_large_indices cell (normal and competition modes) so a P sweep is scriptable; "
        "the op rejects it loudly on row-parallel shapes / out-of-range values",
    )
    p.add_argument("--timeout", type=int, default=900, help="per-cell watchdog seconds")
    p.add_argument("--out", default=os.path.join(REPO, "generated/canonical_sweep/latest"))
    p.add_argument(
        "--resume", action="store_true", help="skip cells whose result JSON already says MEASURED/UNSUPPORTED/WRONG"
    )
    p.add_argument("--report", action="store_true", help="rebuild CSV/markdown from --out without measuring")
    p.add_argument("--allow-header-edit", action="store_true", help="permit the A/B edit of " + HEADER_RELPATH)
    p.add_argument(
        "--run-llk", action="store_true", help="also RUN the tt-llk perf driver (default: ingest existing perf_data)"
    )
    p.add_argument(
        "--competition",
        action="store_true",
        help="deterministic K x W competition table (layers: op/routed/stocknow/prebranch + roofline)",
    )
    p.add_argument(
        "--layers-competition",
        default="op,opstock,routed,stocknow,prebranch",
        help="competition layers to (re)run; order of execution is always the fixed one",
    )
    p.add_argument(
        "--with-blaze",
        action="store_true",
        help="add the single-cell blaze layer (k=2048 W=65536; needs the tt-blaze checkout at "
        + BLAZE_ROOT
        + " with a tt-metal symlink into this repo)",
    )
    p.add_argument(
        "--model-scenarios",
        action="store_true",
        help="named LLM model-scenario sweep (engines: op/routed/stocknow on the committed "
        "header; no header edits) -> scenarios_table.csv",
    )
    p.add_argument(
        "--scenarios-file",
        default=None,
        help="JSON scenario grid ({'scenarios': [...]}) overriding the built-in MODEL_SCENARIOS",
    )
    p.add_argument(
        "--scenarios",
        default=None,
        help="comma-list of scenario names to (re)run (subset of the grid)",
    )
    p.add_argument(
        "--child", action="store_true", help="internal: run as measurement child (or set " + CHILD_SPEC_ENV + ")"
    )
    args = p.parse_args()

    # Child mode: inside `python -m tracy -r -v`. Env var wins so tracy's
    # shell re-invocation never has to carry our arguments.
    spec_path = os.environ.get(CHILD_SPEC_ENV)
    if args.child or spec_path:
        run_child(spec_path or sys.argv[sys.argv.index("--child") + 1])
        return 0

    if args.competition and args.model_scenarios:
        sys.exit("--competition and --model-scenarios are separate modes; pick one")
    if args.competition:
        return run_competition(args)
    if args.model_scenarios:
        # Shapes and grid come from the scenario specs; the classic-grid axis
        # flags (--ks/--ns/--dtypes/--ops/--layers-competition/--op-num-slices)
        # are ignored in this mode.
        return run_scenarios(args)
    if args.iters is None:
        args.iters = 10  # classic-grid default; competition resolves its own

    arms = args.arms.split(",")
    for arm in arms:
        if arm not in ARM_DEFINES:
            sys.exit(f"unknown arm '{arm}'; choose from {list(ARM_DEFINES)}")
    layers = args.layers.split(",")
    os.makedirs(args.out, exist_ok=True)
    cells = build_grid(args)
    with open(os.path.join(args.out, "grid.json"), "w") as f:
        json.dump(cells, f, indent=1)

    llk_rows = []
    if "llk" in layers:
        if args.run_llk and not args.report:
            run_llk_driver()
        llk_rows, llk_note = ingest_llk_rows()
        if llk_note:
            print(llk_note)

    if args.report:
        rows = aggregate(cells, _load_results(args.out), arms, llk_rows)
        write_reports(rows, arms, args.out)
        return 0

    # Non-baseline arms need the header edit; fail BEFORE any device work.
    if any(a != "baseline" for a in arms) and not args.allow_header_edit:
        sys.exit("arms beyond 'baseline' edit " + HEADER_RELPATH + "; pass --allow-header-edit")

    git_dirty = set(
        subprocess.run(
            ["git", "-C", REPO, "diff", "--name-only"], capture_output=True, text=True, check=True
        ).stdout.split()
    )
    if HEADER_RELPATH in git_dirty and not args.allow_header_edit:
        sys.exit(f"{HEADER_RELPATH} is already modified; refusing to measure a mystery arm.")

    def _done(cell, arm, trial):
        path = result_path(args.out, cell["id"], arm, trial)
        if not (args.resume and os.path.exists(path)):
            return False
        with open(path) as f:
            status = json.load(f)["status"]
        # UNSUPPORTED is deterministic (validate() fires before dispatch);
        # FAILED gets retried -- it may have been a transient hang.
        return status in ("MEASURED", "UNSUPPORTED")

    try:
        if "ttnn" in layers:
            for arm in arms:
                checkout_arm(arm, args.allow_header_edit, git_dirty)
                clear_kernel_cache()
                for cell in cells:
                    # A-priori unsupported cells: attempted ONCE, baseline arm
                    # only -- the real error message is the datum, and running
                    # them per-arm/per-trial would just re-collect it.
                    trials = range(args.trials)
                    if cell["apriori"]:
                        if arm != "baseline":
                            continue
                        trials = range(1)
                    for trial in trials:
                        if _done(cell, arm, trial):
                            print(f"SKIP (resume) {cell['id']}.{arm}.t{trial}", flush=True)
                            continue
                        r = run_cell(cell, arm, trial, args)
                        print(
                            f"{r['status']:<11} {cell['id']}.{arm}.t{trial} "
                            f"{r['ns_median'] or ''} {r['error'][:80]}",
                            flush=True,
                        )
                        # Rewrite the reports after every cell: an early stop
                        # (or a hang two hours in) keeps everything measured.
                        rows = aggregate(cells, _load_results(args.out), arms, llk_rows)
                        write_reports(rows, arms, args.out)
                        # UNSUPPORTED is deterministic; don't burn trials on it.
                        if r["status"] == "UNSUPPORTED" and not cell["apriori"]:
                            break
    finally:
        # Never leave the tree on a non-baseline arm, whatever happened above.
        if args.allow_header_edit:
            checkout_arm("baseline", True, git_dirty)

    rows = aggregate(cells, _load_results(args.out), arms, llk_rows)
    write_reports(rows, arms, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
