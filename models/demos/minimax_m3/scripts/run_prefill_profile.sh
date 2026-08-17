#!/bin/bash
# MiniMax-M3 prefill ZONE PROFILE: per-zone device time for one 5k chunk attending a 25k / 55k cache.
#
# Sibling of run_prefill_perf.sh. Same venv / trace-synthesis / tt-smi-reset / logging conventions, but
# each run goes through `python3 -m tracy` and the zone-profiling harness
# (models/demos/minimax_m3/tests/perf/profile_prefill.py) instead of the KV-PCC perf harness. The
# harness warms up (runtime.compile), fills the cache to PROFILE_CACHE tokens un-profiled, then runs
# ONE final chunk with zone signposts on; the device profiler is drained per layer BEFORE that chunk
# and flushed once after it, so the chunk's inter-op gaps stay clean.
#
# Output per run: generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv, whose path
# is printed at the end. Rendering is a separate step (visualize_zones.py) so one capture can be
# re-rendered without paying for it again.
#
# Usage:  LEVEL=1 LAYERS=8 CACHE=25600 ./models/demos/minimax_m3/scripts/run_prefill_profile.sh
#
# Flags (all optional, all env vars):
#   LEVEL=1|2|3     zone detail. 1 = attn vs mlp only (~3 zones/layer), 2 = every block that costs
#                   real time (~20), 3 = everything incl. norms and sub-splits (~35).   [default 2]
#   LAYERS=N        build/run only the first N layers. Layers 0-2 are dense and 3+ sparse, so N>=4
#                   covers both classes; 8 gives 5 sparse samples for the per-chip view.  [default 6]
#   LAYER_IDS=a,b   explicit global layer indices instead of the first N. LAYER_IDS=0,3 is the fastest
#                   useful run: one dense + one sparse layer, ~10 min end to end.
#   CACHE=N         tokens already cached before the profiled chunk (rounded down to a
#                   whole number of chunks). Unset runs both 25600 and 56320.
#   CHUNK=N         tokens in the profiled chunk.                                     [default 5120]
#   EXPERT_DTYPE=bf4|bf8   MoE routed-expert weight dtype.                            [default bf4]
#   NOC_TRACES=1    + tt-npe DRAM/NOC utilization per op (needs tt-npe installed).
#   SKIP_PREFIX=1   skip the cache prefill and attend a zeroed cache. Fast, but MoE routing is then
#                   unrepresentative — bring-up only.
#
# Then visualize:  python3 models/demos/minimax_m3/tests/perf/visualize_zones.py <csv printed below>
set -uo pipefail

# --- config (override via env) ---
# Repo root from this script's own path (…/models/demos/minimax_m3/scripts/x.sh -> 4 levels up), so the
# script is portable across checkouts and users instead of hard-coding one person's home.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$_SCRIPT_DIR/../../../.." && pwd)}"
export TT_MESH_GRAPH_DESC_PATH="${TT_MESH_GRAPH_DESC_PATH:-$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto}"
export HF_MODEL="${HF_MODEL:-/mnt/models/MiniMaxAI/MiniMax-M3-ref}"
export EXPERT_DTYPE="${EXPERT_DTYPE:-bf4}"
export LOGURU_LEVEL=INFO       # suppress python DEBUG logs at the source
export M3_PROFILE_ZONES=1      # arm the zone markers (utils/profiler_utils.py reads this at import)
# Device-side profiler DRAM buffer, in programs. The default is 1000
# (tt_metal/impl/profiler/profiler_state_manager.cpp) and the profiled chunk alone is ~72 ops x
# num_layers, so the default leaves almost no margin now that we do NOT drain inside the chunk.
# Cost is 48 B per program per RISC: 20000 is ~600 MB/chip, where tt-train's 100000 would be ~3 GB —
# too much next to M3's weights.
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="${TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT:-20000}"

GOLDEN="${GOLDEN_DIR:-$HF_MODEL/golden}"
HARNESS="models/demos/minimax_m3/tests/perf/profile_prefill.py"
VISUALIZE="models/demos/minimax_m3/tests/perf/visualize_zones.py"
CSVS=()
FAILED=0
WORKDIR="${PERF_WORKDIR:-/tmp/m3_prefill_perf_traces}"
LOGDIR="${LOGDIR:-$TT_METAL_HOME/prefill_profile_logs}"
REPORTS="${REPORTS:-$TT_METAL_HOME/generated/profiler/reports}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/prefill_profile_${EXPERT_DTYPE}_${STAMP}.log"
CHUNK="${CHUNK:-${PROFILE_CHUNK:-5120}}"
export M3_PROFILE_LEVEL="${LEVEL:-${M3_PROFILE_LEVEL:-2}}"
# Default to 6 layers (3 dense + 3 sparse). Bare `./run_prefill_profile.sh` used to build all 60,
# which is the configuration that exhausted host RAM and was OOM-killed — a default must not be the
# one setting the docs warn against. Pass LAYERS=60 explicitly if you really mean it.
[ -z "${LAYERS:-}" ] && [ -z "${LAYER_IDS:-}" ] && LAYERS=6
[ -n "${LAYERS:-}" ] && export PROFILE_NUM_LAYERS="$LAYERS"
[ -n "${LAYER_IDS:-}" ] && export PROFILE_LAYER_IDS="$LAYER_IDS"
[ -n "${CACHE:-}" ] && PROFILE_CACHE="$CACHE"
[ -n "${SKIP_PREFIX:-}" ] && export PROFILE_SKIP_PREFIX="$SKIP_PREFIX"

# Source of the tokens the synthetic traces are tiled from. Defined before the preflight below, which
# checks it exists.
SRC_TRACE="${SRC_TRACE:-$GOLDEN/longbook_qa_eng_prefill_56320_nopad/metadata.json}"

# Preflight: fail here with something actionable rather than several minutes into a run.
die () { echo "ERROR: $*" >&2; exit 1; }
[ -d "$TT_METAL_HOME" ]          || die "TT_METAL_HOME does not exist: $TT_METAL_HOME"
[ -f "$TT_METAL_HOME/$HARNESS" ] || die "harness not found: $TT_METAL_HOME/$HARNESS (is TT_METAL_HOME right?)"
[ -f "$TT_METAL_HOME/python_env/bin/activate" ] || \
  die "no venv at $TT_METAL_HOME/python_env — run ./create_venv.sh first"
[ -d "$HF_MODEL" ]               || die "HF_MODEL does not exist: $HF_MODEL (set HF_MODEL=<weights dir>)"
[ -f "$SRC_TRACE" ]              || die "source trace not found: $SRC_TRACE (set GOLDEN_DIR or SRC_TRACE)"
command -v tt-smi >/dev/null     || die "tt-smi not on PATH — needed to reset the galaxy between runs"

cd "$TT_METAL_HOME"
# shellcheck disable=SC1091
source python_env/bin/activate
export PYTHONPATH="$TT_METAL_HOME"   # after venv activate so model imports resolve
mkdir -p "$WORKDIR" "$LOGDIR"

# --- synthesize a trace with REAL tokens TILED from a long golden (identical to run_prefill_perf.sh:
# the harness reads only metadata.json's token_ids). Real tokens matter here: MoE routing is
# content-dependent, so random ids would flatten the expert load imbalance. ---
make_trace () {  # $1 = n_tokens ; prints the synthesized trace dir on stdout
  local n="$1" dir="$WORKDIR/synthetic_$1"
  mkdir -p "$dir"
  python3 - "$SRC_TRACE" "$dir/metadata.json" "$n" >&2 <<'PY'
import json, sys
src = json.load(open(sys.argv[1]))["token_ids"]
n = int(sys.argv[3])
assert src, "source trace has no tokens"
tok = [src[i % len(src)] for i in range(n)]  # tile/cycle to reach n
json.dump({"token_ids": tok, "n_tokens": n}, open(sys.argv[2], "w"))
print(f"[profile] wrote {n}-token trace ({len(src)} src tokens tiled) -> {sys.argv[2]}")
PY
  echo "$dir"
}

# DEBUG filter: drops loguru DEBUG lines on top of LOGURU_LEVEL=INFO.
DEBUG_FILTER='\| *DEBUG *\|'

TRACY_OPTS=(-v -r -p)
# Child calls: makes H2D/D2H buffer copies and program-cache misses show up as per-op columns, which
# is the only way to tell "no host<->device movement" from "movement not measured".
TRACY_OPTS+=(--child-functions "HWCommandQueue_write_buffer,HWCommandQueue_read_buffer,CompileProgram")
[ "${NOC_TRACES:-0}" = "1" ] && TRACY_OPTS+=(--collect-noc-traces)

run_cfg () {  # $1=label  $2=cache_tokens
  local label="$1" cache="$2"
  # Newest CSV before the run. Discovery below must find something strictly newer, otherwise a failed
  # capture would silently hand back a PREVIOUS run's report as if it were this one.
  local before; before="$(find "$REPORTS" -name 'ops_perf_results_*.csv' -printf '%T@\n' 2>/dev/null | sort -n | tail -1)"
  before="${before:-0}"
  # cache capacity the harness will allocate: prefix chunks + the profiled chunk
  local total=$(( (cache / CHUNK + 1) * CHUNK ))
  local trace; trace="$(make_trace "$total")"
  {
    echo ""
    echo "############################################################"
    echo "# $label"
    echo "#   chunk=$CHUNK cache=$cache total=$total trace=$trace"
    echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "############################################################"
  } | tee -a "$LOG"
  # A failed reset leaves the galaxy in whatever state the previous run left it; profiling through
  # that produces numbers nobody can trust, so skip the config instead of pretending.
  if ! tt-smi -glx_reset; then
    echo "# [$label] SKIPPED: tt-smi -glx_reset failed" | tee -a "$LOG"
    FAILED=1
    return 1
  fi
  env PROFILE_CHUNK="$CHUNK" PROFILE_CACHE="$cache" PREFILL_TRACE_DIR="$trace" \
    ${PROFILE_NUM_LAYERS:+PROFILE_NUM_LAYERS="$PROFILE_NUM_LAYERS"} \
    ${PROFILE_LAYER_IDS:+PROFILE_LAYER_IDS="$PROFILE_LAYER_IDS"} \
    ${PROFILE_READ_EVERY:+PROFILE_READ_EVERY="$PROFILE_READ_EVERY"} \
    ${PROFILE_SKIP_PREFIX:+PROFILE_SKIP_PREFIX="$PROFILE_SKIP_PREFIX"} \
    python3 -m tracy "${TRACY_OPTS[@]}" "$HARNESS" 2>&1 |
    grep --line-buffered -vE "$DEBUG_FILTER" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "# [$label] exit=$rc" | tee -a "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "# [$label] FAILED (exit $rc) — see $LOG. Not reporting a CSV; any file present is from an" \
         "earlier run." | tee -a "$LOG"
    FAILED=1
    return "$rc"
  fi

  # Report where the CSV landed. Visualization is a separate step on purpose: the capture is the
  # expensive part, and you will want to re-render it more than once.
  local csv; csv="$(find "$REPORTS" -name 'ops_perf_results_*.csv' -newermt "@$before" 2>/dev/null | sort | tail -1)"
  if [ -n "$csv" ]; then
    { echo "# [$label] CSV: $csv"
      echo "# [$label] visualize: python3 $VISUALIZE $csv"; } | tee -a "$LOG"
    CSVS+=("$csv")
  else
    echo "# [$label] WARNING: capture exited 0 but produced no new CSV under $REPORTS" | tee -a "$LOG"
    FAILED=1
  fi
}

echo "logging to $LOG"
{
  echo "MiniMax-M3 prefill zone profile"
  echo "  HF_MODEL=$HF_MODEL  EXPERT_DTYPE=$EXPERT_DTYPE  CHUNK=$CHUNK  NOC_TRACES=${NOC_TRACES:-0}"
  echo "  LAYERS=${PROFILE_LAYER_IDS:-${PROFILE_NUM_LAYERS:-all}}  ZONE LEVEL=$M3_PROFILE_LEVEL  SKIP_PREFIX=${PROFILE_SKIP_PREFIX:-0}"
} | tee "$LOG"

if [ -n "${PROFILE_CACHE:-}" ]; then
  run_cfg "5k at ${PROFILE_CACHE}" "$PROFILE_CACHE"
else
  run_cfg "5k at 25k" 25600   # 5 prefix chunks + 1 profiled = 30720 capacity
  run_cfg "5k at 55k" 56320   # 11 prefix chunks + 1 profiled = 61440 capacity
fi

{
  echo ""
  echo "==================== SUMMARY ===================="
  grep -E "^# |PROFILED CHUNK|wall-clock|whole-cache de-shard" "$LOG"
} | tee -a "$LOG"
echo ""
echo "full log: $LOG"
echo ""
if [ "$FAILED" -ne 0 ]; then
  echo "=================== FAILED ==================="
  echo "  At least one capture failed — see $LOG"
  exit 1
fi
echo "=================== NEXT STEP ==================="
for c in "${CSVS[@]}"; do echo "  python3 $VISUALIZE $c"; done
