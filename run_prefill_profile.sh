#!/bin/bash
# MiniMax-M3 prefill ZONE PROFILE: per-zone device time for one 5k chunk attending a 25k / 55k cache.
#
# Sibling of run_prefill_perf.sh. Same venv / trace-synthesis / tt-smi-reset / logging conventions, but
# each run goes through `python3 -m tracy` and the zone-profiling harness
# (models/demos/minimax_m3/tests/perf/profile_prefill.py) instead of the KV-PCC perf harness. The
# harness warms up (runtime.compile), fills the cache to PROFILE_CACHE tokens un-profiled, then runs
# ONE final chunk with zone signposts on and a ttnn.ReadDeviceProfiler after every layer.
#
# Output per run: generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
# The script then rolls each CSV up per zone with tests/perf/parse_zone_perf.py and writes an HTML report.
#
# Usage:  ./run_prefill_profile.sh                     # both 25k and 55k, bf4 experts
#         PROFILE_CACHE=25600 ./run_prefill_profile.sh # one depth only
#         EXPERT_DTYPE=bf8 ./run_prefill_profile.sh
#         NOC_TRACES=1 ./run_prefill_profile.sh        # + tt-npe DRAM/NOC util per op (needs tt-npe)
#         PROFILE_NUM_LAYERS=6 ./run_prefill_profile.sh # fast bring-up: 3 dense + 3 sparse layers
set -uo pipefail

# --- config (override via env) ---
export TT_METAL_HOME="${TT_METAL_HOME:-/home/vmelnykov/tt-metal}"
export TT_MESH_GRAPH_DESC_PATH="${TT_MESH_GRAPH_DESC_PATH:-$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto}"
export HF_MODEL="${HF_MODEL:-/mnt/models/MiniMaxAI/MiniMax-M3-ref}"
export EXPERT_DTYPE="${EXPERT_DTYPE:-bf4}"
export LOGURU_LEVEL=INFO       # suppress python DEBUG logs at the source
export M3_PROFILE_ZONES=1      # arm the zone markers (utils/profiler_utils.py reads this at import)

GOLDEN="${GOLDEN_DIR:-/data/philei/models/minimax-m3-prefill-cache/golden}"
HARNESS="models/demos/minimax_m3/tests/perf/profile_prefill.py"
PARSER="models/demos/minimax_m3/tests/perf/parse_zone_perf.py"
WORKDIR="${PERF_WORKDIR:-/tmp/m3_prefill_perf_traces}"
LOGDIR="${LOGDIR:-$TT_METAL_HOME/prefill_profile_logs}"
REPORTS="${REPORTS:-$TT_METAL_HOME/generated/profiler/reports}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/prefill_profile_${EXPERT_DTYPE}_${STAMP}.log"
CHUNK="${PROFILE_CHUNK:-5120}"

cd "$TT_METAL_HOME"
# shellcheck disable=SC1091
source python_env/bin/activate
export PYTHONPATH="$TT_METAL_HOME"   # after venv activate so model imports resolve
mkdir -p "$WORKDIR" "$LOGDIR"

# --- synthesize a trace with REAL tokens TILED from a long golden (identical to run_prefill_perf.sh:
# the harness reads only metadata.json's token_ids). Real tokens matter here: MoE routing is
# content-dependent, so random ids would flatten the expert load imbalance. ---
SRC_TRACE="${SRC_TRACE:-$GOLDEN/longbook_qa_eng_prefill_56320_nopad/metadata.json}"
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
[ "${NOC_TRACES:-0}" = "1" ] && TRACY_OPTS+=(--collect-noc-traces)

run_cfg () {  # $1=label  $2=cache_tokens
  local label="$1" cache="$2"
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
  tt-smi -glx_reset
  env PROFILE_CHUNK="$CHUNK" PROFILE_CACHE="$cache" PREFILL_TRACE_DIR="$trace" \
    ${PROFILE_NUM_LAYERS:+PROFILE_NUM_LAYERS="$PROFILE_NUM_LAYERS"} \
    ${PROFILE_READ_EVERY:+PROFILE_READ_EVERY="$PROFILE_READ_EVERY"} \
    ${PROFILE_SKIP_PREFIX:+PROFILE_SKIP_PREFIX="$PROFILE_SKIP_PREFIX"} \
    python3 -m tracy "${TRACY_OPTS[@]}" "$HARNESS" 2>&1 |
    grep --line-buffered -vE "$DEBUG_FILTER" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "# [$label] exit=$rc" | tee -a "$LOG"

  # roll the freshest ops CSV up per zone
  local csv; csv="$(find "$REPORTS" -name 'ops_perf_results_*.csv' -newermt '-30 minutes' 2>/dev/null | sort | tail -1)"
  if [ -n "$csv" ]; then
    local out="$LOGDIR/zones_${label// /_}_${STAMP}"
    echo "# [$label] parsing $csv" | tee -a "$LOG"
    python3 "$PARSER" "$csv" --html "${out}.html" --json "${out}.json" 2>&1 | tee -a "$LOG"
    echo "# [$label] report: ${out}.html" | tee -a "$LOG"
  else
    echo "# [$label] WARNING: no ops_perf_results CSV found under $REPORTS" | tee -a "$LOG"
  fi
}

echo "logging to $LOG"
{
  echo "MiniMax-M3 prefill zone profile"
  echo "  HF_MODEL=$HF_MODEL  EXPERT_DTYPE=$EXPERT_DTYPE  CHUNK=$CHUNK  NOC_TRACES=${NOC_TRACES:-0}"
  echo "  PROFILE_NUM_LAYERS=${PROFILE_NUM_LAYERS:-all}  PROFILE_READ_EVERY=${PROFILE_READ_EVERY:-1}"
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
echo "zone reports: $LOGDIR/zones_*_${STAMP}.html"
