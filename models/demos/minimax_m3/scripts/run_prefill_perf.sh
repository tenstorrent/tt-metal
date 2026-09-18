#!/bin/bash
# MiniMax-M3 prefill perf sweep. Drives the galaxy KV-PCC harness in perf-only mode
# (PREFILL_SKIP_PCC=1), which reads only a trace's metadata.json — so the long contexts are
# synthesized by tiling real tokens out of a golden trace.
#
# Each run prints two numbers (see galaxy_prefill_kv_pcc.py):
#   WHOLE SEQUENCE  every chunk, cold from an empty cache -> total prefill time for the prompt
#   LAST CHUNK      only the final chunk, timed against the already-filled cache
# The harness reports LAST CHUNK as "<chunk> tok @ <a_last> cache", where a_last = (n_chunks-1)*chunk.
# That is why a "5k @ 25k" measurement needs a 30720-token trace, not 25600: 25600 is five chunks, so
# its last chunk sees only 20480 of cache. The default traces below are sized for the cache depth in
# their label, which also makes them line up with the zone profiler's CACHE= setting.
#
# Usage:
#   ./models/demos/minimax_m3/scripts/run_prefill_perf.sh
#   EXPERT_DTYPE=bf8 PREFILL_TPS_ITERS=10 ./models/demos/minimax_m3/scripts/run_prefill_perf.sh
#
# Env (all optional):
#   TT_METAL_HOME      repo root. Defaults to this script's location, so a normal checkout needs
#                      nothing set. Override only if you are running the script from outside its repo.
#   HF_MODEL           real MiniMax-M3 weights dir     [/mnt/models/MiniMaxAI/MiniMax-M3-ref]
#   GOLDEN_DIR         golden traces to tile tokens from
#   SRC_TRACE          a specific metadata.json to tile from (overrides GOLDEN_DIR)
#   EXPERT_DTYPE       bf4 | bf8                       [bf4]
#   PREFILL_TPS_ITERS  timed repetitions per config    [5]
#   LOGDIR             where to write the run log      [$TT_METAL_HOME/prefill_perf_logs]
#
# Companion: run_prefill_profile.sh in this directory breaks one chunk down per zone.
set -uo pipefail

# Repo root from this script's own path (…/models/demos/minimax_m3/scripts/x.sh -> 4 levels up), so
# the script is portable across checkouts and users instead of hard-coding one person's home.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$_SCRIPT_DIR/../../../.." && pwd)}"

export TT_MESH_GRAPH_DESC_PATH="${TT_MESH_GRAPH_DESC_PATH:-$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto}"
export HF_MODEL="${HF_MODEL:-/mnt/models/MiniMaxAI/MiniMax-M3-ref}"
export EXPERT_DTYPE="${EXPERT_DTYPE:-bf4}"
export PREFILL_TPS_ITERS="${PREFILL_TPS_ITERS:-5}"
export PREFILL_SKIP_PCC=1     # perf only — skip the per-layer golden KV PCC
export LOGURU_LEVEL=INFO      # suppress python DEBUG logs at the source

GOLDEN="${GOLDEN_DIR:-$HF_MODEL/golden}"
HARNESS="models/demos/minimax_m3/tests/galaxy_prefill_kv_pcc.py"
WORKDIR="${PERF_WORKDIR:-/tmp/m3_prefill_perf_traces}"
LOGDIR="${LOGDIR:-$TT_METAL_HOME/prefill_perf_logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/prefill_perf_${EXPERT_DTYPE}_${STAMP}.log"
CHUNK=5120
SRC_TRACE="${SRC_TRACE:-$GOLDEN/longbook_qa_eng_prefill_56320_nopad/metadata.json}"
FAILED=0
START_TS=$SECONDS

# --- preflight: fail here with something actionable, rather than several minutes into a run ---
die () { echo "ERROR: $*" >&2; exit 1; }
[ -d "$TT_METAL_HOME" ]            || die "TT_METAL_HOME does not exist: $TT_METAL_HOME"
[ -f "$TT_METAL_HOME/$HARNESS" ]   || die "harness not found: $TT_METAL_HOME/$HARNESS (is TT_METAL_HOME right?)"
[ -f "$TT_METAL_HOME/python_env/bin/activate" ] || \
  die "no venv at $TT_METAL_HOME/python_env — run ./create_venv.sh first"
[ -d "$HF_MODEL" ]                 || die "HF_MODEL does not exist: $HF_MODEL (set HF_MODEL=<weights dir>)"
[ -f "$SRC_TRACE" ]                || die "source trace not found: $SRC_TRACE (set GOLDEN_DIR or SRC_TRACE)"
command -v tt-smi >/dev/null       || die "tt-smi not on PATH — needed to reset the galaxy between runs"

cd "$TT_METAL_HOME"
# shellcheck disable=SC1091
source python_env/bin/activate
export PYTHONPATH="$TT_METAL_HOME"   # after venv activate so model imports resolve
mkdir -p "$WORKDIR" "$LOGDIR"

# --- synthesize a trace with REAL tokens TILED from a long golden. Perf-only reads only
# metadata.json's token_ids (n_tokens = len), so no golden KV is needed. Tiling (rather than a
# src[:n] slice) means any length is reachable, not just up to the source length.
# Called lazily from run_cfg: building all of them up front cost a minute and ~30 MB of JSON for
# traces most sweeps never use. ---
make_trace () {  # $1 = n_tokens ; prints the trace dir on stdout
  local n="$1" dir="$WORKDIR/synthetic_$1"
  if [ -f "$dir/metadata.json" ]; then echo "$dir"; return; fi   # reuse across runs
  mkdir -p "$dir"
  python3 - "$SRC_TRACE" "$dir/metadata.json" "$n" >&2 <<'PY'
import json, sys
src = json.load(open(sys.argv[1]))["token_ids"]
n = int(sys.argv[3])
assert src, "source trace has no tokens"
tok = [src[i % len(src)] for i in range(n)]  # tile/cycle to reach n
json.dump({"token_ids": tok, "n_tokens": n}, open(sys.argv[2], "w"))
print(f"[perf] wrote {n}-token trace ({len(src)} src tokens tiled) -> {sys.argv[2]}")
PY
  echo "$dir"
}

# DEBUG filter: drops loguru DEBUG lines ("... | DEBUG    | ...") on top of LOGURU_LEVEL=INFO.
DEBUG_FILTER='\| *DEBUG *\|'

run_cfg () {  # $1=label  $2=n_tokens (0 => use $3 as a ready-made trace dir)  [$3=trace dir]
  local label="$1" n_tokens="$2" trace="${3:-}"
  [ "$n_tokens" -gt 0 ] && trace="$(make_trace "$n_tokens")"
  {
    echo ""
    echo "############################################################"
    echo "# $label"
    echo "#   trace=$trace  chunk=$CHUNK  iters=$PREFILL_TPS_ITERS"
    echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "############################################################"
  } | tee -a "$LOG"

  # A failed reset leaves the galaxy in whatever state the previous run left it; measuring through
  # that produces numbers nobody can trust, so skip the config instead of pretending.
  if ! tt-smi -glx_reset; then
    echo "# [$label] SKIPPED: tt-smi -glx_reset failed" | tee -a "$LOG"
    FAILED=1
    return 1
  fi

  env PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE="$CHUNK" PREFILL_TRACE_DIR="$trace" \
    python3 "$HARNESS" 2>&1 | grep --line-buffered -vE "$DEBUG_FILTER" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "# [$label] exit=$rc" | tee -a "$LOG"
  # Without this the sweep continues, prints a SUMMARY and exits 0, so a config that died mid-way is
  # only visible by spotting "exit=1" in a multi-MB log.
  [ "$rc" -ne 0 ] && FAILED=1
  return 0
}

echo "logging to $LOG"
{
  echo "MiniMax-M3 prefill perf sweep"
  echo "  TT_METAL_HOME=$TT_METAL_HOME"
  echo "  HF_MODEL=$HF_MODEL  EXPERT_DTYPE=$EXPERT_DTYPE  PREFILL_TPS_ITERS=$PREFILL_TPS_ITERS"
  echo "  SRC_TRACE=$SRC_TRACE"
} | tee "$LOG"

# ============================ DEFAULT SWEEP ============================
# Trace lengths are chosen so LAST CHUNK lands on the cache depth in the label.

# 1) Cold 5k — one 5120-token chunk from an empty cache (WHOLE == LAST CHUNK)
run_cfg "5k cold (1 x 5120, empty cache)" 0 "$GOLDEN/longbook_5120"

# 2) 5k @ 25k — 6 x 5120 = 30720; LAST CHUNK = 5120 tok @ 25600 cache
run_cfg "5k @ 25k (6 x 5120) — WHOLE 30720 tok; LAST CHUNK 5k @ 25600" 30720

# 3) 5k @ 55k — 12 x 5120 = 61440; LAST CHUNK = 5120 tok @ 56320 cache
run_cfg "5k @ 55k (12 x 5120) — WHOLE 61440 tok; LAST CHUNK 5k @ 56320" 61440

# ======================= LONG CONTEXT (uncomment) =======================
# Sized as round chunk counts; LAST CHUNK sits one chunk below the total.

# 4) 100k — 21 x 5120 = 107520; LAST CHUNK 5k @ 102400
# run_cfg "100k (21 x 5120) — WHOLE 107520 tok; LAST CHUNK 5k @ 102400" 107520

# 5) 256k — 51 x 5120 = 261120; LAST CHUNK 5k @ 256000
# run_cfg "256k (51 x 5120) — WHOLE 261120 tok; LAST CHUNK 5k @ 256000" 261120

# 6) 512k — 101 x 5120 = 517120; LAST CHUNK 5k @ 512000
# run_cfg "512k (101 x 5120) — WHOLE 517120 tok; LAST CHUNK 5k @ 512000" 517120

# 7) 1M — 201 x 5120 = 1029120; LAST CHUNK 5k @ 1024000
# run_cfg "1M (201 x 5120) — WHOLE 1029120 tok; LAST CHUNK 5k @ 1024000" 1029120

# --- summary: pull the headline lines out of the log ---
{
  echo ""
  echo "==================== SUMMARY ===================="
  grep -E "^# |WHOLE SEQUENCE|LAST CHUNK|expert_dtype=" "$LOG"
  echo ""
  printf "total elapsed: %dm %02ds\n" $(( (SECONDS-START_TS)/60 )) $(( (SECONDS-START_TS)%60 ))
} | tee -a "$LOG"
echo ""
echo "full log: $LOG"
if [ "$FAILED" -ne 0 ]; then
  echo ""
  echo "!!! At least one config FAILED or was SKIPPED — grep '# \[' in the log. Numbers above are"
  echo "!!! incomplete."
  exit 1
fi
