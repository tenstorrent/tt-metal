#!/bin/bash
# GPT-OSS prefill ZONE PROFILE: per-zone device time for one 8k chunk, one-shot AND attending a cache.
#
# Sibling of minimax_m3/scripts/run_prefill_profile.sh — same venv / logging / results conventions. Each
# run goes through `python3 -m tracy` and the zone-profiling harness
# (models/demos/gpt_oss_d_p/tests/perf/profile_prefill.py). The harness warms up (runtime.compile),
# fills the cache to PROFILE_CACHE tokens un-profiled (draining the device profiler after every layer
# and every chunk), then runs ONE final chunk with zone signposts on and flushes once after it, so the
# chunk's inter-op gaps stay clean.
#
# Each successful capture's ops CSV is MOVED to RESULTS_DIR/<stamp>_gptoss_layers<N>_cache<N>_<dtype>/
# together with a copy of the log, so generated/profiler/ can be wiped between experiments and every
# capture stays re-renderable.
#
# Usage:  LEVEL=1 LAYERS=4 CACHE=24576 ./models/demos/gpt_oss_d_p/scripts/run_prefill_profile.sh
#
# Flags (all optional, all env vars):
#   HF_MODEL        gpt-oss-120b weights dir, with the tilized cache (tensor_cache_bfp8_MeshShape([4, 8]),
#                   incl. the MoE bias sidecars) next to it.  [default /mnt/models/blaze/openai/gpt-oss-120b,
#                   the copy the CI stages use]
#   GOLDEN_DIR      dir of golden traces; SRC_TRACE picks one metadata.json whose token_ids the harness
#                   tiles to the required length.               [default $HF_MODEL/golden/longbook_qa_eng_prefill_5000]
#   RESULTS_DIR     where finished captures are moved.        [default $TT_METAL_HOME/prefill_profile_results]
#   LEVEL=1|2|3     zone detail. 1 = attn vs mlp only (~3 zones/layer), 2 = every block that costs
#                   real time (~15), 3 = everything incl. norms and sub-splits (~25).      [default 2]
#   LAYERS=N        build/run only the first N layers. Layers alternate sliding (even) / full (odd)
#                   attention, so N>=2 covers both classes; 4 gives 2 samples of each.     [default 4]
#   CACHE=N         tokens already cached before the profiled chunk (rounded down to a whole number
#                   of chunks). CACHE=0 profiles the ONE-SHOT (all-gather fallback) path.
#                   Unset runs both 0 (one-shot) and 24576 (chunked ring @ 24k).
#   CHUNK=N         tokens in the profiled chunk (multiple of 256).                    [default 8192]
#   EXPERT_DTYPE=bf4|bf8   MoE routed-expert weight dtype.                             [default bf4]
#   FROM_CACHE=0    load real safetensors weights instead of the tilized TTNN cache. The cache-only
#                   default needs a prior real-weights build (bias sidecars included).  [default 1]
#   NOC_TRACES=1    + tt-npe DRAM/NOC utilization per op (needs tt-npe installed).
#   SKIP_PREFIX=1   skip the cache prefill and attend a zeroed cache. Fast, but MoE routing is then
#                   unrepresentative — bring-up only.
#   READ_IN_CHUNK=1 also drain the device profiler after every layer INSIDE the profiled chunk. Perturbs
#                   the inter-op gaps; compare its totals with a silent run to see whether drains change
#                   the CCL device times.
#   RESET=1         tt-smi -glx_reset before each capture. OFF by default: on exabox galaxies a reset
#                   tears down the torus wraparound links and they do not retrain (the next ring open
#                   fails with "Graph specified in MGD could not fit"), and -glx_reset is an IPMI tray
#                   reset that can wedge the node. Profile on a fresh allocation instead; use RESET=1
#                   only on a galaxy you know retrains.
#
# Then visualize:  python3 models/demos/gpt_oss_d_p/tests/perf/visualize_zones.py <csv printed below>
set -uo pipefail

die () { echo "ERROR: $*" >&2; exit 1; }

# --- config (override via env) ---
# Repo root from this script's own path (…/models/demos/gpt_oss_d_p/scripts/x.sh -> 4 levels up), so
# the script is portable across checkouts and users instead of hard-coding one person's home.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$_SCRIPT_DIR/../../../.." && pwd)}"
# Ring collectives (the default topology) need the cyclic torus route.
export TT_MESH_GRAPH_DESC_PATH="${TT_MESH_GRAPH_DESC_PATH:-$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto}"
export HF_MODEL="${HF_MODEL:-/mnt/models/blaze/openai/gpt-oss-120b}"
export EXPERT_DTYPE="${EXPERT_DTYPE:-bf4}"
export LOGURU_LEVEL=INFO         # suppress python DEBUG logs at the source
export GPTOSS_PROFILE_ZONES=1    # arm the zone markers (utils/profiler_utils.py reads this at import)
# Default to the fast cache-only load; FROM_CACHE=0 forces the real safetensors read (needed once to
# populate the cache + MoE bias sidecars on a fresh HF_MODEL).
export GPT_OSS_WEIGHTS_FROM_CACHE="${FROM_CACHE:-${GPT_OSS_WEIGHTS_FROM_CACHE:-1}}"
# Device-side profiler DRAM buffer, in programs. The default is 1000
# (tt_metal/impl/profiler/profiler_state_manager.cpp) and the profiled chunk alone is ~45 ops x
# num_layers, so the default leaves almost no margin now that we do NOT drain inside the chunk.
# Cost is 48 B per program per RISC: 20000 is ~600 MB/chip.
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="${TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT:-20000}"

GOLDEN="${GOLDEN_DIR:-$HF_MODEL/golden}"
SRC_TRACE="${SRC_TRACE:-$GOLDEN/longbook_qa_eng_prefill_5000/metadata.json}"
HARNESS="models/demos/gpt_oss_d_p/tests/perf/profile_prefill.py"
VISUALIZE="models/demos/gpt_oss_d_p/tests/perf/visualize_zones.py"
CSVS=()
DESTS=()
FAILED=0
LOGDIR="${LOGDIR:-$TT_METAL_HOME/prefill_profile_logs}"
REPORTS="${REPORTS:-$TT_METAL_HOME/generated/profiler/reports}"
RESULTS_DIR="${RESULTS_DIR:-$TT_METAL_HOME/prefill_profile_results}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/gpt_oss_prefill_profile_${EXPERT_DTYPE}_${STAMP}.log"
CHUNK="${CHUNK:-${PROFILE_CHUNK:-8192}}"
export GPTOSS_PROFILE_LEVEL="${LEVEL:-${GPTOSS_PROFILE_LEVEL:-2}}"
case "$GPTOSS_PROFILE_LEVEL" in 1|2|3) ;; *) die "LEVEL must be 1, 2 or 3 (got '$GPTOSS_PROFILE_LEVEL')" ;; esac
# Default to 4 layers (2 sliding + 2 full). A full 36-layer capture is the configuration that
# exhausts host RAM in tracy's post-processing (see README_profiling.md) — a default must not be
# the one setting the docs warn against. Pass LAYERS=36 explicitly if you really mean it.
[ -z "${LAYERS:-}" ] && LAYERS=4
export PROFILE_NUM_LAYERS="$LAYERS"
[ -n "${CACHE:-}" ] && PROFILE_CACHE="$CACHE"
[ -n "${SKIP_PREFIX:-}" ] && export PROFILE_SKIP_PREFIX="$SKIP_PREFIX"
[ -n "${READ_IN_CHUNK:-}" ] && export PROFILE_READ_IN_CHUNK="$READ_IN_CHUNK"
RESET="${RESET:-0}"

# Preflight: fail here with something actionable rather than several minutes into a run.
[ -d "$TT_METAL_HOME" ]          || die "TT_METAL_HOME does not exist: $TT_METAL_HOME"
[ -f "$TT_METAL_HOME/$HARNESS" ] || die "harness not found: $TT_METAL_HOME/$HARNESS (is TT_METAL_HOME right?)"
[ -f "$TT_METAL_HOME/python_env/bin/activate" ] || \
  die "no venv at $TT_METAL_HOME/python_env — run ./create_venv.sh first"
[ -d "$HF_MODEL" ]               || die "HF_MODEL does not exist: $HF_MODEL (set HF_MODEL=<gpt-oss-120b weights dir>)"
[ -f "$SRC_TRACE" ]              || die "source trace not found: $SRC_TRACE (set GOLDEN_DIR or SRC_TRACE to a golden with token_ids)"
# The cache-only load needs the tilized cache for THIS mesh shape next to the weights (or under
# TT_CACHE_PATH), including the MoE bias sidecars; check before spending minutes opening the mesh.
WEIGHT_CACHE="${TT_CACHE_PATH:-$HF_MODEL}/tensor_cache_bfp8_MeshShape([4, 8])"
if [ "$GPT_OSS_WEIGHTS_FROM_CACHE" = "1" ] && [ ! -f "$WEIGHT_CACHE/model.layers.0/mlp/experts_ep/routed_expert_biases.pt" ]; then
  die "no complete tilized cache at $WEIGHT_CACHE — run once with FROM_CACHE=0 to populate it, or set TT_CACHE_PATH"
fi
if [ "$RESET" = "1" ]; then
  command -v tt-smi >/dev/null   || die "RESET=1 but tt-smi is not on PATH"
fi

cd "$TT_METAL_HOME"
# tracy-capture and tracy-csvexport are siblings of the harness, spawned by `python3 -m tracy`, so the
# harness raising its own RLIMIT_NPROC does not cover them. Saving a capture spawns one compression thread
# per core; at the per-user default (512 on the galaxy hosts, counted across every process the user owns)
# that fails with "Resource temporarily unavailable" and the trace is lost AFTER the run. Raise the soft
# limit to the hard limit for everything launched from here.
NPROC_HARD="$(ulimit -Hu)"
ulimit -Su "$NPROC_HARD" 2>/dev/null || echo "WARNING: could not raise RLIMIT_NPROC (soft $(ulimit -Su), hard $NPROC_HARD)"
# shellcheck disable=SC1091
source python_env/bin/activate
export PYTHONPATH="$TT_METAL_HOME"   # after venv activate so model imports resolve
mkdir -p "$LOGDIR"

# DEBUG filter: drops loguru DEBUG lines on top of LOGURU_LEVEL=INFO.
DEBUG_FILTER='\| *DEBUG *\|'

# --check-exit-code: without it `python3 -m tracy` ignores the harness's exit status, writes whatever CSV
# it has and exits 0 — a crashed capture would be reported below as a good one.
TRACY_OPTS=(-v -r -p --check-exit-code)
# Child calls: makes H2D/D2H buffer copies and program-cache misses show up as per-op columns, which
# is the only way to tell "no host<->device movement" from "movement not measured". The report says
# "not measured" when the CSV ends up without these columns.
TRACY_OPTS+=(--child-functions "HWCommandQueue_write_buffer,HWCommandQueue_read_buffer,CompileProgram")
[ "${NOC_TRACES:-0}" = "1" ] && TRACY_OPTS+=(--collect-noc-traces)

run_cfg () {  # $1=label  $2=cache_tokens
  local label="$1" cache="$2"
  # Newest CSV before the run. Discovery below must find something strictly newer, otherwise a failed
  # capture would silently hand back a PREVIOUS run's report as if it were this one.
  local before; before="$(find "$REPORTS" -name 'ops_perf_results_*.csv' -printf '%T@\n' 2>/dev/null | sort -n | tail -1)"
  before="${before:-0}"
  # cache capacity the harness will allocate: prefix chunks + the profiled chunk (informational — the
  # harness derives the same from PROFILE_CHUNK / PROFILE_CACHE and tiles the trace's tokens to it)
  local total=$(( (cache / CHUNK + 1) * CHUNK ))
  {
    echo ""
    echo "############################################################"
    echo "# $label"
    echo "#   chunk=$CHUNK cache=$cache total=$total trace=$SRC_TRACE"
    echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "############################################################"
  } | tee -a "$LOG"
  if [ "$RESET" = "1" ]; then
    # A failed reset leaves the galaxy in whatever state the previous run left it; profiling through
    # that produces numbers nobody can trust, so skip the config instead of pretending.
    if ! tt-smi -glx_reset; then
      echo "# [$label] SKIPPED: tt-smi -glx_reset failed" | tee -a "$LOG"
      FAILED=1
      return 1
    fi
  fi
  env PROFILE_CHUNK="$CHUNK" PROFILE_CACHE="$cache" PREFILL_TRACE_DIR="$(dirname "$SRC_TRACE")" \
    ${PROFILE_NUM_LAYERS:+PROFILE_NUM_LAYERS="$PROFILE_NUM_LAYERS"} \
    ${PROFILE_READ_EVERY:+PROFILE_READ_EVERY="$PROFILE_READ_EVERY"} \
    ${PROFILE_READ_IN_CHUNK:+PROFILE_READ_IN_CHUNK="$PROFILE_READ_IN_CHUNK"} \
    ${PROFILE_SKIP_PREFIX:+PROFILE_SKIP_PREFIX="$PROFILE_SKIP_PREFIX"} \
    python3 -m tracy "${TRACY_OPTS[@]}" "$HARNESS" 2>&1 |
    grep --line-buffered -vE "$DEBUG_FILTER" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "# [$label] exit=$rc" | tee -a "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "# [$label] FAILED (exit $rc) — see $LOG. Not reporting a CSV; any file present is from an" \
         "earlier run." | tee -a "$LOG"
    FAILED=1
    # A crashed harness leaves the driver re-enumerating the chips for a few seconds; a capture started
    # right away dies in topology discovery ("Query mappings failed on device 0: No such device").
    echo "# [$label] pausing 60 s for the devices to recover before the next capture" | tee -a "$LOG"
    sleep 60
    return "$rc"
  fi

  # Report where the CSV landed. Visualization is a separate step on purpose: the capture is the
  # expensive part, and you will want to re-render it more than once.
  local csv; csv="$(find "$REPORTS" -name 'ops_perf_results_*.csv' -newermt "@$before" 2>/dev/null | sort | tail -1)"
  if [ -n "$csv" ]; then
    # Park the CSV under RESULTS_DIR so generated/profiler/ can be wiped between experiments. The log
    # is copied in once the whole run has finished (see the end of the script).
    local dest="$RESULTS_DIR/${STAMP}_gptoss_layers${LAYERS}_cache${cache}_${EXPERT_DTYPE}${PROFILE_READ_IN_CHUNK:+_readinchunk}"
    if mkdir -p "$dest" && mv "$csv" "$dest/"; then
      csv="$dest/$(basename "$csv")"
      DESTS+=("$dest")
    else
      echo "# [$label] FAILED to move $csv into $dest — the capture is still at its original path" | tee -a "$LOG"
      FAILED=1
    fi
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
  echo "GPT-OSS prefill zone profile"
  echo "  HF_MODEL=$HF_MODEL  EXPERT_DTYPE=$EXPERT_DTYPE  CHUNK=$CHUNK  NOC_TRACES=${NOC_TRACES:-0}"
  echo "  LAYERS=$PROFILE_NUM_LAYERS  ZONE LEVEL=$GPTOSS_PROFILE_LEVEL  SKIP_PREFIX=${PROFILE_SKIP_PREFIX:-0}  READ_IN_CHUNK=${PROFILE_READ_IN_CHUNK:-0}  FROM_CACHE=$GPT_OSS_WEIGHTS_FROM_CACHE  RESET=$RESET"
  echo "  SRC_TRACE=$SRC_TRACE  WEIGHT_CACHE=$WEIGHT_CACHE"
  echo "  RESULTS_DIR=$RESULTS_DIR"
  echo "  RLIMIT_NPROC soft=$(ulimit -Su) hard=$NPROC_HARD"
} | tee "$LOG"

if [ -n "${PROFILE_CACHE:-}" ]; then
  run_cfg "8k at ${PROFILE_CACHE}" "$PROFILE_CACHE"
else
  run_cfg "8k one-shot"  0        # single chunk, all-gather fallback path
  run_cfg "8k at 24k"    24576    # 3 prefix chunks + 1 profiled = 32768 capacity, ring cache-read
fi

{
  echo ""
  echo "==================== SUMMARY ===================="
  grep -E "^# |PROFILED CHUNK|wall-clock|expected ring cache-read" "$LOG"
} | tee -a "$LOG"
echo ""
echo "full log: $LOG"
for d in "${DESTS[@]}"; do cp "$LOG" "$d/"; echo "results: $d"; done
echo ""
if [ "$FAILED" -ne 0 ]; then
  echo "=================== FAILED ==================="
  echo "  At least one capture failed — see $LOG"
  exit 1
fi
echo "=================== NEXT STEP ==================="
for c in "${CSVS[@]}"; do echo "  python3 $VISUALIZE $c"; done
