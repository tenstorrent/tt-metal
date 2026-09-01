#!/usr/bin/env bash
# Single-layer per-op captures for Tracy / tt-perf-report debug, both parallelisations.
#
#   ./run_single_layer_profile.sh 1rank   # SP=8 x TP=4, single rank, mesh 8x4, torus_xy
#   ./run_single_layer_profile.sh pp4     # PP=4 x (8,1) over REAL D2D fabric, 1 layer per stage
#   ./run_single_layer_profile.sh pp4full # PP=4, all 36 layers -> for the TRANSPORT fraction only
#   ./run_single_layer_profile.sh 1rank_deep  # single rank, 1 layer, chunk 20 of 20 (KV ~97K)
#   ./run_single_layer_profile.sh pp4_deep    # PP=4,   1 layer/stage, chunk 20 of 20 (KV ~97K)
#   ./run_single_layer_profile.sh all
#
# SHALLOW vs DEEP -- read this before picking one.
#   The `1rank`/`pp4` modes profile ONE 5,120 chunk against an EMPTY KV cache: that is chunk 0, the
#   cheapest chunk in a request. At 100K+ context the interesting chunk is the LAST one, which attends
#   to ~97K of history; measured end-to-end the per-chunk time ramps 114 -> 249 ms at 102,400 and
#   120 -> 442 ms at 261,120, so chunk 0 understates the deep layer budget by 2-3.7x.
#   The `_deep` modes therefore run 20 chunks of 5,120 (ISL 102,400) and leave all 20 in the capture,
#   so the per-op table shows the ramp and the LAST chunk is the 100K layer budget.
#   Do NOT instead profile a single-shot ISL=102,400: chunked prefill never runs that shape (tuned
#   matmul configs are keyed by (weight, seq_len)), and the attention is a different computation.
#
# WHY pp4 AND pp4full. Transport happens once per chunk regardless of how many layers a stage owns.
# With 1 layer/stage the D2D hop is ~1/9 of the layers it would really amortise over, so a 1-layer
# capture OVERSTATES the transport share ~9x. Use `pp4` for the per-op LAYER BUDGET and `pp4full` for
# how much of a chunk is transport. Quoting the transport share off `pp4` is the trap here.
#
# HOW the PP capture works: the tracy python wrapper starts its own capture daemon, so wrapping four
# MPI ranks in it makes four daemons fight over ports. Instead each rank runs with the device profiler
# on and its OWN TT_METAL_PROFILER_DIR (set per-rank in the _profile binding), then each rank's tree is
# post-processed separately with `--process-logs-only`.
set -u
# env.sh must be sourced BEFORE anything reads TT_METAL_HOME: under `set -u` an unset variable is a
# hard error, and env.sh is what defines it (plus the model/cache paths) when the caller has not.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/env.sh"
cd "$TT_METAL_HOME" || exit 1
export PREFILL_HF_MODEL="${PREFILL_HF_MODEL:-$MISTRAL4_HF_MODEL}"
export PYTHONUNBUFFERED=1
OUT=${OUT:-${M4_PROFILE_OUT}}
ISL=${ISL:-5120}          # the production window, NOT the test's 1024 default: tuned matmul configs
                          # are keyed by (weight, seq_len), so a 1024 layer budget is a different shape
mkdir -p "$OUT"
MODE=${1:-all}
TOPO=models/demos/common/prefill/runners/topology_configuration
# The [8,1] column->device map is PER-GALAXY, so prefer a binding generated for this host by
# gen_pp4_binding.py. Falling back to the checked-in template on a different galaxy would not error --
# it would build stages that are not columns and quietly report wrong numbers.
pick_binding(){ local b="$1"; local h="${b%.yaml}.$(hostname).yaml"; [ -f "$h" ] && echo "$h" || echo "$b"; }

# --- single rank: the existing single-layer test, driven by the tracy wrapper (one process, so the
# --- wrapper is fine here) --------------------------------------------------------------------
prof_1rank() {
  export TT_MISTRAL4_PREFILL_TTNN_CACHE=${M4_CACHE_8x4}
  export M4_PROFILE_ISL=$ISL M4_PROFILE_LAYERS=1
  echo "=== $(date -Is) 1rank single-layer profile: isl=$ISL ==="
  $PY -m tracy -v -r -p -o "$OUT/1rank" -n m4_1rank_1layer \
      -m pytest models/demos/deepseek_v3_d_p/tests/perf/test_mistral4_profile_single_layer.py -k tp4 -s
  echo "=== $(date -Is) 1rank EXIT=$? ==="
}

# --- chunked-at-depth via the REAL runner, for either rank count. Uses the runner (not the pytest
# --- single-layer test) because only the runner does chunked prefill, which is what a 100K prompt
# --- actually is. Bonus: single-rank and PP=4 then go through the identical harness.
prof_deep() {
  local ranks=$1 layers=$2 name=$3 binding=$4 cache=$5
  local chunks=${DEEP_CHUNKS:-20}
  # Every rank runs under its OWN tracy wrapper (per-rank port + output dir, derived from the MPI
  # rank). Device-profiler-only capture is not sufficient: the report generator requires the Tracy
  # host capture to resolve op names/times, and without it the multi-GB profile_log_device.csv is
  # unusable. tracy sets TT_METAL_PROFILER_DIR itself per process, so the binding need not.
  unset TT_METAL_PROFILER_DIR TT_METAL_DEVICE_PROFILER
  export PROF_ROOT="$OUT/$name" PROF_NAME="$name" PROF_PORT_BASE=${PROF_PORT_BASE:-8600}
  rm -rf "$PROF_ROOT"; mkdir -p "$PROF_ROOT"
  echo "=== $(date -Is) DEEP profile: ranks=$ranks layers=$layers chunks=${chunks}x5120 (ISL $((chunks*5120))) -> $PROF_ROOT ==="
  RUN_TAG="profile_$name" PP_BINDING="$binding" PP_RANKS=$ranks PP_TTNN_CACHE="$cache" \
  PP_TARGET="$S/tracy_rank_wrapper.sh" \
  PP_REQUESTS=1 PP_USERS=1 PP_CHUNKS=$chunks PP_CHUNK_SIZE=5120 \
  PP_MAX_SEQ_LEN=$((chunks*5120+5120)) PP_KV_ONLY_LAST_LAYER=0 \
  PREFILL_USE_TRACE=0 PREFILL_NUM_LAYERS=$layers \
    "$S/run_pp4_model.sh" 2>&1 | tee "$OUT/${name}.driver.log" \
    | grep -E --line-buffered "^\\[driver\\]|tracy-wrapper|CHUNK_START|E2E_CLOCK|Waiting for lock|TT_FATAL|Error"
  echo "--- reports (tracy -r ran inside each rank) ---"
  for d in "$PROF_ROOT"/rank*; do
    [ -d "$d" ] || continue
    f=$(find "$d" -name 'ops_perf_results*.csv' 2>/dev/null | tail -1)
    [ -n "$f" ] && echo "  $(basename $d): $f" || echo "  $(basename $d): NO ops_perf_results csv"
  done
}

# --- PP=4 over real D2D: device profiler per rank, no tracy wrapper --------------------------
prof_pp() {
  local layers=$1 name=$2
  rm -rf "$OUT"/rank{0,1,2,3}
  # ttrun auto-propagates TT_* vars, so the profiler switch reaches every rank; the per-rank OUTPUT
  # dir comes from the binding's env_overrides.
  # MUST be unset: ttrun passes parent TT_* vars through to every rank, and that passthrough can
  # override the binding's per-rank env_overrides -- which would point all four ranks at one tree.
  unset TT_METAL_PROFILER_DIR
  export TT_METAL_DEVICE_PROFILER=1
  echo "=== $(date -Is) PP=4 real-D2D profile: layers=$layers (per stage $((layers/4))) isl=$ISL ==="
  # KV_ONLY_LAST_LAYER=0 is REQUIRED at 1 layer/stage: otherwise rank 3's only layer is kv-only
  # (no MLA Q/SDPA/wo, no MoE) and its capture is meaningless.
  RUN_TAG="profile_$name" PP_BINDING="$(pick_binding "$TOPO/pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y_profile.yaml")" \
  PP_RANKS=4 PP_TTNN_CACHE=${M4_CACHE_8x1} \
  PP_REQUESTS=${PROF_REQS:-3} PP_USERS=1 PP_CHUNKS=1 PP_CHUNK_SIZE=$ISL \
  PP_MAX_SEQ_LEN=$((ISL*2)) PP_KV_ONLY_LAST_LAYER=0 \
  PREFILL_USE_TRACE=0 PREFILL_NUM_LAYERS=$layers \
    "$S/run_pp4_model.sh" 2>&1 | tee "$OUT/${name}.driver.log" | grep -E --line-buffered "^\\[driver\\]|CHUNK_START|E2E_CLOCK|Waiting for lock|TT_FATAL|Error"
  unset TT_METAL_DEVICE_PROFILER
  echo "--- post-processing each rank's tree separately ---"
  for r in 0 1 2 3; do
    [ -d "$OUT/rank$r" ] || { echo "  rank$r: NO profiler output"; continue; }
    $PY -m tracy --process-logs-only -o "$OUT/rank$r" -r >"$OUT/rank$r/postprocess.log" 2>&1 \
      && echo "  rank$r: $(find "$OUT/rank$r" -name 'ops_perf_results*.csv' | tail -1)" \
      || { echo "  rank$r: post-process FAILED"; tail -5 "$OUT/rank$r/postprocess.log"; }
  done
}

PP_PROF_BIND="$(pick_binding "$TOPO/pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y_profile.yaml")"
case "$MODE" in
  1rank)   prof_1rank ;;
  1rank_deep) prof_deep 1 1 1rank_deep \
               "$TOPO/pipeline_prefill_request_1rank.yaml" ${M4_CACHE_8x4} ;;
  pp4_deep)   prof_deep 4 4 pp4_deep "$PP_PROF_BIND" ${M4_CACHE_8x1} ;;
  pp4)     prof_pp 4 1layer_per_stage ;;
  pp4full) prof_pp 36 36layer ;;
  all)     prof_1rank; prof_pp 4 1layer_per_stage; prof_pp 36 36layer ;;
  *) echo "usage: $0 [1rank|pp4|pp4full|all]"; exit 1 ;;
esac

cat <<'HINT'

--- tt-perf-report, copy/paste ---
  # single rank
  tt-perf-report $(find ${M4_PROFILE_OUT}/1rank -name 'ops_perf_results*.csv' | tail -1)
  # each PP stage (rank 0 = layers 0.., rank 3 = the last stage)
  for r in 0 1 2 3; do
    echo "===== stage $r ====="
    tt-perf-report $(find ${M4_PROFILE_OUT}/rank$r -name 'ops_perf_results*.csv' | tail -1)
  done
HINT
