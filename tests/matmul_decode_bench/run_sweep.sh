#!/usr/bin/env bash
# Sequential tracy sweep for the no-patch chunked-prefill KERNEL table.
# One (stage,proj,cfg) per tracy subprocess. CSV copied into the csv dir; the
# extractor (MMSWEEP_OP + METRIC=KERNEL) parses col-20 KERNEL per MMD_<...>_r<n>
# region; report.py takes the min-of-5 per-forward across the r<n> repeats.
# Self-contained: PYTHONPATH = the fork ONLY (no tt_symbiote). The harness's own sys.path
# bootstrap resolves the vendored _lib package.
set -u
FORK=/home/ttuser/salnahari/tt-metal-matmul_decode
BENCH=$FORK/tests/matmul_decode_bench
PY=$FORK/python_env/bin/python
CSVDIR=$BENCH/results/chunked_sweep_nopatch_csvs
HARNESS=$BENCH/profile_chunked_prefill_nopatch_stages.py
export PYTHONPATH=$FORK
export ARCH_NAME=blackhole MESH_DEVICE=P150 TT_METAL_HOME=$FORK
export N_ITERS=5 N_REPEAT=5
mkdir -p "$CSVDIR/raw"

# valid chunked-T per stage
declare -A VT=( [SigLIP]="32 64" [VLM]="32 96" [DENOISE]="32" )
# projections per stage
declare -A PJ=( [SigLIP]="qkv o fc1 fc2" [VLM]="qkv o gate up down" [DENOISE]="gate up down" )

run_one() {
  local stage=$1 proj=$2 cfg=$3
  local tag="${stage}_${proj}_${cfg}"
  local out="$CSVDIR/raw/${tag}.log"
  echo "==== RUN $tag ===="
  cd "$FORK" || return 1
  rm -f generated/profiler/.logs/ops_perf_results*.csv 2>/dev/null
  timeout 420 env CFG="$cfg" ONLY_STAGE="$stage" ONLY_PROJ="$proj" \
    "$PY" -m tracy -p -r -v --op-support-count 20000 \
    -m "pytest $HARNESS::test_profile -x -s -q" > "$out" 2>&1
  local rc=$?
  # locate the produced ops_perf_results csv (profiler writes under $TT_METAL_HOME=$FORK/generated/profiler/reports/<ts>/)
  local csv
  csv=$(ls -t "$FORK"/generated/profiler/reports/*/ops_perf_results*.csv \
              "$FORK"/generated/profiler/.logs/ops_perf_results*.csv 2>/dev/null | head -1)
  if [ -n "$csv" ]; then
    cp "$csv" "$CSVDIR/${tag}.csv"
    echo "  csv -> ${tag}.csv (rc=$rc)"
  else
    echo "  NO CSV (rc=$rc) -- see ${tag}.log"
  fi
  grep -E "RESULT|PCC=|RESIDENT stable|NOFIT|INVALID|plan:" "$out" | tail -6
}

STAGES=${STAGES:-"SigLIP VLM DENOISE"}
for stage in $STAGES; do
  for proj in ${PJ[$stage]}; do
    for base in native mmd_full mmd_partial; do
      run_one "$stage" "$proj" "$base"          # unchunked T=M
      for t in ${VT[$stage]}; do
        run_one "$stage" "$proj" "${base}_T${t}"  # chunked
      done
    done
  done
done
echo "==== SWEEP DONE ===="
