#!/usr/bin/env bash
# deep-plan_13 §10.6 measurement driver. One projection => one foreground tracy subprocess.
# Self-contained: PYTHONPATH = the fork ONLY (no tt_symbiote). The harness's own sys.path
# bootstrap resolves the vendored _lib package.
set -u
FORK=/home/ttuser/salnahari/tt-metal-matmul_decode
export TT_METAL_HOME=$FORK
export ARCH_NAME=blackhole MESH_DEVICE=P150 TT_SYMBIOTE_SIGNPOST_MODE=1 PI05_TRACY_SIGNPOST=1
export PYTHONPATH=$FORK
PY=$FORK/python_env/bin/python
unset TT_MESH_GRAPH_DESC_PATH
HARNESS_DIR=$FORK/tests/matmul_decode_bench
OUT=$HARNESS_DIR/results/mmd13_csvs
REPORTS=$FORK/generated/profiler/reports
mkdir -p "$OUT"
cd "$HARNESS_DIR"

run_one() {
  local kind="$1" testfile="$2" testfn="$3" tag="$4"
  echo ">>> RUN $kind $tag  ($(date +%H:%M:%S))"
  $PY -m tracy -p -r -v --op-support-count 20000 \
    -m "pytest $testfile::$testfn -x -s" > "$OUT/log_${kind}_${tag}.log" 2>&1
  local rc=$?
  # newest report dir
  local newest
  newest=$(ls -1dt "$REPORTS"/*/ 2>/dev/null | head -1)
  local csv
  csv=$(ls -1t "$newest"ops_perf_results_*.csv 2>/dev/null | head -1)
  if [ -n "$csv" ] && [ -f "$csv" ]; then
    cp "$csv" "$OUT/${kind}_${tag}.csv"
    echo "    rc=$rc  CSV -> $OUT/${kind}_${tag}.csv ($(wc -l < "$csv") rows)"
  else
    echo "    rc=$rc  NO CSV FOUND in $newest"
  fi
  return 0
}

"$@"
