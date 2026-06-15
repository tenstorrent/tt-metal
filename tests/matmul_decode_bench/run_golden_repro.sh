#!/usr/bin/env bash
# Golden reproduction driver. One (stage,proj) => one foreground tracy subprocess.
# HARD RULE: --op-support-count 20000 on EVERY run. timeout 420s. one device job at a time.
# Self-contained: PYTHONPATH = the fork ONLY (no tt_symbiote). The vendored _lib package
# resolves via each harness's own sys.path bootstrap (it inserts its own dir).
set -u
FORK=/home/ttuser/salnahari/tt-metal-matmul_decode
export TT_METAL_HOME=$FORK
export ARCH_NAME=blackhole MESH_DEVICE=P150 N_ITERS=5
export TT_SYMBIOTE_SIGNPOST_MODE=1 PI05_TRACY_SIGNPOST=1
export PYTHONPATH=$FORK
unset TT_MESH_GRAPH_DESC_PATH TT_METAL_VISIBLE_DEVICES TT_VISIBLE_DEVICES
PY=$FORK/python_env/bin/python
HARNESS_DIR=$FORK/tests/matmul_decode_bench
REPORTS=$FORK/generated/profiler/reports
cd "$HARNESS_DIR"

# args: KIND(nat|mmd) STAGE PROJ TESTSPEC
KIND="$1"; STAGE="$2"; PROJ="$3"; TESTSPEC="$4"
OUT=$HARNESS_DIR/results/golden_repro/$KIND
mkdir -p "$OUT"
tag="${STAGE}_${PROJ}"
echo ">>> RUN $KIND $tag  ($(date +%H:%M:%S))"
ONLY_STAGE="$STAGE" ONLY_PROJ="$PROJ" timeout 420 $PY -m tracy -p -r -v --op-support-count 20000 \
  -m "pytest $TESTSPEC -x -s -q" > "$OUT/log_${tag}.log" 2>&1
rc=$?
newest=$(ls -1dt "$REPORTS"/*/ 2>/dev/null | head -1)
csv=$(ls -1t "$newest"ops_perf_results_*.csv 2>/dev/null | head -1)
if [ -n "${csv:-}" ] && [ -f "$csv" ]; then
  cp "$csv" "$OUT/${KIND}_${tag}.csv"
  echo "    rc=$rc  CSV -> $OUT/${KIND}_${tag}.csv ($(wc -l < "$csv") rows)"
else
  echo "    rc=$rc  NO CSV FOUND in ${newest:-<none>}"
fi
exit $rc
