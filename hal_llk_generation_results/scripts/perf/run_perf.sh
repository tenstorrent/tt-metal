#!/bin/bash
# One LLK perf-test run (eltwise binary, Elwadd variants only) after a card reset.
# Usage: run_perf.sh <label> <perf python test file> [ws dir]   (inside a bh-30 board container)
set -uo pipefail
B=/proj_sw/user_dev/ctr-sbratic; P=$B/tools/perf_cycles
LABEL=${1:?label}; PY=${2:?python test}; WS=${3:-$P/ws_8a}
L=${OUTDIR:-$P/out}/$LABEL; mkdir -p $L
source $B/hal-env-ird/.venv/bin/activate; export PATH=$PATH:$HOME/.local/bin
export TMPDIR=$P/tmp/$LABEL PERF_RUN_TAG=$LABEL PERF_KEEP_RUNS=0; mkdir -p $TMPDIR
T=$WS/tt_metal/tt-llk/tests
tt-smi -r > $L/reset.txt 2>&1
start=$(date +%s)
(cd $T && timeout 90m pytest python_tests/$PY -k Elwadd -q > $L/pytest.txt 2>&1); rc=$?
echo "exit $rc, $(( $(date +%s) - start )) s, $(tail -1 $L/pytest.txt)" > $L/status.txt
mkdir -p $L/perf_data; cp -r $T/../perf_data/runs/$LABEL/. $L/perf_data/
rm -rf $TMPDIR
