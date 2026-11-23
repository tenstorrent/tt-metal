#!/bin/bash
# Original-LLK perf test (eltwise binary, Elwadd only) on the frozen base, after a card reset.
# Usage: run_orig_perf.sh <label>   (run inside a bh-30 board container)
set -uo pipefail
B=/proj_sw/user_dev/ctr-sbratic; P=$B/tools/perf_cycles; L=$P/out/${1:?label}
mkdir -p $L; source $B/hal-env-ird/.venv/bin/activate; export PATH=$PATH:$HOME/.local/bin
export TMPDIR=$P/tmp/$1; mkdir -p $TMPDIR; export PERF_RUN_TAG=$1 PERF_KEEP_RUNS=0
T=$P/ws_8a/tt_metal/tt-llk/tests
tt-smi -r > $L/reset.txt 2>&1
start=$(date +%s)
(cd $T && timeout 90m pytest python_tests/perf_eltwise_binary.py -k Elwadd -q > $L/pytest.txt 2>&1); rc=$?
echo "exit $rc, $(( $(date +%s) - start )) s, $(tail -1 $L/pytest.txt)" > $L/status.txt
mkdir -p $L/perf_data; cp -r $T/../perf_data/runs/$1/. $L/perf_data/ 2>/dev/null || mkdir -p $L/perf_data
ls $L >> $L/status.txt
