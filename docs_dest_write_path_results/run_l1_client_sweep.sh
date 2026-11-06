#!/usr/bin/env bash
# Runs the Dest write path perf tests once per Quasar L1 client counter selection (subport*8 + event) and keeps
# each run's CSVs under logs/l1_client/<label>/sel_<selection>/.
#
# Usage, from tt_metal/tt-llk/tests/python_tests with the emulator environment set:
#   run_l1_client_sweep.sh <label> "<pytest -k expression>" <selection> [<selection> ...]
set -u
label=$1
filter=$2
shift 2
out=$(cd "$(dirname "$0")" && pwd)/logs/l1_client/$label
mkdir -p "$out"
for sel in "$@"; do
    dest=$out/sel_$sel
    mkdir -p "$dest"
    rm -f /tmp/tt-llk-build/dest_write_path_bank_stamps.jsonl
    LLK_PERF_L1_CLIENT_SEL=$sel pytest --timeout=600 -rN --run-simulator --enable-perf-counters \
        quasar/perf_dest_write_path_exhaustive_quasar.py quasar/perf_dest_preload_matmul_exhaustive_quasar.py \
        -k "$filter" >"$dest/pytest.log" 2>&1
    echo "sel $sel exit=$? $(tail -1 "$dest/pytest.log")"
    latest=$(readlink -f ../../perf_data/latest)
    cp "$latest"/perf_dest_*/perf_dest_*_quasar.csv "$dest"/ 2>/dev/null
    cp /tmp/tt-llk-build/dest_write_path_bank_stamps.jsonl "$dest"/ 2>/dev/null
done
