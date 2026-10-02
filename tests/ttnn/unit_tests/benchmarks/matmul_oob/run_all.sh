#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Full matmul default-config validation, as run on Wormhole for #57884 (see README.md in this directory):
#   1. selector gtests (device-free)
#   2. benchmark suite: every case in cases.csv, legacy selection (oob) vs the new one (v2), device time + PCC
#   3. matmul pytest directory with ttnn.CONFIG.matmul_auto_config_v2 off and on, outcome + device time per test
#
# Usage, from the repo root with python_env active:
#   tests/ttnn/unit_tests/benchmarks/matmul_oob/run_all.sh [--out DIR] [--skip-suite] [--skip-pytest]
# Every stage writes into DIR (default generated/matmul_oob/<arch>_<git rev>) and can be rerun after an
# interruption: the suite resumes where it stopped, a finished pytest pass is skipped.
set -u

HERE=tests/ttnn/unit_tests/benchmarks/matmul_oob
OUT=""
SKIP_SUITE=0
SKIP_PYTEST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --skip-suite) SKIP_SUITE=1; shift ;;
        --skip-pytest) SKIP_PYTEST=1; shift ;;
        *) echo "unknown argument $1"; exit 2 ;;
    esac
done

if [ ! -f "$HERE/run_suite.py" ]; then
    echo "run from the tt-metal repo root"
    exit 2
fi
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "activate python_env first (source python_env/bin/activate)"
    exit 2
fi

ARCH=$(python3 -c "import ttnn; d = ttnn.open_device(device_id=0); print(str(d.arch()).split('.')[-1].lower()); ttnn.close_device(d)" 2>/dev/null | tail -1)
REV=$(git rev-parse --short HEAD)
OUT=${OUT:-generated/matmul_oob/${ARCH}_${REV}}
mkdir -p "$OUT"
echo "arch=$ARCH rev=$REV out=$OUT" | tee "$OUT/run_info.txt"
git log -1 --oneline >> "$OUT/run_info.txt"

# A profiler log from another build makes the profiler abort with colliding source-location hashes
rm -f generated/profiler/.logs/zone_src_locations.log

step() { echo; echo "=== $(date '+%F %T') $*" | tee -a "$OUT/run_info.txt"; }

step "1/3 selector gtests"
./build_Release/test/ttnn/unit_tests_ttnn --gtest_filter='MatmulAutoConfig.*' > "$OUT/gtests.log" 2>&1
grep -E "PASSED|FAILED" "$OUT/gtests.log" | tee -a "$OUT/run_info.txt"

if [ $SKIP_SUITE -eq 0 ]; then
    step "2/3 benchmark suite (legacy vs v2, $(($(wc -l < $HERE/cases.csv) - 1)) cases)"
    python3 $HERE/run_suite.py --cases-csv $HERE/cases.csv --modes oob v2 --out "$OUT/suite.csv" --resume \
        > "$OUT/suite.log" 2>&1
    echo "exit $?" | tee -a "$OUT/run_info.txt"
    python3 $HERE/summarize.py "$OUT/suite.csv" --base-mode oob --new-mode v2 > "$OUT/suite_summary.txt" 2>&1
    head -12 "$OUT/suite_summary.txt" | tee -a "$OUT/run_info.txt"
fi

if [ $SKIP_PYTEST -eq 0 ]; then
    for mode in off on; do
        if grep -q "pytest $mode done" "$OUT/run_info.txt" 2>/dev/null; then
            continue
        fi
        step "3/3 matmul pytest directory, matmul_auto_config_v2 $mode"
        rm -f "$OUT/pytest_$mode.jsonl"
        if [ $mode = on ]; then
            overrides='{"matmul_auto_config_v2": true}'
        else
            overrides='{}'
        fi
        PYTHONPATH=$HERE:${PYTHONPATH:-} TTNN_CONFIG_OVERRIDES="$overrides" DEVICE_TIME_OUT="$OUT/pytest_$mode.jsonl" \
            pytest -q -p no:logging -p pytest_device_time tests/ttnn/unit_tests/operations/matmul/ \
            > "$OUT/pytest_$mode.log" 2>&1
        echo "pytest $mode done (exit $?)" | tee -a "$OUT/run_info.txt"
    done
    python3 $HERE/compare_pytest_times.py "$OUT/pytest_off.jsonl" "$OUT/pytest_on.jsonl" > "$OUT/pytest_compare.txt" 2>&1
    head -12 "$OUT/pytest_compare.txt" | tee -a "$OUT/run_info.txt"
fi

step "done"
tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "results: $OUT (archive $OUT.tar.gz)"
