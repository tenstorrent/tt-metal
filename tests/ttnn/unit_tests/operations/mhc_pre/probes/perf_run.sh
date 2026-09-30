#!/bin/bash
# Perf 1 measurement helper (in-process DEVICE KERNEL DURATION via test_mhc_pre_perf_inproc.py).
# usage: DEV=<card> [DT=xbf16] [KN=default] [TEST=<inproc test path>] probes/perf_run.sh "<TxC,...>" "<kernel defines>" [repeat]
#   defines e.g. "KERNEL_PERF_ZONES" or "MHC_ABLATE_PROJ;MHC_ABLATE_COEF" (MHC_PRE_KERNEL_DEFINES)
#   zone CSV of the run: generated/dev$DEV/profiler/.logs/profile_log_device.csv (-> probes/zone_report.py <csv>)
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate
DEV=${DEV:-0}
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
export TT_METAL_PROFILER_DIR="$PWD/generated/dev$DEV/profiler"
export MHC_PRE_PERF_SHAPES="$1" MHC_PRE_KERNEL_DEFINES="$2" MHC_PRE_PERF_DTYPES=${DT:-xbf16} MHC_PRE_PERF_KNOBS=${KN:-default} MHC_PRE_PERF_REPEAT=${3:-3}
scripts/run_safe_pytest.sh --device "$DEV" --run-all -s ${TEST:-tests/ttnn/unit_tests/operations/mhc_pre/test_mhc_pre_perf_inproc.py} 2>&1 \
  | grep -v 'riscv-tt-elf' | grep -E '^PERF .*us \||passed|failed| error|Error:|SAFE_PYTEST_RESULT' | cut -c1-400
