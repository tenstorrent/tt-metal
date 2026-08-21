#!/usr/bin/env bash
# Usage: run_tracy.sh <log-name> <pytest args...>
# Tracy op-level profiling. build_Release already has ENABLE_TRACY=ON (verified in CMakeCache),
# so no rebuild is needed. --op-support-count must be raised well above its 1000 default.
set -o pipefail
cd /data/ssalice/temp/tt-metal || exit 1
export TT_METAL_HOME=$PWD
export PYTHONPATH=$PWD
export MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603
export TT_MISTRAL4_PREFILL_TTNN_CACHE=/data/ssalice/mistral4_ttnn_cache
source python_env/bin/activate
NAME="$1"; shift
LOG="mistral4_bringup/test_logs/${NAME}.log"
rm -rf generated/profiler/.logs
echo "### TRACY CMD: python -m tracy -p -r -v --op-support-count 100000 -m pytest $*" | tee "$LOG"
python -m tracy -p -r -v --op-support-count 100000 -m pytest "$@" -v -s 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "### END rc=$RC" | tee -a "$LOG"
ls -t generated/profiler/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1 | tee -a "$LOG"
exit $RC
