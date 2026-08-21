#!/usr/bin/env bash
# Usage: run.sh <log-name> <pytest args...>
# Sets up the standard mistral4 prefill test environment and tees output to a log.
set -o pipefail
cd /data/ssalice/temp/tt-metal || exit 1
export TT_METAL_HOME=$PWD
export PYTHONPATH=$PWD
export MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603
export TT_METAL_OPERATION_TIMEOUT_SECONDS=120
export TT_MISTRAL4_PREFILL_TTNN_CACHE=/data/ssalice/mistral4_ttnn_cache
# The checkpoint dir and its tensor_cache_bh_32dev are read-only, so the weight cache must
# live elsewhere or every run re-converts weights from fp8.
# shellcheck disable=SC1091
source python_env/bin/activate
NAME="$1"; shift
LOG="mistral4_bringup/test_logs/${NAME}.log"
echo "### CMD: python3 -m pytest $*" | tee "$LOG"
echo "### START: $(date -Is)" | tee -a "$LOG"
python3 -m pytest "$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "### END: $(date -Is) rc=$RC" | tee -a "$LOG"
exit $RC
