#!/usr/bin/env bash
# Run a test against the ssalice/mistral4-tests WORKTREE build (not the main repo's).
# Usage: run_wt.sh <log-name> <pytest args...>
set -o pipefail
export TT_METAL_HOME=/tmp/claude-4076/-data-ssalice-temp-tt-metal/b1e60977-fbb5-439d-844a-6d926fd407e2/scratchpad/sf-trial
# Three-entry PYTHONPATH is REQUIRED. With just $TT_METAL_HOME, the shared python_env's
# ttnn-custom.pth wins and `import ttnn` silently loads the MAIN repo's older _ttnn.so.
export PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools
export MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603
export TT_MISTRAL4_PREFILL_TTNN_CACHE=/data/ssalice/mistral4_ttnn_cache
export TT_METAL_OPERATION_TIMEOUT_SECONDS=120
# shellcheck disable=SC1091
source "$TT_METAL_HOME/python_env/bin/activate"
cd "$TT_METAL_HOME" || exit 1
# Guard against the silent-fallback failure mode above.
python3 -c "import ttnn; assert 'sf-trial' in ttnn._ttnn.__file__, ttnn._ttnn.__file__; assert hasattr(ttnn.RoutedExpertActivation,'SituGlu')" || { echo "FATAL: wrong _ttnn.so loaded"; exit 1; }
NAME="$1"; shift
LOG="/data/ssalice/temp/tt-metal/mistral4_bringup/test_logs/on_mistral4_tests/${NAME}.log"
echo "### BRANCH: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)" | tee "$LOG"
echo "### CMD: python3 -m pytest $*" | tee -a "$LOG"
echo "### START: $(date -Is)" | tee -a "$LOG"
python3 -m pytest "$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "### END: $(date -Is) rc=$RC" | tee -a "$LOG"
exit $RC
