#!/usr/bin/env bash
# run_trial.sh <trial_name> [probe args...]
# Resets device, runs probe under flock+timeout 120, appends outcome row to trials.tsv
set -u
SP=/tmp/claude-1000/-home-nachiket-tt-metal/9f8f10d4-baba-4138-8904-cb9bdebdbd08/scratchpad/storm/validate
NAME="$1"; shift
cd /home/nachiket/tt-metal
./tt_metal/tt-llk/tests/.venv/bin/tt-smi -r >/dev/null 2>&1
source python_env/bin/activate
LOG="$SP/trial_${NAME}.log"
flock /tmp/tt-device.lock timeout 120 python "$SP/probe_topk.py" "$@" > "$LOG" 2>&1
RC=$?
if grep -q PROBE_DONE "$LOG"; then OUT=PASS
elif grep -q VALUES_READ "$LOG"; then OUT=HANG_AFTER_VALUES
elif grep -q DISPATCH_RETURNED "$LOG"; then OUT=HANG_IN_READBACK
elif grep -q PROBE_START "$LOG"; then OUT=HANG_BEFORE_DISPATCH
else OUT=STARTUP_FAIL
fi
echo -e "${NAME}\t${OUT}\trc=${RC}\targs: $*" | tee -a "$SP/trials.tsv"
