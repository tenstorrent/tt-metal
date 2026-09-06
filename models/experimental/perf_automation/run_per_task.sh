#!/usr/bin/env bash
# Per-task perf optimization for a MULTI-TASK model.
#
# Optimizes one task at a time: for each task we pin ITS perf test and run the
# full tool (before_loop + loop). exec_scope then auto-restricts edits to the
# files that task actually executes. So "what is edited" and "what is measured"
# always belong to the SAME task -- which fixes the multi-task mismatch at the
# orchestration level (no per-edit routing / component->task map needed).
#
# Usage:  DEVICES=0,1 MAX_ITER=10 TASKS="t2t s2tt t2s" ./run_per_task.sh [<demo_dir>]
set -u

DEMO="${1:-/home/ttuser/tt-metal/models/demos/hf_seamless_m4t_medium}"
RELDEMO="${DEMO#*/tt-metal/}"                       # repo-relative path for the perf-test pin
DEVICES="${DEVICES:-0,1}"
MAX_ITER="${MAX_ITER:-10}"
TASKS="${TASKS:-t2t s2tt t2s}"

export TT_METAL_HOME=/home/ttuser/tt-metal PYTHONPATH=/home/ttuser/tt-metal
export PATH=/home/ttuser/tt-metal/python_env/bin:/home/ttuser/.local/bin:$PATH
cd /home/ttuser/tt-metal/models/experimental/perf_automation || exit 1

for TASK in $TASKS; do
    PERF="$RELDEMO/tests/e2e/test_${TASK}_perf.py::test_${TASK}_perf"
    echo "==================================================================="
    echo "  TASK: $TASK   ->  $PERF"
    echo "==================================================================="
    if python -m agent.before_loop "$DEMO" --metric device_ms --devices "$DEVICES" \
        --perf-test "$PERF" -k in0 --max-iter "$MAX_ITER"; then
        python -m agent.loop runs || echo "  [loop failed for $TASK; continuing]"
    else
        echo "  [before_loop failed for $TASK; skipping]"
    fi
done
echo "==================================================================="
echo "  ALL TASKS DONE"
echo "==================================================================="
