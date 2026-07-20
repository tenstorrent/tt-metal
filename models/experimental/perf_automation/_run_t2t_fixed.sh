#!/usr/bin/env bash
# Full T2T optimize run — NO early quit. High budget (not the $5 default) + high
# max-iter so the loop stops ONLY at natural bucket exhaustion (matmul -> datamove
# -> reduction -> eltwise), with the waste-judge fixes (advance, not stop).
set -u
cd /home/ttuser/tt-metal/models/experimental/perf_automation || exit 1
export TT_METAL_HOME=/home/ttuser/tt-metal PYTHONPATH=/home/ttuser/tt-metal
export PATH=/home/ttuser/tt-metal/python_env/bin:/home/ttuser/.local/bin:$PATH
set -a; source .env.agent 2>/dev/null; set +a

DEMO=/home/ttuser/tt-metal/models/demos/hf_seamless_m4t_medium
PERF="models/demos/hf_seamless_m4t_medium/tests/e2e/test_t2t_perf.py::test_t2t_perf"

echo "===== $(date) : BEFORE_LOOP ====="
python -m agent.before_loop "$DEMO" --metric device_ms --devices 0,1 \
    --perf-test "$PERF" -k in0 --budget-usd 1000000000 --max-iter 1000 2>&1
BL=$?
echo "before_loop exit=$BL"
if [ "$BL" -eq 0 ]; then
    echo "===== $(date) : LOOP ====="
    python -m agent.loop runs 2>&1
    echo "loop exit=$?"
else
    echo "before_loop FAILED; not starting loop"
fi
echo "===== $(date) : RUN COMPLETE ====="
