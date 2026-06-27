#!/usr/bin/env bash
set -u
cd /home/ttuser/tt-metal/models/experimental/perf_automation || exit 1
export TT_METAL_HOME=/home/ttuser/tt-metal PYTHONPATH=/home/ttuser/tt-metal
export PATH=/home/ttuser/tt-metal/python_env/bin:/home/ttuser/.local/bin:$PATH
set -a; source .env.agent 2>/dev/null; set +a
DEMO=/home/ttuser/tt-metal/models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16
PERF="models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16/tests/e2e/test_perf.py::test_prefill_perf"
echo "===== $(date) BEFORE_LOOP ====="
python -m agent.before_loop "$DEMO" --metric device_ms --devices 0,1 --perf-test "$PERF" -k device_params0 --budget-usd 1000000000 --max-iter 1000 2>&1
[ $? -eq 0 ] && { echo "===== $(date) LOOP ====="; python -m agent.loop runs 2>&1; echo "LOOP_EXIT=$?"; } || echo "before_loop FAILED"
echo "===== $(date) RUN COMPLETE ====="
