#!/usr/bin/env bash
# 400-ep LIBERO benchmark on ttnn_16_decode (8-chip pipelined matmul_decode).
# Mirrors production run_libero_400ep_1x8_parallel.sh config (4 suites x 10 tasks
# x 10 episodes = 400, N=5) but SERIAL: the 16-chip pipeline owns the full (4,8)
# Galaxy, so suites cannot run in parallel. Each suite is a FRESH process (robust
# to a mid-run crash; isolates per-suite state). Target: match production 393/400.
set -uo pipefail
ROOT=/home/tt-admin/sdawle/pi05_openpi_upstream_bh_glx_trace/tt-metal
cd "$ROOT"
source _bench_runs/pi05_production.env 2>/dev/null || true
export TT_METAL_HOME="$ROOT"
export TT_VISIBLE_DEVICES="$(seq -s, 0 31)"
export PI0_TOKENIZER_PATH=/home/tt-admin/pi05_cache/tokenizer/paligemma_tokenizer.model
export PYTHONPATH="$ROOT:/home/tt-admin/pi05_cache/libero_repo"
export LIBERO_REPO_PATH=/home/tt-admin/pi05_cache/libero_repo
export MUJOCO_GL=osmesa

SUITES=(libero_spatial libero_object libero_goal libero_10)
LOG_DIR="$ROOT/_bench_runs/16decode_400ep_logs"
mkdir -p "$LOG_DIR"

for suite in "${SUITES[@]}"; do
    echo "############ SUITE $suite (fresh process) ############"
    python_env/bin/python -u -m models.experimental.pi0_5.eval.libero_rollout \
        --backend ttnn_16_decode \
        --checkpoint /home/tt-admin/pi05_cache/pi05_libero_upstream \
        --suites "$suite" --task-range 0 9 \
        --num-episodes 10 --action-horizon 10 --steps-sweep 5 --state-in-prompt false \
        2>&1 | tee "$LOG_DIR/${suite}.log"
    echo "SUITE_${suite}_EXIT=${PIPESTATUS[0]}"
    echo "############ device reset after $suite ############"
    tt-smi -r >/dev/null 2>&1 || true
    sleep 15
done

echo "############ 400EP RUN COMPLETE — per-suite totals ############"
for suite in "${SUITES[@]}"; do
    echo "--- $suite ---"
    grep -E "$suite +task +[0-9]:" "$LOG_DIR/${suite}.log" 2>/dev/null | tail -12
done
