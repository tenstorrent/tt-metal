#!/usr/bin/env bash
# 400-ep LIBERO benchmark on the e2e SOCKET pipeline (no host bounce): traced
# vision+prefill + device-direct KV-concat sockets + traced denoise. ~52ms/chunk.
# 4 suites x 10 tasks x 10 ep = 400, N=5. glx_reset before each suite (proper
# Galaxy reset). Per-prompt socket teardown+re-setup handled in pipeline_16_decode.
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
export PI0_KV_SOCKET=1
LOG_DIR="$ROOT/_bench_runs/16decode_400ep_socket_logs"
mkdir -p "$LOG_DIR"
for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "############ glx_reset before $suite ############"
    tt-smi -glx_reset >/dev/null 2>&1 || true
    sleep 10
    echo "############ SUITE $suite (fresh process, socket e2e) ############"
    python_env/bin/python -u -m models.experimental.pi0_5.eval.libero_rollout \
        --backend ttnn_16_decode --checkpoint /home/tt-admin/pi05_cache/pi05_libero_upstream \
        --suites "$suite" --task-range 0 9 \
        --num-episodes 10 --action-horizon 10 --steps-sweep 5 --state-in-prompt false \
        2>&1 | tee "$LOG_DIR/${suite}.log"
    echo "SUITE_${suite}_EXIT=${PIPESTATUS[0]}"
done
echo "############ SOCKET 400EP COMPLETE — per-suite totals ############"
for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "--- $suite ---"
    grep -E "$suite +task +[0-9]:" "$LOG_DIR/${suite}.log" 2>/dev/null | tail -12
done
