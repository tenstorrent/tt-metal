#!/usr/bin/env bash
# 11-episode parity: ttnn_16_decode (8-chip pipelined matmul_decode) vs production
# ttnn_1x8, same task (libero_spatial:0), N=5, action-horizon 10. Compares per-episode
# success so we can confirm the 16-chip denoise matches production.
set -uo pipefail
ROOT=/home/tt-admin/sdawle/pi05_openpi_upstream_bh_glx_trace/tt-metal
cd "$ROOT"
source _bench_runs/pi05_production.env 2>/dev/null || true
export TT_METAL_HOME="$ROOT"
export PI0_TOKENIZER_PATH=/home/tt-admin/pi05_cache/tokenizer/paligemma_tokenizer.model
export PYTHONPATH="$ROOT:/home/tt-admin/pi05_cache/libero_repo"
export LIBERO_REPO_PATH=/home/tt-admin/pi05_cache/libero_repo
export MUJOCO_GL=osmesa

COMMON_ARGS="--checkpoint /home/tt-admin/pi05_cache/pi05_libero_upstream \
  --suites libero_spatial --task-range 0 0 \
  --num-episodes 11 --action-horizon 10 --steps-sweep 5 --state-in-prompt false"

echo "############ RUN A: ttnn_16_decode (32 chips visible) ############"
TT_VISIBLE_DEVICES="$(seq -s, 0 31)" \
  python_env/bin/python -u -m models.experimental.pi0_5.eval.libero_rollout \
    --backend ttnn_16_decode $COMMON_ARGS
echo "A_EXIT=$?"

echo "############ device reset between backends ############"
tt-smi -r >/dev/null 2>&1 || true
sleep 15

echo "############ RUN B: ttnn_1x8 (chips 8-15, production) ############"
TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  python_env/bin/python -u -m models.experimental.pi0_5.eval.libero_rollout \
    --backend ttnn_1x8 $COMMON_ARGS
echo "B_EXIT=$?"
echo "############ PARITY RUN COMPLETE ############"
