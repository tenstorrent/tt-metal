#!/usr/bin/env bash
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
export PI0_TIMED=1
export PI0_KV_SOCKET=1
python_env/bin/python -u -m models.experimental.pi0_5.eval.libero_rollout \
    --backend ttnn_16_decode --checkpoint /home/tt-admin/pi05_cache/pi05_libero_upstream \
    --suites libero_spatial --task-range 0 0 --num-episodes 1 --action-horizon 10 --steps-sweep 5 --state-in-prompt false
echo "TIMED_EXIT=$?"
