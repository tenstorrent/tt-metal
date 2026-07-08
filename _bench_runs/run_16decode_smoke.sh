#!/usr/bin/env bash
# Rung-0 re-confirm: does the existing carve-based ttnn_16_decode prefill QKV
# still FATAL with current code (env-var defaults now applied to this backend)?
# 1 ep, N=5, libero_spatial:0. Logs to _bench_runs/16decode_smoke.log.
set -uo pipefail
cd /home/tt-admin/sdawle/pi05_openpi_upstream_bh_glx_trace/tt-metal

source _bench_runs/pi05_production.env 2>/dev/null || true

export TT_METAL_HOME="$PWD"
export TT_VISIBLE_DEVICES="$(seq -s, 0 31)"
export PI0_TOKENIZER_PATH=/home/tt-admin/pi05_cache/tokenizer/paligemma_tokenizer.model
export PYTHONPATH="$PWD:/home/tt-admin/pi05_cache/libero_repo"
export LIBERO_REPO_PATH=/home/tt-admin/pi05_cache/libero_repo
export MUJOCO_GL=osmesa

python_env/bin/python -u -m models.experimental.pi0_5.eval.libero_rollout \
    --backend ttnn_16_decode \
    --checkpoint /home/tt-admin/pi05_cache/pi05_libero_upstream \
    --suites libero_spatial --task-range 0 0 \
    --num-episodes 1 --action-horizon 10 --steps-sweep 5 \
    --state-in-prompt false
echo "EXIT_CODE=$?"
