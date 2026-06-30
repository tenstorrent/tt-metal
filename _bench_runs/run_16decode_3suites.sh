#!/usr/bin/env bash
# Re-run the 3 suites missed by the wedge (object/goal/10) on the traced pipeline,
# with a PROPER tt-smi -glx_reset before each (the -r reset is ineffective on this
# Galaxy's CPLD FW, which caused the inter-suite IndexError wedge).
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
LOG_DIR="$ROOT/_bench_runs/16decode_400ep_logs"
mkdir -p "$LOG_DIR"
for suite in libero_object libero_goal libero_10; do
    echo "############ glx_reset before $suite ############"
    tt-smi -glx_reset >/dev/null 2>&1 || true
    sleep 10
    echo "############ SUITE $suite (fresh process) ############"
    python_env/bin/python -u -m models.experimental.pi0_5.eval.libero_rollout \
        --backend ttnn_16_decode --checkpoint /home/tt-admin/pi05_cache/pi05_libero_upstream \
        --suites "$suite" --task-range 0 9 \
        --num-episodes 10 --action-horizon 10 --steps-sweep 5 --state-in-prompt false \
        2>&1 | tee "$LOG_DIR/${suite}.log"
    echo "SUITE_${suite}_EXIT=${PIPESTATUS[0]}"
done
echo "############ 3-SUITE RERUN COMPLETE ############"
