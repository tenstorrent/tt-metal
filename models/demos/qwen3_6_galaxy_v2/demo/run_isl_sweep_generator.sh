#!/usr/bin/env bash
# Qwen3.6-27B BH_GLX generator-path (server decode path) ISL sweep.
# Runs test_qwen36_demo_generator_batch1 (Generator.prefill_forward_text +
# Generator.decode_forward, TRACED decode + on-device top-k/p/temp sampling)
# across all ISLs, each in its own process, full output to its own log file.
# Reports decode tok/s/user + generated text (coherence eyeball) per ISL.
set -uo pipefail
cd /home/tt-admin/ssinghal/qwen36/new/tt-metal
export TT_METAL_HOME="$(pwd)"
export PYTHONPATH="$(pwd)"
source python_env/bin/activate

export HF_MODEL=Qwen/Qwen3.6-27B
export MESH_DEVICE=BH_GLX
# decode coherence-eyeball length (generator path caps at QWEN36_GEN_DECODE_STEPS)
export QWEN36_GEN_DECODE_STEPS=64

LOGDIR=/tmp/qwen36_isl_sweep
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/SUMMARY.txt"
: > "$SUMMARY"

ISLS=(128 4096 8192 16384 32768 65536 131072 262144)

for ISL in "${ISLS[@]}"; do
    LOG="$LOGDIR/isl_${ISL}.log"
    echo "================ ISL=$ISL  ($(date '+%H:%M:%S')) ================" | tee -a "$SUMMARY"
    # chunked GDN prefill for long context (>=16k) per run_text_demo.sh guidance
    CHUNK_ENV=""
    if [ "$ISL" -ge 16384 ]; then
        CHUNK_ENV="QWEN36_PREFILL_CHUNK=4096"
    fi
    env QWEN36_PERF_T_PREFILL="$ISL" $CHUNK_ENV \
        python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/demo/text_demo_qwen36.py::test_qwen36_demo_generator_batch1 \
        -v -s > "$LOG" 2>&1
    RC=$?
    echo "ISL=$ISL exit=$RC log=$LOG  ($(date '+%H:%M:%S'))" | tee -a "$SUMMARY"
done
echo "================ SWEEP DONE ($(date '+%H:%M:%S')) ================" | tee -a "$SUMMARY"
