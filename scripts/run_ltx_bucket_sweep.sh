#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/generated/ltx_bucket_sweep/$RUN_ID}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
LOG_PATH="$OUTPUT_DIR/sweep.log"

export TT_METAL_SHM_TRACKING_DISABLED="${TT_METAL_SHM_TRACKING_DISABLED:-1}"
export TT_METAL_INSPECTOR="${TT_METAL_INSPECTOR:-0}"
export TT_METAL_LOGS_PATH="${TT_METAL_LOGS_PATH:-/tmp/tt-logs}"
export LTX_VOC_TRACE="${LTX_VOC_TRACE:-0}"
export LTX_BWE_TRACE="${LTX_BWE_TRACE:-0}"
export LTX_VAE_TRACE=0
export LTX_VAE_TEMPORAL_CHUNK_LATENTS=0
export LTX_BUCKET_TEST_SERVED_CONFIGS=all
export LTX_BUCKET_TEST_CONFIGS=all
export LTX_BUCKET_TEST_CONTINUE_ON_FAILURE=1
export LTX_BUCKET_TEST_OUTPUT_DIR="$OUTPUT_DIR"

echo "LTX bucket sweep"
echo "Outputs: $OUTPUT_DIR"
echo "The pipeline will warm all traces once, then render every served config without VAE chunking."

set +e
python_env/bin/pytest \
    models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py \
    -k bucket_multi_rung -x -s --timeout=86400 "$@" 2>&1 | tee "$LOG_PATH"
status=${PIPESTATUS[0]}
set -e

printf '%s\n' "$status" >"$OUTPUT_DIR/exit_code.txt"
echo "Sweep exit code: $status"
echo "Summary: $OUTPUT_DIR/summary.json"
echo "Results: $OUTPUT_DIR/results.jsonl"
echo "Log:     $LOG_PATH"
exit "$status"
