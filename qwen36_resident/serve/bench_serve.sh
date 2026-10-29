#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# One user TTFT / TPOT of a running serve_local.sh server: vllm bench serve, random prompts of exactly ISL tokens
# (completions endpoint, no chat template), OSL generated tokens (ignore_eos), one request at a time.
#
# usage: bench_serve.sh OUT_DIR TAG ISL [ISL...]
# env: PORT (8000), OSL (128), NUM_PROMPTS (5), SAMPLING ("--temperature 0"; vllm bench serve sends no temperature
#   otherwise and the server then samples with the checkpoint's generation config), VLLM_VENV, HF_MODEL (tokenizer),
#   SERVED_MODEL_NAME (Qwen/Qwen3.8-27B)
set -euo pipefail
OUT=${1:?OUT_DIR}
TAG=${2:?TAG}
shift 2
mkdir -p "$OUT"
VLLM=${VLLM_VENV:+$VLLM_VENV/bin/}vllm
for ISL in "$@"; do
  # shellcheck disable=SC2086
  "$VLLM" bench serve --backend vllm --base-url "http://localhost:${PORT:-8000}" \
    --model "${SERVED_MODEL_NAME:-Qwen/Qwen3.8-27B}" --tokenizer "${HF_MODEL:?}" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "${OSL:-128}" --ignore-eos \
    --num-prompts "${NUM_PROMPTS:-5}" --num-warmups 1 --max-concurrency 1 --request-rate inf \
    ${SAMPLING:---temperature 0} \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90 \
    --save-result --result-dir "$OUT" --result-filename "${TAG}_isl${ISL}.json" 2>&1 |
    grep -E "Successful|Benchmark duration|Total generated|Output token throughput|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|P90 TPOT|Mean ITL|Mean E2EL|Failed" |
    sed "s/^/[$TAG isl=$ISL] /"
done
