#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Run DiffusionGemma up-front capture + traced early halt on GPQA-Diamond.

Usage:
  run_upfront_gpqa.sh smoke   # samples 0 and 1
  run_upfront_gpqa.sh full    # smoke first, then all 198 samples (default)

Useful overrides:
  TT_METAL_ROOT=/home/zni/tt-metal
  TT_VLLM_ROOT=/home/zni/tt-vllm
  TT_INFERENCE_SERVER_ROOT=/home/zni/tt-inference-server
  MODEL_VENV=/home/zni/venvs/tt-diffusion-gemma
  DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
  HOST=127.0.0.1 PORT=8010
  MAX_MODEL_LEN=4096 MAX_GEN_TOKS=1536
  THINKING_MODE=1                 # inject the checkpoint's <|think|> system token
  OUTPUT_ROOT=/tmp/dg-upfront-gpqa-<timestamp>
  RESET_BEFORE=1 RESET_AFTER=1

The default prefill whitelist is exact for the current 198-sample
r1_gpqa_diamond task + DiffusionGemma chat template in thinking mode.
Recompute it if the checkpoint, tokenizer, chat template, system prompt,
thinking mode, or task prompt changes.
EOF
}

MODE="${1:-full}"
case "${MODE}" in
    smoke | full) ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "ERROR: mode must be 'smoke' or 'full', got '${MODE}'" >&2
        usage >&2
        exit 2
        ;;
esac

TT_METAL_ROOT="${TT_METAL_ROOT:-/home/zni/tt-metal}"
TT_VLLM_ROOT="${TT_VLLM_ROOT:-/home/zni/tt-vllm}"
TT_INFERENCE_SERVER_ROOT="${TT_INFERENCE_SERVER_ROOT:-/home/zni/tt-inference-server}"
MODEL_VENV="${MODEL_VENV:-/home/zni/venvs/tt-diffusion-gemma}"
DG_CKPT="${DG_CKPT:-/home/zni/dg_models/diffusiongemma-26B-A4B-it}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
MODEL_NAME="${MODEL_NAME:-google/diffusiongemma-26B-A4B-it}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-1536}"
THINKING_MODE="${THINKING_MODE:-1}"
TRACE_REGION_SIZE="${TRACE_REGION_SIZE:-12884901888}" # 12 GiB
RESET_BEFORE="${RESET_BEFORE:-1}"
RESET_AFTER="${RESET_AFTER:-1}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"

case "${THINKING_MODE}" in
    1)
        # Exact aligned lengths after adding a system turn containing <|think|>.
        DEFAULT_PREFILL_WARMUP_LENS="128,160,192,224,256,288,320,352,384,416,448,480,512,544,576,608,672,832,2432"
        ;;
    0)
        DEFAULT_PREFILL_WARMUP_LENS="96,128,160,192,224,256,288,320,352,384,416,448,480,512,544,608,640,832,2432"
        ;;
    *)
        echo "ERROR: THINKING_MODE must be 0 or 1, got '${THINKING_MODE}'" >&2
        exit 2
        ;;
esac
PREFILL_WARMUP_LENS="${PREFILL_WARMUP_LENS:-${DEFAULT_PREFILL_WARMUP_LENS}}"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/dg-upfront-gpqa-${TIMESTAMP}}"
SERVER_LOG="${OUTPUT_ROOT}/server.log"
SMOKE_OUTPUT="${OUTPUT_ROOT}/smoke"
FULL_OUTPUT="${OUTPUT_ROOT}/full"

MODEL_PYTHON="${MODEL_VENV}/bin/python"
LM_EVAL="${TT_INFERENCE_SERVER_ROOT}/.workflow_venvs/.venv_evals_common/bin/lm_eval"
TT_SMI_BIN="${TT_SMI_BIN:-tt-smi}"

require_path() {
    if [[ ! -e "$1" ]]; then
        echo "ERROR: required path does not exist: $1" >&2
        exit 1
    fi
}

require_path "${TT_METAL_ROOT}"
require_path "${TT_VLLM_ROOT}"
require_path "${TT_INFERENCE_SERVER_ROOT}"
require_path "${MODEL_PYTHON}"
require_path "${LM_EVAL}"
require_path "${DG_CKPT}/config.json"

for command in curl flock setsid timeout "${TT_SMI_BIN}"; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "ERROR: command not found: ${command}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_ROOT}"
exec 9>/tmp/dg-mesh.lock
if ! flock -n 9; then
    echo "ERROR: another DiffusionGemma device job owns /tmp/dg-mesh.lock" >&2
    exit 1
fi

if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
    echo "ERROR: an existing vLLM API server process is running" >&2
    exit 1
fi

SERVER_PID=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM

    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        echo "Stopping vLLM server (process group ${SERVER_PID})..."
        kill -TERM -- "-${SERVER_PID}" >/dev/null 2>&1 || true
        for _ in $(seq 1 60); do
            if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
            echo "Server did not stop cleanly; sending SIGKILL" >&2
            kill -KILL -- "-${SERVER_PID}" >/dev/null 2>&1 || true
        fi
    fi

    if [[ "${RESET_AFTER}" == "1" ]]; then
        echo "Resetting devices after run..."
        timeout 180 "${TT_SMI_BIN}" -r || true
    fi

    echo "Artifacts: ${OUTPUT_ROOT}"
    exit "${status}"
}
trap cleanup EXIT INT TERM

if [[ "${RESET_BEFORE}" == "1" ]]; then
    echo "Resetting devices before run..."
    timeout 180 "${TT_SMI_BIN}" -r
    timeout 60 "${TT_SMI_BIN}" -ls --local
fi

TT_CONFIG="$(printf '{"tt":{"sample_on_device_mode":"all","enable_model_warmup":true,"trace_mode":"all","trace_region_size":%s}}' "${TRACE_REGION_SIZE}")"
PYTHONPATH_VALUE="${TT_METAL_ROOT}:${TT_VLLM_ROOT}:${TT_VLLM_ROOT}/plugins/vllm-tt-plugin/src"

echo "Starting DiffusionGemma server..."
echo "  output: ${OUTPUT_ROOT}"
echo "  max_model_len: ${MAX_MODEL_LEN}"
echo "  max_gen_toks: ${MAX_GEN_TOKS}"
echo "  thinking mode: ${THINKING_MODE}"
echo "  prefill whitelist: ${PREFILL_WARMUP_LENS}"

# The host mode generates IID full-vocabulary Gumbel noise with torch, then keeps
# sampling/denoise on device. DiffusionConfig supplies the checkpoint's released
# T=0.8->0.4 schedule, entropy bound 0.1, and early-halt thresholds. Do not replace
# this with chunked: QB2's current 1024-wide RNG has a known distribution bias.
setsid env \
    TT_METAL_HOME="${TT_METAL_ROOT}" \
    TT_METAL_RUNTIME_ROOT="${TT_METAL_ROOT}" \
    PYTHONPATH="${PYTHONPATH_VALUE}" \
    MESH_DEVICE=P150x4 \
    DG_CKPT="${DG_CKPT}" \
    DG_UPFRONT_CAPTURE=1 \
    DG_UPFRONT_PREFILL_WARMUP_LENS="${PREFILL_WARMUP_LENS}" \
    DG_DENOISE_REVEAL_MASK=1 \
    DG_DENOISE_REVEAL_PMAX="${MAX_MODEL_LEN}" \
    DG_DENOISE_LAZY_CAPTURE=0 \
    DG_DENOISE_EARLY_HALT=1 \
    DG_DENOISE_EARLY_HALT_WINDOW=1 \
    DG_VLLM_GUMBEL_MODE=host \
    DG_VLLM_TRACE=1 \
    DG_TRACE_REGION_SIZE="${TRACE_REGION_SIZE}" \
    DG_VLLM_MAX_DENOISE_STEPS=48 \
    DG_SPARSE_MOE=1 \
    DG_SPARSE_MOE_TUNED=1 \
    DG_DEDUP_ARGMAX=1 \
    VLLM_RPC_TIMEOUT=1800000 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TT_LOGGER_LEVEL=ERROR \
    "${MODEL_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${DG_CKPT}" \
    --served-model-name "${MODEL_NAME}" \
    --generation-config vllm \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_MODEL_LEN}" \
    --max-num-seqs 1 \
    --block-size 64 \
    --additional-config "${TT_CONFIG}" \
    --host "${HOST}" \
    --port "${PORT}" \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Waiting up to ${READY_TIMEOUT_S}s for server readiness..."
ready=0
for _ in $(seq 1 "${READY_TIMEOUT_S}"); do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        echo "ERROR: server exited during startup" >&2
        tail -n 200 "${SERVER_LOG}" >&2
        exit 1
    fi
    if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done

if [[ "${ready}" != "1" ]]; then
    echo "ERROR: server did not become ready within ${READY_TIMEOUT_S}s" >&2
    tail -n 200 "${SERVER_LOG}" >&2
    exit 1
fi

echo "Server is ready."

MODEL_ARGS="model=${MODEL_NAME},base_url=http://${HOST}:${PORT}/v1/chat/completions,tokenizer_backend=huggingface,max_length=${MAX_MODEL_LEN},num_concurrent=1"
# HTTP temperature/top-k/top-p/seed are not part of the released DiffusionGemma
# sampler and are not wired into its model-owned denoise loop. Keep only transport
# and output-length settings; sampling comes from the checkpoint configuration.
GEN_KWARGS="stream=false,max_gen_toks=${MAX_GEN_TOKS},until=[]"
COMMON_EVAL_ARGS=(
    --tasks r1_gpqa_diamond
    --model local-chat-completions
    --model_args "${MODEL_ARGS}"
    --gen_kwargs "${GEN_KWARGS}"
    --seed 42
    --num_fewshot 0
    --batch_size 1
    --log_samples
    --apply_chat_template
    --trust_remote_code
    --confirm_run_unsafe_code
)
if [[ "${THINKING_MODE}" == "1" ]]; then
    # The model card's recommended reasoning mode is enabled by putting this
    # control token at the start of the system prompt.
    COMMON_EVAL_ARGS+=(--system_instruction "<|think|>")
fi

echo "Running two-sample smoke..."
SMOKE_SAMPLES='{"r1_gpqa_diamond":[0,1]}'
"${LM_EVAL}" \
    "${COMMON_EVAL_ARGS[@]}" \
    --output_path "${SMOKE_OUTPUT}" \
    --samples "${SMOKE_SAMPLES}" \
    2>&1 | tee "${OUTPUT_ROOT}/smoke.log"

if [[ "${MODE}" == "full" ]]; then
    echo "Running all 198 GPQA-Diamond samples..."
    "${LM_EVAL}" \
        "${COMMON_EVAL_ARGS[@]}" \
        --output_path "${FULL_OUTPUT}" \
        2>&1 | tee "${OUTPUT_ROOT}/full.log"
fi

echo
echo "Run completed successfully."
echo "Server metrics summary:"
grep 'DG_VLLM_METRIC.*\"event\": \"prefill_block0\"' "${SERVER_LOG}" || true
echo
echo "Request releases:"
grep -c 'DG_VLLM_METRIC.*\"event\": \"request_release\"' "${SERVER_LOG}" || true
