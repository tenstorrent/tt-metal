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
  MAX_MODEL_LEN=4096 MAX_GEN_TOKS=<derived: MAX_MODEL_LEN - 2432, floored to a canvas>
  THINKING_MODE=1                 # enable the checkpoint's server-side thinking template
  REASONING_PARSER=0              # 1 splits the answer into message.reasoning; the score then lies
  OUTPUT_ROOT=/home/zni/dg_runs/diffusion_gemma/upfront_gpqa/<timestamp>
  RESET_BEFORE=1 RESET_AFTER=1

The default prefill whitelist is exact for the current 198-sample
gpqa_diamond_cot_zeroshot task + DiffusionGemma chat template in thinking mode.
Recompute it with doc/optimize_perf/compute_prefill_whitelist.py if the
checkpoint, tokenizer, chat template, system prompt, thinking mode, or task
prompt changes -- do not edit it by hand.
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
# DERIVED, not chosen. The server rejects max_tokens > max_model_len outright, and prompt + output
# must also fit, so the budget is bounded by MAX_MODEL_LEN minus the longest whitelisted prompt.
# Raising it to 8192 at MAX_MODEL_LEN=4096 made every one of the 198 requests fail with
# "max_tokens=8192 cannot be greater than max_model_len=4096" -- the original 1536 was not an
# arbitrary value, it was this bound rounded to a canvas multiple.
#
# This matters for interpreting a score: the A100 reference ran at max_model_len=262144 with
# max_gen_toks=126976, i.e. ~496 canvases of room, while a 4096 run gets 6. A CoT chain truncated
# before its conclusion cannot emit an extractable answer, so that budget asymmetry depresses the TT
# number for reasons unrelated to whatever flag is under test. It is CONSTANT across arms, so
# arm-to-arm comparison stays valid; comparing a 4096-class arm directly to the 70.45% bar does not.
# Raise MAX_MODEL_LEN for a bar-comparable run -- on QB2 that competes with DRAM (concat-experts
# weights alone are 7.8 GiB and the trace reservation grows with reveal_pmax).
MAX_GEN_TOKS_DEFAULT=$(( MAX_MODEL_LEN - 2432 ))                       # longest whitelisted prompt
MAX_GEN_TOKS_DEFAULT=$(( MAX_GEN_TOKS_DEFAULT / 256 * 256 ))           # whole canvases
MAX_GEN_TOKS="${MAX_GEN_TOKS:-${MAX_GEN_TOKS_DEFAULT}}"
if (( MAX_GEN_TOKS > MAX_MODEL_LEN - 2432 )); then
    echo "ERROR: MAX_GEN_TOKS=${MAX_GEN_TOKS} leaves no room for the longest whitelisted prompt" >&2
    echo "       (2432 padded tokens) inside MAX_MODEL_LEN=${MAX_MODEL_LEN}; the server would reject" >&2
    echo "       every request. Lower it or raise MAX_MODEL_LEN." >&2
    exit 1
fi
THINKING_MODE="${THINKING_MODE:-1}"
TRACE_REGION_SIZE="${TRACE_REGION_SIZE:-6442450944}" # 6 GiB. Measured 2026-07-27: the 48 up-front traces need 3.04 GiB at reveal_pmax=4096 (3 GiB fails, 4 GiB is the floor), so the historical 12 GiB reserved ~8 GiB of DRAM that nothing could allocate. Scale this WITH reveal_pmax - it is not a universal constant. See doc/optimize_perf/bisect_trace_region.sh
RESET_BEFORE="${RESET_BEFORE:-1}"
RESET_AFTER="${RESET_AFTER:-1}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"

case "${THINKING_MODE}" in
    1)
        # Exact aligned lengths with server-side enable_thinking=true.
        # gpqa_diamond_cot_zeroshot, thinking=1. COMPUTED, not guessed:
        #   compute_prefill_whitelist.py --task gpqa_diamond_cot_zeroshot --thinking 1
        # 19 distinct padded lengths, raw min/median/max 103/236/2432. The CoT prompt adds a
        # 576 that the old r1_gpqa_diamond list did not have -- without it that request is
        # rejected at admission and the run aborts mid-eval.
        DEFAULT_PREFILL_WARMUP_LENS="128,160,192,224,256,288,320,352,384,416,448,480,512,544,576,608,672,832,2432"
        SERVER_CHAT_ARGS=(
            --default-chat-template-kwargs '{"enable_thinking":true}'
        )
        # The reasoning parser splits the response on the model's own `<channel|>`
        # delimiter: everything before it (the chain of thought, INCLUDING the \boxed{}
        # answer) goes to `message.reasoning`, and only what follows goes to
        # `message.content`. lm_eval local-chat-completions reads `content`, so with the
        # parser on, every response that closes its thinking channel scores [invalid] --
        # measured on the 2-question smoke, 1/2 vs 2/2, on bit-identical model output.
        #
        # This is new on the vllm-tt-plugin / vLLM 0.24 path only. The parser has always
        # asked for its delimiters (`adjust_request` sets skip_special_tokens=False); the
        # fork ignored that, so `extract_reasoning` never saw `<channel|>` and handed the
        # whole text to `content`. 0.24 honours it. See
        # doc/vllm_integration/plugin_migration_024.md, break 5.
        #
        # Default OFF for scoring: an eval must see the answer. Set
        # REASONING_PARSER=1 to serve the way an API client should be served (clean
        # `content`, thinking in `reasoning`) -- but then do not read the lm_eval score.
        if [[ "${REASONING_PARSER:-0}" == "1" ]]; then
            SERVER_CHAT_ARGS+=(--reasoning-parser diffusion_gemma)
            echo "WARNING: reasoning parser ON -- the \boxed{} answer will land in" >&2
            echo "         message.reasoning, and the lm_eval score (which reads" >&2
            echo "         message.content) will be meaningless." >&2
        fi
        ;;
    0)
        # gpqa_diamond_cot_zeroshot, thinking=0. COMPUTED:
        #   compute_prefill_whitelist.py --task gpqa_diamond_cot_zeroshot --thinking 0
        # 18 distinct padded lengths, raw min/median/max 100/233/2429. Differs from the old
        # r1 list in both directions (no 96, no 640), which is why it must be recomputed and
        # not edited by hand.
        DEFAULT_PREFILL_WARMUP_LENS="128,160,192,224,256,288,320,352,384,416,448,480,512,544,608,672,832,2432"
        SERVER_CHAT_ARGS=()
        ;;
    *)
        echo "ERROR: THINKING_MODE must be 0 or 1, got '${THINKING_MODE}'" >&2
        exit 2
        ;;
esac
PREFILL_WARMUP_LENS="${PREFILL_WARMUP_LENS:-${DEFAULT_PREFILL_WARMUP_LENS}}"

# Pad the low end regardless of the task list. The task-derived lengths above cover the 198 eval
# prompts exactly -- verified byte-exact against the server's own tokenization (63/63 served
# prompt_len at offset 0) -- but they cover ONLY those prompts. Anything else that touches a live
# server (a `curl` smoke test, a readiness ping, a hand-typed question) lands on a short length that
# is not listed, and until the generator learned to reject per-request that meant the whole engine
# died. Measured cost per extra short shape: 0.216-0.286 s of one-time compile, ZERO trace bytes
# (across 710 trace-stat records the capture is one event at a single prompt_len of 32, replayed
# 645 times) and low single-digit MiB of DRAM. ~1 s total to make the server robust to its own
# operators is not a trade worth thinking about.
LOW_END_WARMUP_LENS="32,64,96"
PREFILL_WARMUP_LENS="$(printf '%s\n%s\n' "${LOW_END_WARMUP_LENS//,/$'\n'}" "${PREFILL_WARMUP_LENS//,/$'\n'}" \
    | grep -E '^[0-9]+$' | sort -n -u | paste -sd, -)"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zni/dg_runs/diffusion_gemma/upfront_gpqa/${TIMESTAMP}}"
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

REQUIRED_COMMANDS=(curl flock setsid timeout)
# Only require the reset tool when a reset is actually going to be run. It was required
# unconditionally, which made the whole runner unusable on a host that has no tt-smi on PATH even
# with RESET_BEFORE=0 RESET_AFTER=0 — the two flags that exist precisely to skip it.
if [[ "${RESET_BEFORE}" == "1" || "${RESET_AFTER}" == "1" ]]; then
    REQUIRED_COMMANDS+=("${TT_SMI_BIN}")
fi
for command in "${REQUIRED_COMMANDS[@]}"; do
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

# This is the complete model-level launch contract. Up-front capture intrinsically
# uses reveal masking, eager startup capture, and one-step/window early halt.
# `device` mode draws the full-vocabulary Gumbel noise on device and keeps
# sampling/denoise there (per-position marginals are validated; it is NOT IID
# across canvas positions -- see doc/decision_fidelity/gumbel_position_correlation.md). DiffusionConfig supplies the released temperature,
# entropy, and halt settings. Do not replace it with chunked: QB2's current
# 1024-wide RNG has a known distribution bias, and chunked is not a materialized
# full tensor, so up-front capture rejects it.
#
# DG_VLLM_GUMBEL_MODE defaults to `device`: the on-device permuted-vocab RNG, ~53.6 vs
# ~36.3 tokens/block/s against host (~1.48x), since it removes the per-step host RNG and
# its replicated PCIe copy.
#
# This default moved twice. It corrupted generated text on 2 of 4 matched seeds on
# 2026-07-25 and was reverted to `host`; it is restored now that the cause is fixed. The
# cause was in ttnn.rand: the Blackhole SFPU PRNG is a sliding window over one stream, so
# 64 of the 256 canvas positions received a byte-identical copy of another position's
# noise. See doc/decision_fidelity/degenerate_output_fix.md.
#
# `device` is now the ONLY mode. The `host` IID full-vocabulary reference (and its
# DG_HOST_GUMBEL_PREFETCH overlap) was DELETED on 2026-07-28 after being measured NOT
# to be the TT language-drift cause: it drifts on exactly the same prompts as `device`,
# repairs 0, and costs 1.40x per request. The real cause was the canvas attending
# prefill pad keys, fixed in d0936d4da4f (DG_DENOISE_HIDE_PREFILL_PADS, default on).
setsid env \
    TT_METAL_HOME="${TT_METAL_ROOT}" \
    TT_METAL_RUNTIME_ROOT="${TT_METAL_ROOT}" \
    PYTHONPATH="${PYTHONPATH_VALUE}" \
    MESH_DEVICE=P150x4 \
    DG_CKPT="${DG_CKPT}" \
    DG_UPFRONT_CAPTURE=1 \
    DG_UPFRONT_PREFILL_WARMUP_LENS="${PREFILL_WARMUP_LENS}" \
    DG_DENOISE_REVEAL_PMAX="${MAX_MODEL_LEN}" \
    DG_VLLM_GUMBEL_MODE="${DG_VLLM_GUMBEL_MODE:-device}" \
    DG_TRACE_REGION_SIZE="${TRACE_REGION_SIZE}" \
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
    "${SERVER_CHAT_ARGS[@]}" \
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
# TASK: gpqa_diamond_cot_zeroshot, NOT r1_gpqa_diamond. Changed 2026-07-28 after the strict task
# produced a meaningless gate. r1_gpqa_diamond scores exact_match,none -- a bare \boxed{} extraction
# -- and its prompt does not ask for one, so on the 07-28 full run only 4 of 198 responses contained
# a boxed answer and the score was 6.57%: a number about output FORMAT, not reasoning, and useless
# for separating two configurations. The CoT task fixes both halves: its prompt instructs "put your
# final answer (only the letter A, B, C, or D) within \boxed{}", and its flexible-extract filter
# reads the boxed value first, then explicit answer markers, and only accepts A-D. It is also the
# task the A100 reference used, so the numbers are directly comparable to that 70.71% / 70.20% bar.
# The gated Idavidrein/gpqa diamond split is already in the local HF cache, so HF_HUB_OFFLINE=1 is
# fine; scoring reads exact_match,flexible-extract.
EVAL_TASK="gpqa_diamond_cot_zeroshot"
COMMON_EVAL_ARGS=(
    --tasks "${EVAL_TASK}"
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

echo "Running two-sample smoke..."
# Keyed on ${EVAL_TASK}, not a literal. --samples silently matches nothing when the key is not the
# running task name, so the stale "r1_gpqa_diamond" key made the smoke stage run all 198 questions
# and the run then did it again for the full stage -- double the device time, no error.
SMOKE_SAMPLES="{\"${EVAL_TASK}\":[0,1]}"
"${LM_EVAL}" \
    "${COMMON_EVAL_ARGS[@]}" \
    --output_path "${SMOKE_OUTPUT}" \
    --samples "${SMOKE_SAMPLES}" \
    2>&1 | tee "${OUTPUT_ROOT}/smoke.log"

if [[ "${MODE}" == "full" ]]; then
    SEL_ARGS=()
    if [[ -n "${SAMPLES:-}" ]]; then
        SEL_ARGS=(--samples "${SAMPLES}")
        echo "Running specific GPQA-Diamond samples: ${SAMPLES}"
    elif [[ "${LIMIT:-0}" -gt 0 ]]; then
        SEL_ARGS=(--limit "${LIMIT}")
        echo "Running a ${LIMIT}-sample subset of GPQA-Diamond (LIMIT=${LIMIT})..."
    else
        echo "Running all 198 GPQA-Diamond samples..."
    fi
    "${LM_EVAL}" \
        "${COMMON_EVAL_ARGS[@]}" \
        "${SEL_ARGS[@]}" \
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
