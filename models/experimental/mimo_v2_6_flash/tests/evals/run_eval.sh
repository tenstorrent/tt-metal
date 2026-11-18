#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# One benchmark against one OpenAI-compatible endpoint, OpenRouter-shaped requests, one results JSON.
#
#   run_eval.sh <bench> <base_url> <model> [--limit N] [--concurrency C] [--out DIR] [--target vllm|openrouter]
#               [--provider slug[,slug]] [--thinking on|off|default] [--max-tokens N] [--epochs N]
#               [--user-model M] [--tag T] [--max-retries N] [--dry-run]
#
# Benches (harness, task, defaults):
#   mmmu_pro          lmms-eval  mmmu_pro_standard_cot (1730 q, 10 options, CoT)   thinking off
#   mmmu_pro_aa       lmms-eval  mmmu_pro_aa: Artificial Analysis's protocol (1730, AA prompt + regex, thinking on);
#                                the summary prints TT vs AA's 73.06%, the delta and 95% intervals
#   ocrbench          lmms-eval  ocrbench (1000)                                    thinking off
#   videomme_short    lmms-eval  videomme_short (900 q / 300 videos, no subtitles)  thinking off
#   mmau_mini         lmms-eval  mmau_test_mini (1000)                              thinking off
#   librispeech_clean lmms-eval  librispeech_test_clean (2620, WER)                 thinking off
#   vgi_bench         lmms-eval  vgi_bench: Seldon's VGI-Bench public split (439 q, long videos by URL, Seldon's
#                                prompt + parser, temperature 0, no max_tokens / reasoning switch: endpoint defaults)
#   gpqa_diamond      inspect    gpqa_shuffled.py (198 q x 10 epochs, T=0.5, per-question shuffle = OpenRouter's)
#   gpqa_openbench    openbench  gpqa_diamond (openbench as published: fixed permutation)
#   tau2_airline      openbench  tau_bench_airline (50 tasks, T=0; user simulator --user-model, needs its key)
# The text benches send no reasoning switch (the model's default, thinking on), as OpenRouter's harness does.
#
# Targets: "vllm" (our server; reasoning switch = chat_template_kwargs.enable_thinking, API key from API_KEY or none)
# or "openrouter" (base_url https://openrouter.ai/api/v1, OPENROUTER_API_KEY; --provider pins providers with
# allow_fallbacks false; reasoning switch = OpenRouter's reasoning.enabled). A reference run is the same command
# with the OpenRouter target, e.g.
#   run_eval.sh mmau_mini https://openrouter.ai/api/v1 xiaomi/mimo-v2.6-flash --target openrouter --provider gmicloud
#
# Resume: rerun the same command. lmms-eval benches answer finished requests from the journal in the output dir;
# inspect / openbench benches continue an unfinished log with `inspect eval-retry`.
# --dry-run: no network to the endpoint. lmms-eval benches capture and shape-check every request in-process
# (<out>/journal.dryrun.jsonl) and answer "A"; inspect benches use inspect's mockllm model. Datasets are still loaded.
#
# Output: <out>/summary.json (score, n, failures by status, tokens, wall time, versions, command; every finish also
# keeps a timestamped copy summary.<UTC>.json), harness logs beside; lmms-eval benches also keep every request (media
# by reference) and full raw response in <out>/journal.raw.jsonl. --max-retries: retries of 5xx / 408 / 429 /
# connection errors per request (default 3; full-jitter exponential backoff capped at 120 s).
# Env: MM_EVALENV (venv, default /data/ttuser-agentx/mm/evalenv), MM_DATASETS (default /data/ttuser-agentx/mm/datasets;
# HF_HOME defaults to $MM_DATASETS/hf), MM_RESULTS (default /data/ttuser-agentx/mm/evals/results), MM_HF_ENV (a file
# exporting HF_TOKEN, default /data/ttuser-agentx/.hf_env).

set -euo pipefail

usage() { sed -n '4,43p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[ $# -ge 3 ] || usage

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$1 BASE_URL=${2%/} MODEL=$3
shift 3
LIMIT="" CONC=16 OUT="" TARGET="" PROVIDER="" THINKING="" MAX_TOKENS="" EPOCHS="" USER_MODEL="" TAG="" DRY=0 RETRIES=""
while [ $# -gt 0 ]; do
    case $1 in
        --limit) LIMIT=$2; shift 2 ;;
        --concurrency) CONC=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --target) TARGET=$2; shift 2 ;;
        --provider) PROVIDER=$2; shift 2 ;;
        --thinking) THINKING=$2; shift 2 ;;
        --max-tokens) MAX_TOKENS=$2; shift 2 ;;
        --epochs) EPOCHS=$2; shift 2 ;;
        --user-model) USER_MODEL=$2; shift 2 ;;
        --tag) TAG=$2; shift 2 ;;
        --max-retries) RETRIES=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown option $1" >&2; usage ;;
    esac
done

EVALENV=${MM_EVALENV:-/data/ttuser-agentx/mm/evalenv}
MM_DATASETS=${MM_DATASETS:-/data/ttuser-agentx/mm/datasets}
export HF_HOME=${HF_HOME:-$MM_DATASETS/hf}
export TAU2_DATA_DIR=${TAU2_DATA_DIR:-$MM_DATASETS/tau2}
HF_ENV=${MM_HF_ENV:-/data/ttuser-agentx/.hf_env}
# shellcheck disable=SC1090
[ -f "$HF_ENV" ] && source "$HF_ENV"
# lmms-eval logs HF_TOKEN verbatim (its evaluation-tracker args), so the token goes to the harnesses as a file.
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$MM_DATASETS/.hf_token}
    if [ ! -s "$HF_TOKEN_PATH" ]; then (umask 077; printf '%s' "$HF_TOKEN" > "$HF_TOKEN_PATH"); fi
    unset HF_TOKEN
fi
PY=$EVALENV/bin/python

if [ -z "$TARGET" ]; then
    case $BASE_URL in *openrouter.ai*) TARGET=openrouter ;; *) TARGET=vllm ;; esac
fi
if [ "$TARGET" = openrouter ]; then
    KEY_ENV=OPENROUTER_API_KEY
    [ "$DRY" = 1 ] || [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is not set" >&2; exit 2; }
else
    KEY_ENV=API_KEY
    export API_KEY=${API_KEY:-none}
fi

SAFE_MODEL=$(echo "$MODEL" | tr '/:' '__')
SUFFIX=$TARGET${PROVIDER:+-${PROVIDER//,/+}}${TAG:+-$TAG}${LIMIT:+-limit$LIMIT}
[ "$DRY" = 1 ] && SUFFIX=$SUFFIX-dryrun
OUT=${OUT:-${MM_RESULTS:-/data/ttuser-agentx/mm/evals/results}/$BENCH/$SAFE_MODEL-$SUFFIX}
mkdir -p "$OUT"
START=$(date +%s)
CMDLINE="$0 $BENCH $BASE_URL $MODEL $*"

run_lmms() {  # $1 task, $2 default thinking
    local task=$1 think=${THINKING:-$2}
    local args="model_version=$MODEL,base_url=$BASE_URL,api_key_env=$KEY_ENV,target=$TARGET,thinking=$think"
    args="$args,num_concurrent=$CONC,journal=$OUT/journal.jsonl,dry_run=$DRY"
    [ -n "$PROVIDER" ] && args="$args,provider=${PROVIDER//,/+}"
    [ -n "$MAX_TOKENS" ] && args="$args,max_new_tokens=$MAX_TOKENS"
    [ -n "$RETRIES" ] && args="$args,max_retries=$RETRIES"
    # MM_TEMPERATURE / MM_TOP_P override the task's sampling for every request (e.g. 1.0 / 0.95, the model card's
    # recommendation, which Xiaomi's endpoint applies even when a request asks for temperature 0).
    [ -n "${MM_TEMPERATURE:-}" ] && args="$args,temperature=$MM_TEMPERATURE"
    [ -n "${MM_TOP_P:-}" ] && args="$args,top_p=$MM_TOP_P"
    local extra=()
    [ -n "$LIMIT" ] && extra+=(--limit "$LIMIT")
    # --include_path although the plugin also lists its tasks: lmms-eval 0.7.4 adds a plugin's task dir only after
    # it has built the task index.
    LMMS_EVAL_PLUGINS=mimo_lmms PYTHONPATH="$HERE/lmms_plugin${PYTHONPATH:+:$PYTHONPATH}" \
        "$PY" -m lmms_eval --model mimo_openrouter --model_args "$args" --tasks "$task" \
        --include_path "$HERE/lmms_plugin/mimo_lmms/tasks" --batch_size 1 --output_path "$OUT/lmms" --log_samples \
        --seed 0 "${extra[@]}" 2>&1 | tee -a "$OUT/run.log"
    "$PY" "$HERE/summarize_results.py" --harness lmms --bench "$BENCH" --task "$task" --out "$OUT" \
        --model "$MODEL" --base-url "$BASE_URL" --target "$TARGET" --thinking "$think" --start "$START" \
        --cmd "$CMDLINE"
    cp "$OUT/summary.json" "$OUT/summary.$(date -u +%Y%m%dT%H%M%SZ).json" 2>/dev/null || true
}

inspect_model() {  # the inspect-ai model string for the target
    if [ "$DRY" = 1 ]; then
        echo "mockllm/model"
    elif [ "$TARGET" = openrouter ]; then
        echo "openrouter/$MODEL"
    else
        # inspect's OpenAI-compatible provider: openai-api/<service>/<model> reads <SERVICE>_BASE_URL / _API_KEY.
        export TT_BASE_URL=$BASE_URL TT_API_KEY=$API_KEY
        echo "openai-api/tt/$MODEL"
    fi
}

run_inspect() {  # $1 "inspect" | "bench", $2 task spec, then extra args
    local tool=$1 spec=$2
    shift 2
    local model margs=()
    model=$(inspect_model)
    if [ "$TARGET" = openrouter ] && [ -n "$PROVIDER" ] && [ "$DRY" = 0 ]; then
        local order
        order=$(echo "$PROVIDER" | sed 's/[^,][^,]*/"&"/g')
        margs+=(-M "provider={\"order\": [$order], \"allow_fallbacks\": false}")
    fi
    local common=(--model "$model" --max-connections "$CONC" --log-dir "$OUT/logs")
    [ -n "$LIMIT" ] && common+=(--limit "$LIMIT")
    [ -n "$MAX_TOKENS" ] && common+=(--max-tokens "$MAX_TOKENS")
    local prev
    prev=$("$PY" "$HERE/summarize_results.py" --unfinished-log "$OUT/logs" || true)
    if [ -n "$prev" ]; then
        echo "resuming unfinished log $prev" | tee -a "$OUT/run.log"
        "$EVALENV/bin/inspect" eval-retry "$prev" --log-dir "$OUT/logs" --max-connections "$CONC" 2>&1 | tee -a "$OUT/run.log"
    elif [ "$tool" = inspect ]; then
        "$EVALENV/bin/inspect" eval "$spec" "${common[@]}" "${margs[@]}" "$@" --display plain 2>&1 | tee -a "$OUT/run.log"
    else
        "$EVALENV/bin/bench" eval "$spec" "${common[@]}" "${margs[@]}" "$@" --display plain 2>&1 | tee -a "$OUT/run.log"
    fi
    "$PY" "$HERE/summarize_results.py" --harness inspect --bench "$BENCH" --task "$spec" --out "$OUT" \
        --model "$MODEL" --base-url "$BASE_URL" --target "$TARGET" --thinking default --start "$START" --cmd "$CMDLINE"
}

case $BENCH in
    mmmu_pro) run_lmms mmmu_pro_standard_cot off ;;
    ocrbench) run_lmms ocrbench off ;;
    videomme_short)
        [ -d "$HF_HOME/videomme/data" ] || { echo "run prefetch_datasets.py videomme --durations short first" >&2; exit 2; }
        run_lmms videomme_short off ;;
    mmmu_pro_aa) run_lmms mmmu_pro_aa on ;;
    mmau_mini) run_lmms mmau_test_mini off ;;
    librispeech_clean) run_lmms librispeech_test_clean off ;;
    vgi_bench) run_lmms vgi_bench default ;;
    gpqa_diamond) run_inspect inspect "$HERE/openbench_tasks/gpqa_shuffled.py@gpqa_diamond_shuffled" -T "epochs=${EPOCHS:-10}" ;;
    gpqa_openbench) run_inspect bench gpqa_diamond --epochs "${EPOCHS:-10}" ;;
    tau2_airline)
        if [ -z "$USER_MODEL" ]; then
            if [ "$DRY" = 1 ]; then USER_MODEL=mockllm/model
            elif [ -n "${OPENROUTER_API_KEY:-}" ]; then USER_MODEL=openrouter/openai/gpt-4.1
            elif [ -n "${OPENAI_API_KEY:-}" ]; then USER_MODEL=openai/gpt-4.1
            else echo "tau2_airline needs a user-simulator model: set OPENROUTER_API_KEY or OPENAI_API_KEY, or --user-model" >&2; exit 2
            fi
        fi
        run_inspect bench tau_bench_airline --alpha -T "user_model=$USER_MODEL" ${EPOCHS:+--epochs "$EPOCHS"} ;;
    *) echo "unknown bench $BENCH" >&2; usage ;;
esac
echo "summary: $OUT/summary.json"
