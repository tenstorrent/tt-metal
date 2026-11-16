#!/usr/bin/env bash
# Local serving gate (CI unavailable) for Gemma 4 26B on QB2 (CI recipe settings from models/demos/gemma4_26b_a4b_qb2/tests/run_ci.sh).
#   serve_bench.sh <variant: qb2port|gemma4demo> <out-dir> [modes: perf|gpqa|perf,gpqa]
set -euo pipefail
variant=$1; out=$2; modes=${3:-perf}
W=${GEMMA26_EVAL:-$HOME/gemma26-eval}   # eval-env, datasets, prepared GPQA inputs (see PLAN.md setup)
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source python_env/bin/activate
mkdir -p "$out"
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD EXTRA_MODELS_DIR="$PWD/models/demos" MESH_DEVICE=P300x2
export HF_HOME=/mnt/models/huggingface HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=$W/datasets HF_MODULES_CACHE=$W/hf-modules
export TT_METAL_CACHE=$W/tt-cache-$variant TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
export TT_MESH_GRAPH_DESC_PATH="$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
export VLLM_DEBUG_LOG_API_SERVER_RESPONSE=false GEMMA4_26B_ALLOW_HOST_SAMPLING=1
extra=()
[ "$variant" = qb2port ] && extra=(--hf-overrides '{"architectures":["TTGemma4A4BForCausalLM"]}')
tt_config='{"tt":{"fabric_config":"FABRIC_1D","trace_region_size":1000000000,"sample_on_device_mode":"all","trace_mode":"decode_only","enable_model_warmup":false}}'
started=$SECONDS
setsid python -u -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-26B-A4B-it --seed 9472 \
    --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --tokenizer-revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 \
    --served-model-name google/gemma-4-26B-A4B-it "${extra[@]}" \
    --host 127.0.0.1 --port 8000 --block-size 32 --max-num-seqs 16 \
    --max-model-len 262144 --max-num-batched-tokens 262144 --max-logprobs -1 \
    --async-scheduling --no-enable-prefix-caching --no-enable-chunked-prefill \
    --enable-auto-tool-choice --tool-call-parser gemma4 \
    --reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking":true}' \
    --no-enable-log-requests --no-enable-log-outputs \
    --additional-config "$tt_config" > "$out/server.log" 2>&1 &
pid=$!
stop() { kill -- -"$pid" 2>/dev/null || true; for _ in $(seq 60); do kill -0 -- -"$pid" 2>/dev/null || break; sleep 2; done; kill -KILL -- -"$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
trap stop EXIT
until curl --max-time 2 -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "SERVER_EXITED after $((SECONDS-started))s"; tail -30 "$out/server.log"; exit 1; }
    (( SECONDS - started > 1500 )) && { echo "SERVER_TIMEOUT"; exit 1; }
    sleep 4
done
echo "SERVER_READY after $((SECONDS-started))s"
ev() { HF_HUB_OFFLINE=0 "$W/eval-env/bin/python" models/demos/gemma4_26b_a4b_qb2/tests/benchmark.py "$@"; }
if [[ $modes == *perf* ]]; then
    for c in 1 16; do
        t=$SECONDS
        ev --mode performance --server-capacity 16 --performance-concurrency $c --performance-input-lengths 128 \
            --output-dir "$out/perf_c$c" > "$out/perf_c$c.log" 2>&1 || echo "PERF_C$c FAILED"
        echo "PERF_C$c done in $((SECONDS-t))s"
    done
fi
if [[ $modes == *gpqa* ]]; then
    t=$SECONDS
    mkdir -p "$out/gpqa" && cp -r "$W/results/gpqa/." "$out/gpqa/"
    ev --mode gpqa --server-capacity 16 --output-dir "$out/gpqa" > "$out/gpqa.log" 2>&1 || echo "GPQA FAILED"
    echo "GPQA done in $((SECONDS-t))s"
fi
echo "TOTAL $((SECONDS-started))s"
