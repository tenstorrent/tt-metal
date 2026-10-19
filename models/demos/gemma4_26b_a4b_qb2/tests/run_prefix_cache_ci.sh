#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export OMP_NUM_THREADS=8
cd "${TT_METAL_HOME:?}"
model_dir=models/demos/gemma4_26b_a4b_qb2
results=${GEMMA4_PREFIX_RESULTS:?}
mkdir -p "$results"
server_pid=
cleanup() {
    if [ -n "$server_pid" ]; then
        kill -- -"$server_pid" 2>/dev/null || true
        for attempt in {1..20}; do
            if ! kill -0 -- -"$server_pid" 2>/dev/null; then break; fi
            sleep 1
        done
        kill -KILL -- -"$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python -m pytest "$model_dir/tests/test_prefix_caching.py" -k 'not live' --junitxml "$results/host.xml" -q
export EXTRA_MODELS_DIR="$PWD/models/demos" MESH_DEVICE=P300x2
export GEMMA4_26B_ALLOW_HOST_SAMPLING=1 GEMMA4_PREFIX_CACHING_EXPERIMENT=1
export GEMMA4_PREFIX_DIAGNOSE=1
export GEMMA4_PREFIX_SHARED_BATCH=0 GEMMA4_PREFIX_TRACE_SUFFIX=0
case "${GEMMA4_PREFIX_SUFFIX_MODE:-serial}" in
    serial) ;;
    shared) export GEMMA4_PREFIX_SHARED_BATCH=1 ;;
    trace) export GEMMA4_PREFIX_TRACE_SUFFIX=1 ;;
    *) echo 'Unsupported suffix experiment mode'; exit 2 ;;
esac
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
export TT_MESH_GRAPH_DESC_PATH="$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
export TT_METAL_CACHE="$results/kernel-cache"
tt_config='{"tt":{"fabric_config":"FABRIC_1D","trace_region_size":1000000000,"sample_on_device_mode":"all","trace_mode":"decode_only","enable_model_warmup":false}}'
max_context=262144
server_host=127.0.0.1
cache_flag=--enable-prefix-caching
if [ "${GEMMA4_PREFIX_CACHE_MODE:-on}" = off ]; then cache_flag=--no-enable-prefix-caching; fi
serving_args=()
test_file="$model_dir/tests/test_prefix_caching.py"
test_selection=live_decode_contract
case "${GEMMA4_PREFIX_SUITE:-boundary}" in
    boundary) ;;
    lifetime) max_context=4096; test_file="$model_dir/tests/test_prefix_cache_lifetime.py"; test_selection=live_lifetime ;;
    growing) test_file="$model_dir/tests/test_prefix_cache_lifetime.py"; test_selection=live_growing ;;
    tracecheck)
        max_context=4096
        export GEMMA4_PREFIX_DIAGNOSE=0
        export TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1
        export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
        test_file="$model_dir/tests/test_prefix_cache_lifetime.py"
        test_selection=live_trace_chat_transitions
        ;;
    traceperf)
        max_context=4096
        export GEMMA4_PREFIX_DIAGNOSE=0
        test_file="$model_dir/tests/test_prefix_cache_lifetime.py"
        test_selection=live_trace_chat_transitions
        ;;
    agentic)
        server_host=0.0.0.0
        export GEMMA4_PREFIX_DIAGNOSE=0
        serving_args=(--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking":true}')
        ;;
    *) echo 'Unsupported prefix experiment suite'; exit 2 ;;
esac
setsid python -u -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-26B-A4B-it --seed 9472 \
    --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 \
    --tokenizer-revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 \
    --hf-overrides '{"architectures":["TTGemma4A4BForCausalLM"]}' \
    --host "$server_host" --port 8000 --block-size 32 --max-num-seqs 1 \
    --max-model-len "$max_context" --max-num-batched-tokens "$max_context" --max-logprobs -1 \
    --async-scheduling "$cache_flag" --no-enable-chunked-prefill "${serving_args[@]}" \
    --no-enable-log-requests --no-enable-log-outputs \
    --additional-config "$tt_config" > "$results/server.log" 2>&1 &
server_pid=$!
started=$SECONDS
while ! curl --max-time 2 -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null || (( SECONDS - started > 1200 )); then
        tail -100 "$results/server.log"
        if kill -0 "$server_pid" 2>/dev/null; then
            timeout 120 python tools/tt-triage.py --llm-output \
                --llm-output-path "$results/startup-triage.txt" \
                --triage-summary-path "$results/startup-triage-summary.txt" || true
        fi
        exit 1
    fi
    sleep 5
done
if [ "${GEMMA4_PREFIX_SUITE:-boundary}" = agentic ]; then
    touch "$results/server-ready.txt"
    wait "$server_pid"
    exit $?
fi
status=0
timeout 1800 python -m pytest "$test_file" -k "$test_selection" \
    --gemma-server-url http://127.0.0.1:8000 --junitxml "$results/live.xml" -x -v || status=$?
if [ "$status" -eq 124 ]; then
    timeout 120 python tools/tt-triage.py --llm-output \
        --llm-output-path "$results/triage.txt" --triage-summary-path "$results/triage-summary.txt" || true
fi
curl --max-time 10 -fsS http://127.0.0.1:8000/metrics > "$results/metrics.txt" || true
exit "$status"
