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
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
export TT_MESH_GRAPH_DESC_PATH="$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
export TT_METAL_CACHE="$results/kernel-cache"
tt_config='{"tt":{"fabric_config":"FABRIC_1D","trace_region_size":1000000000,"sample_on_device_mode":"all","trace_mode":"decode_only","enable_model_warmup":false}}'
setsid python -u -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-26B-A4B-it --seed 9472 \
    --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 \
    --tokenizer-revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 \
    --hf-overrides '{"architectures":["TTGemma4A4BForCausalLM"]}' \
    --host 127.0.0.1 --port 8000 --block-size 32 --max-num-seqs 1 \
    --max-model-len 262144 --max-num-batched-tokens 262144 --max-logprobs -1 \
    --async-scheduling --enable-prefix-caching --no-enable-chunked-prefill \
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
status=0
timeout 1800 python -m pytest "$model_dir/tests/test_prefix_caching.py" -k live \
    --gemma-server-url http://127.0.0.1:8000 --junitxml "$results/live.xml" -x -v || status=$?
if [ "$status" -eq 124 ]; then
    timeout 120 python tools/tt-triage.py --llm-output \
        --llm-output-path "$results/triage.txt" --triage-summary-path "$results/triage-summary.txt" || true
fi
curl --max-time 10 -fsS http://127.0.0.1:8000/metrics > "$results/metrics.txt" || true
exit "$status"
