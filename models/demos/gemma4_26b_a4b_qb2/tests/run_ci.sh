#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
set -euo pipefail

cd "${TT_METAL_HOME:?Set TT_METAL_HOME}"
model_dir=models/demos/gemma4_26b_a4b_qb2
results="$PWD/generated/test_reports/gemma4_26b_a4b_qb2"
logs="$PWD/generated/test_logs"
mkdir -p "$results" "$logs"
work=$(mktemp -d)
run_started=$SECONDS
server_pid=
startup_log=

mark_phase() {
    printf 'Gemma4 26B CI phase=%s elapsed_seconds=%s\n' "$1" "$((SECONDS - run_started))" \
        | tee -a "$results/phase-timings.log"
}

stop_server() {
    if [ -z "$server_pid" ]; then return; fi
    kill -- -"$server_pid" 2>/dev/null || true
    for attempt in {1..30}; do
        if ! kill -0 -- -"$server_pid" 2>/dev/null; then break; fi
        sleep 1
    done
    kill -KILL -- -"$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
}

cleanup() {
    local status=$?
    if [ "$status" -ne 0 ] && [ -n "$startup_log" ] && [ -f "$startup_log" ]; then
        python -m models.demos.gemma4_26b_a4b_qb2.tests.startup_status \
            "$startup_log" "$results/server-failure.json" || true
        if [ -f "$results/server-failure.json" ]; then cat "$results/server-failure.json"; fi
    fi
    stop_server
    rm -rf "$work"
    mark_phase cleanup_complete
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mark_phase setup
export EXTRA_MODELS_DIR="$PWD/models/demos" MESH_DEVICE=P300x2
export HF_HOME="${HF_HOME:-/mnt/MLPerf/huggingface}"
export HF_DATASETS_CACHE="$work/datasets" HF_MODULES_CACHE="$work/hf-modules"
export TT_METAL_CACHE="$work/tt-cache" TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
export TT_MESH_GRAPH_DESC_PATH="$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
export VLLM_DEBUG_LOG_API_SERVER_RESPONSE=false GEMMA4_26B_ALLOW_HOST_SAMPLING=1

git clone https://github.com/tenstorrent/vllm-tt-plugin.git "$work/vllm-tt-plugin"
# Keep installation and serving on the validated plugin revision.
git -C "$work/vllm-tt-plugin" checkout 0a76258e14ca7237067ca9a22ec02e257bee570a
git -C "$work/vllm-tt-plugin" rev-parse HEAD > "$results/plugin-revision.txt"
unset VLLM_TT_PLUGIN_REF
pushd "$work/vllm-tt-plugin"
source docs/install-vllm-tt.sh
uv pip install 'pytest>=8,<9'
popd

mark_phase host_tests
python -m pytest "$model_dir/tests/test_precision_policy.py" "$model_dir/tests/test_benchmark.py" \
    "$model_dir/tests/test_prefix_caching.py" "$model_dir/tests/test_seed_continuity.py" \
    --timeout 300 --junitxml "$results/host.xml"
mark_phase host_tests_complete

mark_phase device_numerics
python -m pytest "$model_dir/tests/test_device_numerics.py" \
    --timeout 900 --junitxml "$results/device-numerics.xml"
mark_phase device_numerics_complete

uv venv --python 3.11 "$work/eval-env"
UV_TORCH_BACKEND=cpu uv pip install --python "$work/eval-env/bin/python" \
    -r "$model_dir/tests/requirements-eval.txt"

evaluate() {
    local status=0
    HF_HUB_OFFLINE=0 "$work/eval-env/bin/python" "$model_dir/tests/benchmark.py" "$@" \
        > "$work/evaluator.log" 2>&1 || status=$?
    for summary in "$results"/*/summary.json; do
        if [ -f "$summary" ] && [ ! -f "$summary.reported" ]; then
            CI=true python "$model_dir/tests/report.py" "$summary"
            touch "$summary.reported"
        fi
    done
    if [ "$status" -ne 0 ]; then
        echo "Evaluator failed; see sanitized error.json in reports."
    fi
    return "$status"
}

mark_phase prepare_dataset
evaluate --prepare-only --server-capacity 16 --output-dir "$results/gpqa"

tt_config='{"tt":{"fabric_config":"FABRIC_1D","trace_region_size":1000000000,"sample_on_device_mode":"all","trace_mode":"decode_only","enable_model_warmup":false}}'
capacity=16
startup_log="$work/server_${capacity}.log"
mark_phase "start_server_$capacity"
setsid python -u -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-26B-A4B-it \
    --seed 9472 \
    --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 \
    --tokenizer-revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 \
    --served-model-name google/gemma-4-26B-A4B-it \
    --hf-overrides '{"architectures":["TTGemma4A4BForCausalLM"]}' \
    --host 127.0.0.1 --port 8000 --block-size 32 --max-num-seqs "$capacity" \
    --max-model-len 262144 --max-num-batched-tokens 262144 --max-logprobs -1 \
    --async-scheduling --no-enable-prefix-caching --no-enable-chunked-prefill \
    --enable-auto-tool-choice --tool-call-parser gemma4 \
    --reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking":true}' \
    --no-enable-log-requests --no-enable-log-outputs \
    --additional-config "$tt_config" > "$startup_log" 2>&1 &
server_pid=$!

wait_started=$SECONDS
for ((attempt = 0; SECONDS - wait_started < 1200; attempt++)); do
    if curl --max-time 2 -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        python -m models.demos.gemma4_26b_a4b_qb2.tests.startup_status \
            "$startup_log" "$results/server-startup-$capacity.json"
        cat "$results/server-startup-$capacity.json"
        echo "Gemma4 26B server exited before readiness at capacity=$capacity" >&2
        exit 1
    fi
    if (( attempt % 15 == 0 )); then
        python -m models.demos.gemma4_26b_a4b_qb2.tests.startup_status \
            "$startup_log" "$results/server-startup-$capacity.json"
        cat "$results/server-startup-$capacity.json"
    fi
    sleep 4
done
curl --max-time 2 -fsS http://127.0.0.1:8000/health
mark_phase "server_ready_$capacity"

python -m pytest "$model_dir/tests/test_api.py" \
    --gemma-server-url http://127.0.0.1:8000 --junitxml "$results/api.xml"
mark_phase api_tests_complete

for request_concurrency in 1 8 16; do
    evaluate --mode performance --server-capacity 16 --performance-concurrency "$request_concurrency" \
        --performance-input-lengths 128 --output-dir "$results/concurrency_$request_concurrency"
    mark_phase "performance_complete_$request_concurrency"
done

echo "GPQA coverage: 10/198 Diamond questions for CI regression within the time budget."
evaluate --mode gpqa --server-capacity 16 --output-dir "$results/gpqa"
mark_phase gpqa_complete
