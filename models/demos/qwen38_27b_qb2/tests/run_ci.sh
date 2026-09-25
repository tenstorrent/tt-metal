#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
set -euo pipefail

cd "${TT_METAL_HOME:?Set TT_METAL_HOME}"
model_dir=models/demos/qwen38_27b_qb2
results="$PWD/generated/test_reports/qwen38_27b_qb2"
mkdir -p "$results"
work=$(mktemp -d)
server_pid=
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
trap stop_server EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

export EXTRA_MODELS_DIR="$PWD/models/demos" MESH_DEVICE="(1, 4)"
export HF_HOME="${HF_HOME:-/mnt/MLPerf/huggingface}"
export HF_DATASETS_CACHE="$work/datasets" HF_MODULES_CACHE="$work/hf-modules"
export TT_METAL_CACHE="$work/tt-cache" TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
export TT_MESH_GRAPH_DESC_PATH="$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
git clone https://github.com/tenstorrent/vllm-tt-plugin.git "$work/plugin"
git -C "$work/plugin" checkout "${VLLM_TT_PLUGIN_REF:-35090660433d5606957ded97f7130b5cc75f94f7}"
git -C "$work/plugin" rev-parse HEAD > "$results/plugin-revision.txt"
unset VLLM_TT_PLUGIN_REF
pushd "$work/plugin"
source docs/install-vllm-tt.sh
uv pip install 'pytest>=8,<9'
popd
export MODEL_WEIGHTS_DIR
MODEL_WEIGHTS_DIR=$(python -c 'from models.demos.qwen38_27b_qb2.tt.model import checkpoint_path; print(checkpoint_path())')
python -m pytest "$model_dir/tests/unit" "$model_dir/tests/vllm" "$model_dir/tests/test_benchmark.py" \
    --timeout 300 --junitxml "$results/host.xml"
python -m pytest "$model_dir/tests/test_decode_conv.py" --timeout 300 --junitxml "$results/device.xml"

export QWEN_DECODE_BUCKETS=1 QWEN_COMPACT_DECODE_RESIDUAL=1 QWEN_COMPACT_DECODE_MLP=1
export QWEN_BATCHED_DECODE_ROPE=1 QWEN_COMPACT_DECODE_ATTENTION=1
export QWEN_VLLM_KV_POOL_TOKENS=1050592 QWEN_BATCHED_PREFILL=1
export QWEN_PREFILL_RESIDUAL_LAYOUT=sharded_replicated_norm QWEN_PREFILL_BATCHED_HEAD=1
export QWEN_PREFILL_SKIP_INTERMEDIATE_HEAD=1 QWEN_PREFILL_STARTUP_WARMUP=1
export QWEN_VLLM_HOST_COMPATIBILITY=1 VLLM_DEBUG_LOG_API_SERVER_RESPONSE=false

uv venv --python 3.11 "$work/eval-env"
UV_TORCH_BACKEND=cpu uv pip install --python "$work/eval-env/bin/python" -r "$model_dir/tests/requirements-eval.txt"
# Evaluation logs can contain gated dataset content; only sanitized JSON is uploaded.
evaluate() {
    local status=0
    HF_HUB_OFFLINE=0 HF_HUB_CACHE="$work/eval-hub" "$work/eval-env/bin/python" "$model_dir/tests/benchmark.py" "$@" \
        > "$work/evaluator.log" 2>&1 || status=$?
    # Convert completed measurements even when the accuracy gate fails.
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
evaluate --prepare-only --server-capacity 16 --output-dir "$results/gpqa"
tt_config='{"tt":{"fabric_config":"FABRIC_1D_RING","fabric_max_packet_payload_size_bytes":8192,"l1_small_size":24576,"sample_on_device_mode":"all","trace_mode":"decode_only","trace_region_size":134217728}}'
for capacity in 1 8 16; do
    echo "Starting Qwen3.8 server capacity=$capacity"
    setsid python -u -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_WEIGHTS_DIR" --served-model-name Qwen/Qwen3.8-27B \
        --hf-overrides '{"architectures":["TTQwen38ForCausalLM"]}' \
        --host 127.0.0.1 --port 8000 --block-size 32 --max-num-seqs "$capacity" \
        --max-model-len 262144 --max-num-batched-tokens 262144 --max-logprobs -1 \
        --async-scheduling --no-enable-prefix-caching --no-enable-chunked-prefill \
        --no-enable-log-requests --no-enable-log-outputs \
        --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
        --additional-config "$tt_config" > "$work/server_$capacity.log" 2>&1 &
    server_pid=$!
    wait_started=$SECONDS
    while ! curl --max-time 2 -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; do
        kill -0 "$server_pid"
        if (( SECONDS - wait_started > 1200 )); then
            echo "Qwen3.8 readiness deadline exceeded at capacity=$capacity"
            exit 1
        fi
        sleep 4
    done
    evaluate --mode performance --server-capacity "$capacity" --performance-input-lengths 128 1024 \
        --output-dir "$results/capacity_$capacity"
    if [ "$capacity" = 16 ]; then
        evaluate --mode gpqa --server-capacity 16 --output-dir "$results/gpqa"
        pushd "$work/plugin"
        python -m pytest --confcutdir=. \
            tests/tt/test_seeding_and_variety.py::TestSeedingAndVariety::test_same_seeds_reproduce_across_batches \
            tests/tt/test_request_isolation.py::TestBatchIsolation::test_mixed_params_batch \
            tests/tt/test_host_only_params.py::TestHostOnlyParameters::test_allowed_token_ids \
            tests/tt/test_tt_penalties.py::TestFrequencyPenalty::test_frequency_penalty_mixed_batch \
            'tests/tt/test_logprobs.py::TestLogprobs::test_logprobs[5-1]' \
            --tt-server-url http://127.0.0.1:8000 --tt-model-name Qwen/Qwen3.8-27B \
            --tt-max-num-seqs 16 --junitxml "$results/api.xml"
        popd
    fi
    stop_server
done
