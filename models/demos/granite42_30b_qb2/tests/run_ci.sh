#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
set -euo pipefail
umask 077

cd "${TT_METAL_HOME:?Set TT_METAL_HOME}"
model_dir=models/demos/granite42_30b_qb2
results="$PWD/generated/test_reports/granite42_30b_qb2"
mkdir -p "$results" generated/test_logs
work=$(mktemp -d)
server_pid=
run_started=$SECONDS
mark_phase() {
    printf 'Granite CI phase=%s elapsed_seconds=%s\n' "$1" "$((SECONDS - run_started))" \
        | tee -a "$results/phase-timings.log"
}
cleanup() {
    local status=$?
    if [ "$status" -ne 0 ] && [ -f "$work/server.log" ]; then
        # Public diagnostics contain stack locations and types, not request bodies.
        python - "$work/server.log" "$results/server-failure.json" <<'PY'
import json
import re
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="replace")
frames = re.findall(r'File "[^"\n]*[/\\]([\w.]+\.py)", line (\d+), in ([\w<>]+)', text)
result = {
    "traceback_frames": [{"file": f, "line": int(n), "function": fn} for f, n, fn in frames[-40:]],
    "exception_types": sorted(set(re.findall(r"\b([A-Za-z_]\w*(?:Error|Exception)):", text))),
    "loaded_layers": [int(n) for n in re.findall(r"GRANITE_LAYER_READY (\d+)/", text)][-1:],
}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
PY
        cat "$results/server-failure.json"
    fi
    if [ -n "$server_pid" ]; then
        kill -- -"$server_pid" 2>/dev/null || true
        for attempt in {1..30}; do
            if ! kill -0 -- -"$server_pid" 2>/dev/null; then break; fi
            sleep 1
        done
        kill -KILL -- -"$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
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
export VLLM_DEBUG_LOG_API_SERVER_RESPONSE=false

git clone https://github.com/tenstorrent/vllm-tt-plugin.git "$work/plugin"
plugin_ref="${VLLM_TT_PLUGIN_REF:-a2500313c990b129d78b74dadca312ed696caca0}"
git -C "$work/plugin" checkout "$plugin_ref"
git -C "$work/plugin" rev-parse HEAD > "$results/plugin-revision.txt"
unset VLLM_TT_PLUGIN_REF
pushd "$work/plugin"
source docs/install-vllm-tt.sh
uv pip install 'pytest>=8,<9'
popd

export HF_MODEL
HF_MODEL=$(python -c 'from models.demos.granite42_30b_qb2.tt.checkpoint import resolve_checkpoint; print(resolve_checkpoint())')
mark_phase host_tests
python -m pytest "$model_dir/tests/test_host_contract.py" "$model_dir/tests/test_benchmark.py" \
    "$model_dir/tests/test_sampler_contract.py" \
    --timeout 300 --junitxml "$results/host.xml"
mark_phase host_tests_complete
mark_phase device_tests
python -m pytest "$model_dir/tests/test_token_history.py" "$model_dir/tests/test_external_cache.py" \
    "$model_dir/tests/test_decoder.py" \
    --timeout 900 --junitxml "$results/device.xml"
mark_phase device_tests_complete

uv venv --python 3.11 "$work/eval-env"
UV_TORCH_BACKEND=cpu uv pip install --python "$work/eval-env/bin/python" -r "$model_dir/tests/requirements-eval.txt"
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
    if [ "$status" -ne 0 ]; then echo "Evaluator failed; see error.json in the reports."; fi
    return "$status"
}
mark_phase prepare_dataset
evaluate --prepare-only --server-capacity 16 --output-dir "$results/gpqa"

mark_phase server_start
setsid env VLLM_SERVER_DEV_MODE=1 python -u -m vllm.entrypoints.openai.api_server \
    --model "$HF_MODEL" --served-model-name ibm-granite/granite-4.2-30b \
    --revision 9e668ce1c538387ef24d3644e9b0606647762636 \
    --tokenizer-revision 9e668ce1c538387ef24d3644e9b0606647762636 \
    --hf-overrides '{"architectures":["TTGraniteForCausalLM"]}' \
    --host 127.0.0.1 --port 8000 --block-size 32 --max-num-seqs 16 \
    --max-model-len 131072 --max-num-batched-tokens 1024 \
    --enable-chunked-prefill --enable-prefix-caching --enable-prompt-tokens-details --async-scheduling \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --default-chat-template-kwargs '{"enable_thinking":true,"low_effort":false}' \
    --no-enable-log-requests --no-enable-log-outputs \
    --additional-config '{"tt":{"fabric_config":"FABRIC_1D_RING","sample_on_device_mode":"all","trace_region_size":134217728}}' \
    > "$work/server.log" 2>&1 &
server_pid=$!
wait_started=$SECONDS
while ! curl --max-time 2 -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; do
    kill -0 "$server_pid"
    if (( SECONDS - wait_started > 1200 )); then
        echo 'Granite server readiness deadline exceeded'
        exit 1
    fi
    sleep 4
done
mark_phase server_ready
GRANITE_SERVER_URL=http://127.0.0.1:8000 python -m pytest "$model_dir/tests/test_api.py" "$model_dir/tests/test_prefix_cache.py" \
    --timeout 900 --junitxml "$results/api.xml"
mark_phase api_complete
for concurrency in 1 8 16; do
    evaluate --mode performance --server-capacity 16 --performance-concurrency "$concurrency" \
        --performance-input-lengths 128 --output-dir "$results/concurrency_$concurrency"
done
mark_phase performance_complete
evaluate --mode gpqa --server-capacity 16 --output-dir "$results/gpqa"
mark_phase gpqa_complete
