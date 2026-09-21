#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
shopt -s nullglob

: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
: "${PREFILL_SUMMARIES:?PREFILL_SUMMARIES must be shared with the CI runner}"
export PYTHONPATH="$TT_METAL_HOME"
export PYTHONUNBUFFERED=1
cd "$TT_METAL_HOME"

artifacts="$PREFILL_SUMMARIES/gemma4_mock256k"
mkdir -p "$artifacts"
scratch=$(mktemp -d /tmp/gemma4-mock256k.XXXXXX)
log_pid=""

cleanup() {
    local status=$?
    trap - EXIT
    if [ -n "$log_pid" ]; then
        kill "$log_pid" 2>/dev/null || true
        wait "$log_pid" 2>/dev/null || true
    fi
    for file in "$scratch"/pytest/test*/{runner.log,producer.log,gemma4_slot0.json,device_map.json}; do
        cp "$file" "$artifacts/" || status=1
    done
    rm -rf "$scratch"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 - <<'PYTHON'
import json
import os
from pathlib import Path

from models.common.weight_cache import weight_cache_is_complete
from models.demos.gemma4_d_p.tt.common import weight_cache_identity
from models.demos.gemma4_d_p.tt.precision import Gemma4Precision
from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter, Gemma4ServiceConfig

adapter = Gemma4PrefillAdapter()
trace = Path(os.environ["PREFILL_TRACE_DIR"])
metadata = json.loads((trace / "metadata.json").read_text())
if metadata["layout"] != "gemma4_kv_heads_v1":
    raise RuntimeError(f"CI requires a prepared GPU capture: {trace}")
for layer in range(Gemma4ServiceConfig.NUM_LAYERS):
    with (trace / "kv_cache" / f"layer_{layer}.safetensors").open("rb") as tensor_file:
        if not tensor_file.read(8):
            raise RuntimeError(f"Empty GPU capture for layer {layer}")
adapter.load_hf_config()
cache = adapter.weight_cache_path(Gemma4ServiceConfig.MESH_SHAPE)
identity = weight_cache_identity(
    adapter.hf_model_id,
    Gemma4ServiceConfig.NUM_LAYERS,
    Gemma4ServiceConfig.MESH_SHAPE,
    Gemma4Precision.load(adapter.hf_model_id),
)
if not weight_cache_is_complete(cache, **identity):
    raise RuntimeError(f"CI requires a complete, compatible TT weight cache: {cache}")
print(f"Preflight passed: GPU capture={trace}, TT cache={cache}", flush=True)
PYTHON

stream_runner_log() {
    while true; do
        for log in "$scratch"/pytest/test*/runner.log; do
            exec tail -n +1 -f -s 0.2 "$log"
        done
        sleep 1
    done
}
stream_runner_log &
log_pid=$!

timeout --kill-after=30s 25m python3 -m pytest \
    'models/demos/gemma4_d_p/tests/test_prefill_migration.py::test_prefill_migration[mock-256k]' \
    -sv --tb=short --basetemp="$scratch/pytest" --junitxml="$artifacts/junit.xml"

reports=("$scratch"/pytest/test*/gemma4_slot0.json)
if [ "${#reports[@]}" -eq 0 ]; then
    echo "Missing PCC report: mock-256k must execute successfully, not skip" >&2
    exit 1
fi
