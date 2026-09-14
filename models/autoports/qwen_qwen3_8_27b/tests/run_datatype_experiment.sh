#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export TT_MESH_PASS_THROUGH_THREAD_POOL=${TT_MESH_PASS_THROUGH_THREAD_POOL:-1}
BRINGUP_EXPORTS=$(python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")
eval "$BRINGUP_EXPORTS"
export PYTHONPATH=.:$PYTHONPATH
experiment_name=$1
experiment_module=$2
shift 2
experiment_doc=models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep
trap 'experiment_status=$?; printf "%s\n" "$experiment_status" > "$experiment_doc/$experiment_name.exit_status"' EXIT
if [[ -v QWEN_PRECISION_CONFIG ]]; then
    printf 'QWEN_PRECISION_CONFIG=%q ' "$QWEN_PRECISION_CONFIG" >> "$experiment_doc/commands.log"
else
    printf 'env -u QWEN_PRECISION_CONFIG ' >> "$experiment_doc/commands.log"
fi
printf '%q ' "$0" "$experiment_name" "$experiment_module" "$@" >> "$experiment_doc/commands.log"
printf '\n' >> "$experiment_doc/commands.log"
mkdir -p "$experiment_doc/source_snapshots/$experiment_name"
cp models/autoports/qwen_qwen3_8_27b/tt/*.py "$experiment_doc/source_snapshots/$experiment_name/"
cp models/autoports/qwen_qwen3_8_27b/tests/run_datatype_candidate.py "$experiment_doc/source_snapshots/$experiment_name/"
mkdir -p "$experiment_doc/source_snapshots/$experiment_name/tests"
cp models/autoports/qwen_qwen3_8_27b/tests/*.py "$experiment_doc/source_snapshots/$experiment_name/tests/"
experiment_precision=${QWEN_PRECISION_CONFIG:-$experiment_doc/selected_precision_config.json}
if [[ -f "$experiment_precision" ]]; then
    cp "$experiment_precision" "$experiment_doc/source_snapshots/$experiment_name/precision_config.json"
fi
git rev-parse HEAD > "$experiment_doc/$experiment_name.commit"
python_env/bin/python - "$experiment_doc/$experiment_name.environment.json" <<'PY'
import json, os, sys
keys = ("QWEN_PRECISION_CONFIG", "TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_TRACEBACKS", "TT_METAL_WATCHER", "TT_METAL_WATCHER_NOINLINE", "TT_METAL_WATCHER_DISABLE_ETH", "TT_METAL_FABRIC_OPT_LEVEL", "TT_METAL_DEVICE_PROFILER", "TT_METAL_CACHE", "TT_MESH_PASS_THROUGH_THREAD_POOL")
with open(sys.argv[1], "w") as out:
    json.dump({k: os.environ.get(k) for k in keys}, out, indent=2)
    out.write("\n")
PY
sha256sum models/autoports/qwen_qwen3_8_27b/tt/*.py models/autoports/qwen_qwen3_8_27b/tests/*.py models/common/sampling/{tt_sampling,generator}.py ttnn/ttnn/_ttnn.so build/lib/_ttnncpp.so > "$experiment_doc/$experiment_name.source.sha256"
python_env/bin/python -m "$experiment_module" --output "$experiment_doc/$experiment_name.json" "$@" > "$experiment_doc/$experiment_name.log" 2>&1
