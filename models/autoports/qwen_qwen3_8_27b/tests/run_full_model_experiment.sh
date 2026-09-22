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
experiment_doc=models/autoports/qwen_qwen3_8_27b/doc/full_model
trap 'experiment_status=$?; printf "%s\n" "$experiment_status" > "$experiment_doc/$experiment_name.exit_status"' EXIT
printf '%q ' "$0" "$experiment_name" "$experiment_module" "$@" >> "$experiment_doc/commands.log"
printf '\n' >> "$experiment_doc/commands.log"
git rev-parse HEAD > "$experiment_doc/$experiment_name.commit"
python_env/bin/python - "$experiment_doc/$experiment_name.environment.json" <<'PY'
import json, os, sys
keys = ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_TRACEBACKS", "TT_METAL_WATCHER", "TT_METAL_WATCHER_NOINLINE", "TT_METAL_WATCHER_DISABLE_ETH", "TT_METAL_FABRIC_OPT_LEVEL", "TT_METAL_DEVICE_PROFILER", "TT_METAL_CACHE", "TT_MESH_PASS_THROUGH_THREAD_POOL")
with open(sys.argv[1], "w") as out:
    json.dump({k: os.environ.get(k) for k in keys}, out, indent=2)
    out.write("\n")
PY
sha256sum models/autoports/qwen_qwen3_8_27b/tt/*.py models/autoports/qwen_qwen3_8_27b/tests/full_model*.py models/autoports/qwen_qwen3_8_27b/tests/run_full_model.py models/common/sampling/{tt_sampling,generator}.py ttnn/ttnn/_ttnn.so build/lib/_ttnncpp.so > "$experiment_doc/$experiment_name.source.sha256"
python_env/bin/python -m "$experiment_module" --output "$experiment_doc/$experiment_name.json" "$@" > "$experiment_doc/$experiment_name.log" 2>&1
