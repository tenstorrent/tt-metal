#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
QWEN_TASK_ROOT=${1:?Provide the isolated runtime directory}
QWEN_MODEL_SOURCE=${2:?Provide the pinned candidate source checkout}
QWEN_VALIDATION_DIR=${3:?Provide a new validation directory}
mkdir "$QWEN_VALIDATION_DIR"
export PATH="$QWEN_TASK_ROOT/python_env/bin:$PATH"
export TT_METAL_HOME="$QWEN_TASK_ROOT/metal"
export PYTHONPATH="$QWEN_MODEL_SOURCE:$TT_METAL_HOME:$TT_METAL_HOME/tools"
export LD_LIBRARY_PATH="$QWEN_TASK_ROOT/metal-install/lib:$QWEN_TASK_ROOT/metal-build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_CACHE="$QWEN_TASK_ROOT/jit-cache-metal-galaxy"
export MODEL_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR:?Set the pinned local checkpoint directory}
export ARCH_NAME=blackhole OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export QWEN_PRECISION_CONFIG="$QWEN_MODEL_SOURCE/models/demos/qwen38_27b_qb2/config/precision_single_step_gdn.json"
unset TT_METAL_SLOW_DISPATCH_MODE TT_METAL_ALLOCATOR_MODE_HYBRID
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_MID_RUN_DUMP TT_METAL_PROFILER_CPP_POST_PROCESS
cd "$QWEN_MODEL_SOURCE"
python -m pytest models/demos/qwen38_27b_qb2/tests/unit -q --junitxml="$QWEN_VALIDATION_DIR/unit.xml"
export QWEN_GDN_NORMALIZE_QK=1 QWEN_GDN_VALUE_SPLITS=4 QWEN_GDN_INPUT_BUFFER_ITEMS=2
timeout --signal=TERM --kill-after=180 2400 /bin/bash \
    "$QWEN_MODEL_SOURCE/models/demos/qwen38_27b_qb2/demo/run_gdn_step_candidate.sh" \
    "$QWEN_TASK_ROOT" "$QWEN_VALIDATION_DIR/long-horizon" "$QWEN_MODEL_SOURCE"
python - "$QWEN_VALIDATION_DIR/long-horizon/candidate.json" <<'PY'
import json
import sys
receipt = json.load(open(sys.argv[1]))
assert receipt["state"] == "completed" and receipt["passed"] is True
assert receipt["normalize_qk"] is True
assert receipt["long_horizon"][-1]["steps"] == 4096
PY
export QWEN_COMPACT_DECODE_RESIDUAL=1 QWEN_COMPACT_DECODE_MLP=1
export QWEN_COMPACT_DECODE_ATTENTION=1 QWEN_BATCHED_DECODE_ROPE=1
export QWEN_BATCHED_PREFILL=1 QWEN_PREFILL_RESIDUAL_LAYOUT=sharded_replicated_norm
export QWEN_PREFILL_BATCHED_HEAD=1 QWEN_PREFILL_SKIP_INTERMEDIATE_HEAD=1 QWEN_PREFILL_STARTUP_WARMUP=1
export QWEN_DECODE_BUCKETS=1 QWEN_GALAXY_SMOKE=1
export QWEN_GALAXY_RECEIPT="$QWEN_VALIDATION_DIR/full-model.json"
exec /bin/bash "$QWEN_TASK_ROOT/source/scripts/run_safe_pytest.sh" \
    "$QWEN_MODEL_SOURCE/models/demos/qwen38_27b_qb2/tests/test_galaxy_smoke.py" \
    -vv -s --timeout=3000 --junitxml="$QWEN_VALIDATION_DIR/full-model.xml"
