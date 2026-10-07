#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
QWEN_TASK_ROOT=${1:?Provide the isolated runtime directory}
QWEN_MODEL_SOURCE=${2:?Provide the pinned candidate source checkout}
QWEN_SWEEP_ROOT=${3:?Provide a new sweep directory}
QWEN_VALIDATION_DIR=${4:?Provide the completed kernel validation directory}
mkdir "$QWEN_SWEEP_ROOT"
export PATH="$QWEN_TASK_ROOT/python_env/bin:$PATH"
export TT_METAL_HOME="$QWEN_TASK_ROOT/metal"
export PYTHONPATH="$QWEN_MODEL_SOURCE:$TT_METAL_HOME:$TT_METAL_HOME/tools"
export LD_LIBRARY_PATH="$QWEN_TASK_ROOT/metal-install/lib:$QWEN_TASK_ROOT/metal-build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_CACHE="$QWEN_TASK_ROOT/jit-cache-metal-galaxy"
export MPLCONFIGDIR="$QWEN_TASK_ROOT/matplotlib-cache"
export MODEL_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR:?Set the pinned host-local checkpoint}
export ARCH_NAME=blackhole OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
unset TT_METAL_SLOW_DISPATCH_MODE TT_METAL_ALLOCATOR_MODE_HYBRID
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_MID_RUN_DUMP TT_METAL_PROFILER_CPP_POST_PROCESS
export QWEN_PRECISION_CONFIG="$QWEN_MODEL_SOURCE/models/demos/qwen38_27b_qb2/config/precision_single_step_gdn.json"
cd "$QWEN_MODEL_SOURCE"
python - "$QWEN_MODEL_SOURCE" "$QWEN_VALIDATION_DIR" <<'PY'
import json
import sys
from pathlib import Path
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import verify_qualified_source
source, validation = map(Path, sys.argv[1:])
model = json.loads((validation / "full-model.json").read_text())
kernel = json.loads((validation / "long-horizon/candidate.json").read_text())
assert model["passed"] is True and model["repeat_equal"] is True and model["layers"] == 64
assert model["precision"]["decode_recurrence"] == "single_step"
verify_qualified_source(model, source / "models/demos/qwen38_27b_qb2")
assert kernel["passed"] is True and kernel["normalize_qk"] is True
assert kernel["long_horizon"][-1]["steps"] == 4096
print("Kernel accuracy and full-model repeatability gates match this source; reference eval qualification remains pending", flush=True)
PY
python -m pytest models/demos/qwen38_27b_qb2/tests/unit -q --junitxml="$QWEN_SWEEP_ROOT/unit.xml"
python - "$QWEN_SWEEP_ROOT" <<'PY'
import sys
from pathlib import Path
from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, render, save_report
root = Path(sys.argv[1])
for variant in ("native", "single-step"):
    plan = make_plan(1, batches=(8, 4, 16, 1, 2, 32, 64), input_lengths=(131072, 262016, 32768, 8192, 128))
    plan["recurrence_variant"] = variant
    plan["qualification_scope"] = "FP32 kernel and one-replica repeatability; GPQA and eight-replica qualification pending"
    save_report(plan, root / variant)
    render(plan, root / variant)
PY
# Exact-batch native measurements bypass serving's 1/8/16 bucket selection.
# Both modes use identical projection, attention and prefill settings.
export QWEN_DECODE_BUCKETS=0 QWEN_GALAXY_SWEEP=1
export QWEN_COMPACT_DECODE_RESIDUAL=1 QWEN_COMPACT_DECODE_MLP=1
export QWEN_COMPACT_DECODE_ATTENTION=1 QWEN_BATCHED_DECODE_ROPE=1
export QWEN_BATCHED_PREFILL=1 QWEN_PREFILL_RESIDUAL_LAYOUT=sharded_replicated_norm
export QWEN_PREFILL_BATCHED_HEAD=1 QWEN_PREFILL_SKIP_INTERMEDIATE_HEAD=1 QWEN_PREFILL_STARTUP_WARMUP=1
for QWEN_SWEEP_VARIANT in native single-step; do
    QWEN_POLICY_FILE=precision_accurate_decode.json
    if [[ "$QWEN_SWEEP_VARIANT" == single-step ]]; then
        QWEN_POLICY_FILE=precision_single_step_gdn.json
    fi
    export QWEN_PRECISION_CONFIG="$QWEN_MODEL_SOURCE/models/demos/qwen38_27b_qb2/config/$QWEN_POLICY_FILE"
    export QWEN_SWEEP_RESULTS="$QWEN_SWEEP_ROOT/$QWEN_SWEEP_VARIANT"
    timeout --signal=TERM --kill-after=180 21600 /bin/bash "$QWEN_TASK_ROOT/source/scripts/run_safe_pytest.sh" \
        "$QWEN_MODEL_SOURCE/models/demos/qwen38_27b_qb2/tests/test_galaxy_perf_sweep.py" \
        -vv -s --timeout=21000 --junitxml="$QWEN_SWEEP_RESULTS/hardware.xml"
done
