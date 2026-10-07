#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
QWEN_TASK_ROOT=${1:?Provide the isolated Qwen runtime directory}
QWEN_ATTENTION_DIR=${2:?Provide a new result directory}
mkdir "$QWEN_ATTENTION_DIR"
export PATH="$QWEN_TASK_ROOT/python_env/bin:$PATH"
export TT_METAL_HOME="$QWEN_TASK_ROOT/metal"
export PYTHONPATH="$QWEN_TASK_ROOT/metal-galaxy:$TT_METAL_HOME:$TT_METAL_HOME/tools"
export LD_LIBRARY_PATH="$QWEN_TASK_ROOT/metal-install/lib:$QWEN_TASK_ROOT/metal-build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_CACHE="$QWEN_TASK_ROOT/jit-cache-metal-galaxy"
export ARCH_NAME=blackhole OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export QWEN_MODEL_ATTENTION=1 QWEN_MODEL_ATTENTION_RECEIPT="$QWEN_ATTENTION_DIR/attention.json"
unset TT_METAL_SLOW_DISPATCH_MODE TT_METAL_ALLOCATOR_MODE_HYBRID
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_MID_RUN_DUMP TT_METAL_PROFILER_CPP_POST_PROCESS
exec /bin/bash "$QWEN_TASK_ROOT/source/scripts/run_safe_pytest.sh" \
    "$QWEN_TASK_ROOT/metal-galaxy/models/demos/qwen38_27b_qb2/tests/test_model_decode_attention.py" \
    -vv -s --timeout=1800 --junitxml="$QWEN_ATTENTION_DIR/hardware.xml"
