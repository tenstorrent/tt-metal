#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
QWEN_TASK_ROOT=${1:?Provide the isolated Qwen runtime directory}
QWEN_GDN_SOURCE=${2:?Provide the isolated GDN source checkout}
QWEN_GDN_RESULTS=${3:?Provide a new results directory}
mkdir "$QWEN_GDN_RESULTS"
export PATH="$QWEN_TASK_ROOT/python_env/bin:$PATH"
export TT_METAL_HOME="$QWEN_TASK_ROOT/metal"
export PYTHONPATH="$QWEN_GDN_SOURCE:$TT_METAL_HOME:$TT_METAL_HOME/tools"
export LD_LIBRARY_PATH="$QWEN_TASK_ROOT/metal-install/lib:$QWEN_TASK_ROOT/metal-build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_CACHE="$QWEN_TASK_ROOT/jit-cache-metal-galaxy"
export MODEL_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR:?Set the pinned checkpoint directory}
export ARCH_NAME=blackhole OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export QWEN_GDN_LAYER_INTEGRATION=1 QWEN_GDN_LAYER_RECEIPT="$QWEN_GDN_RESULTS/layer.json"
unset TT_METAL_SLOW_DISPATCH_MODE TT_METAL_ALLOCATOR_MODE_HYBRID
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_MID_RUN_DUMP TT_METAL_PROFILER_CPP_POST_PROCESS
cd "$QWEN_GDN_SOURCE"
exec /bin/bash "$QWEN_TASK_ROOT/source/scripts/run_safe_pytest.sh" \
    "$QWEN_GDN_SOURCE/models/demos/qwen38_27b_qb2/tests/test_gdn_layer_integration.py" \
    -vv -s --timeout=1800 --junitxml="$QWEN_GDN_RESULTS/hardware.xml"
