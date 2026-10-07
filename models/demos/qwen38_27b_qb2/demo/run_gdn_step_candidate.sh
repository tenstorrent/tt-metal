#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
QWEN_TASK_ROOT=${1:?Provide the isolated Qwen runtime directory}
QWEN_GDN_STEP_DIR=${2:?Provide a new candidate results directory}
QWEN_GDN_SOURCE=${3:-$QWEN_TASK_ROOT/metal-galaxy}
if [[ -n "${QWEN_WAIT_FOR_UNIT:-}" ]]; then
    echo "Waiting for $QWEN_WAIT_FOR_UNIT before the isolated GDN experiment"
    while true; do
        QWEN_PREVIOUS_STATE=$(systemctl --user show "$QWEN_WAIT_FOR_UNIT" -p ActiveState --value)
        case "$QWEN_PREVIOUS_STATE" in
            inactive|failed) break ;;
            active|activating|deactivating) sleep 5 ;;
            *) echo "Unrecognized predecessor state: $QWEN_PREVIOUS_STATE" >&2; exit 3 ;;
        esac
    done
fi
mkdir "$QWEN_GDN_STEP_DIR"
export PATH="$QWEN_TASK_ROOT/python_env/bin:$PATH"
export TT_METAL_HOME="$QWEN_TASK_ROOT/metal"
export PYTHONPATH="$QWEN_GDN_SOURCE:$TT_METAL_HOME:$TT_METAL_HOME/tools"
export LD_LIBRARY_PATH="$QWEN_TASK_ROOT/metal-install/lib:$QWEN_TASK_ROOT/metal-build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_CACHE="$QWEN_TASK_ROOT/jit-cache-metal-galaxy"
export ARCH_NAME=blackhole OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export QWEN_GDN_STEP_CANDIDATE=1 QWEN_GDN_STEP_RECEIPT="$QWEN_GDN_STEP_DIR/candidate.json"
unset TT_METAL_SLOW_DISPATCH_MODE TT_METAL_ALLOCATOR_MODE_HYBRID
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_MID_RUN_DUMP TT_METAL_PROFILER_CPP_POST_PROCESS
exec /bin/bash "$QWEN_TASK_ROOT/source/scripts/run_safe_pytest.sh" \
    "$QWEN_GDN_SOURCE/models/demos/qwen38_27b_qb2/tests/test_gdn_step_candidate.py" \
    -vv -s --timeout=1800 --junitxml="$QWEN_GDN_STEP_DIR/hardware.xml"
