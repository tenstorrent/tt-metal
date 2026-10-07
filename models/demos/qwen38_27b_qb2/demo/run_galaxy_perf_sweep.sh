#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
QWEN_TASK_ROOT=${1:?Provide the isolated Qwen runtime directory}
export QWEN_SWEEP_RESULTS=${2:?Provide the initialized sweep directory}
if [[ -n "${QWEN_WAIT_FOR_UNIT:-}" ]]; then
    echo "Waiting for $QWEN_WAIT_FOR_UNIT to finish before joining the hardware queue"
    while true; do
        QWEN_PREVIOUS_STATE=$(systemctl --user show "$QWEN_WAIT_FOR_UNIT" -p ActiveState --value)
        case "$QWEN_PREVIOUS_STATE" in
            inactive|failed) break ;;
            active|activating|deactivating) sleep 5 ;;
            *) echo "Unrecognized predecessor state: $QWEN_PREVIOUS_STATE" >&2; exit 3 ;;
        esac
    done
fi
export PATH="$QWEN_TASK_ROOT/python_env/bin:$PATH"
export TT_METAL_HOME="$QWEN_TASK_ROOT/metal"
export PYTHONPATH="$QWEN_TASK_ROOT/metal-galaxy:$TT_METAL_HOME:$TT_METAL_HOME/tools"
export LD_LIBRARY_PATH="$QWEN_TASK_ROOT/metal-install/lib:$QWEN_TASK_ROOT/metal-build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_CACHE="$QWEN_TASK_ROOT/jit-cache-metal-galaxy"
export MPLCONFIGDIR="$QWEN_TASK_ROOT/matplotlib-cache"
export MODEL_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR:?Set the pinned checkpoint directory}
export ARCH_NAME=blackhole OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 QWEN_GALAXY_SWEEP=1
export QWEN_COMPACT_DECODE_RESIDUAL=1
unset TT_METAL_SLOW_DISPATCH_MODE TT_METAL_ALLOCATOR_MODE_HYBRID
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_MID_RUN_DUMP TT_METAL_PROFILER_CPP_POST_PROCESS
# This baseline sweep uses the same model defaults as the passing TP4 smoke.
# Avoid silently inheriting performance experiments from an interactive shell.
unset QWEN_DECODE_BUCKETS QWEN_COMPACT_DECODE_MLP QWEN_BATCHED_DECODE_ROPE QWEN_COMPACT_DECODE_ATTENTION
unset QWEN_BATCHED_PREFILL QWEN_PREFILL_RESIDUAL_LAYOUT QWEN_PREFILL_BATCHED_HEAD
unset QWEN_PREFILL_SKIP_INTERMEDIATE_HEAD QWEN_PREFILL_STARTUP_WARMUP
exec /bin/bash "$QWEN_TASK_ROOT/source/scripts/run_safe_pytest.sh" \
    "$QWEN_TASK_ROOT/metal-galaxy/models/demos/qwen38_27b_qb2/tests/test_galaxy_perf_sweep.py" \
    -vv -s --timeout=10800 --junitxml="$QWEN_SWEEP_RESULTS/hardware.xml"
