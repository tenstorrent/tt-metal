#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Isolated bring-up layout: reuse the matching pinned runtime and environment;
# import only the Python model modifications from the separate Metal worktree.
QWEN_TASK_ROOT=${1:?Provide the isolated Qwen runtime directory}
QWEN_RUN_DIR=${2:?Provide a new receipt directory}
if [[ -n "${QWEN_WAIT_FOR_UNIT:-}" ]]; then
    echo "Waiting for $QWEN_WAIT_FOR_UNIT before qualification"
    while true; do
        QWEN_PREVIOUS_STATE=$(systemctl --user show "$QWEN_WAIT_FOR_UNIT" -p ActiveState --value)
        case "$QWEN_PREVIOUS_STATE" in
            inactive|failed) break ;;
            active|activating|deactivating) sleep 5 ;;
            *) echo "Unrecognized predecessor state: $QWEN_PREVIOUS_STATE" >&2; exit 3 ;;
        esac
    done
fi
mkdir "$QWEN_RUN_DIR"
export PATH="$QWEN_TASK_ROOT/python_env/bin:$PATH"
export TT_METAL_HOME="$QWEN_TASK_ROOT/metal"
export PYTHONPATH="$QWEN_TASK_ROOT/metal-galaxy:$TT_METAL_HOME:$TT_METAL_HOME/tools"
export LD_LIBRARY_PATH="$QWEN_TASK_ROOT/metal-install/lib:$QWEN_TASK_ROOT/metal-build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_CACHE="$QWEN_TASK_ROOT/jit-cache-metal-galaxy"
export MODEL_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR:?Set the pinned checkpoint directory}
export ARCH_NAME=blackhole
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
unset TT_METAL_SLOW_DISPATCH_MODE TT_METAL_ALLOCATOR_MODE_HYBRID
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_MID_RUN_DUMP TT_METAL_PROFILER_CPP_POST_PROCESS
cd "$QWEN_TASK_ROOT/metal-galaxy"
python - <<'PY'
from pathlib import Path
from models.demos.qwen38_27b_qb2.tt import model
assert Path(model.__file__).resolve().is_relative_to(Path.cwd()), model.__file__
print("Model source:", model.__file__, flush=True)
PY
python -m pytest models/demos/qwen38_27b_qb2/tests/unit -q --junitxml="$QWEN_RUN_DIR/unit.xml"
export QWEN_GALAXY_SMOKE=1
export QWEN_GALAXY_RECEIPT="$QWEN_RUN_DIR/full-model.json"
QWEN_TEST=test_galaxy_smoke.py
QWEN_TEST_TIMEOUT=1800
if [[ "${QWEN_GALAXY_REPLICAS:-1}" != 1 ]]; then
    QWEN_TEST=test_galaxy_replicas.py
    QWEN_TEST_TIMEOUT=5400
fi
exec /bin/bash "$QWEN_TASK_ROOT/source/scripts/run_safe_pytest.sh" \
    "$QWEN_TASK_ROOT/metal-galaxy/models/demos/qwen38_27b_qb2/tests/$QWEN_TEST" \
    -vv -s --timeout="$QWEN_TEST_TIMEOUT" --junitxml="$QWEN_RUN_DIR/full-model.xml"
