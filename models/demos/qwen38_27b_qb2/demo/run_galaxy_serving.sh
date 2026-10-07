#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
QWEN_TASK_ROOT=${1:?Provide the isolated Qwen runtime directory}
QWEN_SERVING_RESULTS=${2:?Provide a new receipt directory}
QWEN_G0_RECEIPT=${3:?Provide the passing eight-replica G0 receipt}
if [[ -n "${QWEN_WAIT_FOR_UNIT:-}" ]]; then
    echo "Waiting for $QWEN_WAIT_FOR_UNIT before serving"
    while true; do
        QWEN_PREVIOUS_STATE=$(systemctl --user show "$QWEN_WAIT_FOR_UNIT" -p ActiveState --value)
        case "$QWEN_PREVIOUS_STATE" in
            inactive) break ;;
            failed) echo "Preceding hardware job failed; inspect its receipt before serving" >&2; exit 3 ;;
            active|activating|deactivating) sleep 5 ;;
            *) echo "Unrecognized predecessor state: $QWEN_PREVIOUS_STATE" >&2; exit 3 ;;
        esac
    done
    if [[ "$(systemctl --user show "$QWEN_WAIT_FOR_UNIT" -p ExecMainStatus --value)" != 0 ]]; then
        echo "Preceding hardware job did not exit successfully" >&2
        exit 3
    fi
fi
export PATH="$QWEN_TASK_ROOT/serving_env/bin:$PATH"
export TT_METAL_HOME="$QWEN_TASK_ROOT/metal"
export PYTHONPATH="$QWEN_TASK_ROOT/metal-galaxy:$TT_METAL_HOME:$TT_METAL_HOME/tools"
export LD_LIBRARY_PATH="$QWEN_TASK_ROOT/metal-install/lib:$QWEN_TASK_ROOT/metal-build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_CACHE="$QWEN_TASK_ROOT/jit-cache-metal-galaxy"
export MPLCONFIGDIR="$QWEN_TASK_ROOT/matplotlib-cache"
export ARCH_NAME=blackhole OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export MODEL_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR:?Set the pinned checkpoint directory}
# Check G0 before touching the hardware. The supervisor repeats this validation
# and records the exact receipt hash alongside the actual worker assignments.
python - "$QWEN_G0_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path
from models.demos.qwen38_27b_qb2.demo import galaxy_serving
receipt = json.loads(Path(sys.argv[1]).read_text())
groups = galaxy_serving.qualified_groups(receipt)
galaxy_serving.verify_qualified_source(receipt, Path(galaxy_serving.__file__).resolve().parents[1])
print("QUALIFIED_GROUPS", groups, flush=True)
PY
exec 9>/tmp/tt-device.lock
echo "Waiting for the shared device lock before serving"
flock 9
if [[ -f /tmp/tt-device.dirty ]]; then
    # A preceding diagnostic can leave the device dirty. Use the already
    # authorized Galaxy reset only after acquiring the shared lock.
    tt-smi -glx_reset
fi
touch /tmp/tt-device.dirty
# Keep the marker pessimistically: a long-lived server can be stopped between
# receipt updates. The next safe runner will reset before using the devices.
exec python "$QWEN_TASK_ROOT/metal-galaxy/models/demos/qwen38_27b_qb2/demo/run_galaxy_serving.py" \
    --task-root "$QWEN_TASK_ROOT" --output "$QWEN_SERVING_RESULTS" --qualification "$QWEN_G0_RECEIPT"
