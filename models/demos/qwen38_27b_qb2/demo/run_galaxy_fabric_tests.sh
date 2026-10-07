#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

QWEN_TASK_ROOT=${1:?Provide the isolated Qwen runtime directory}
QWEN_RUN_DIR=${2:?Provide a new receipt directory}
test -x "$QWEN_TASK_ROOT/metal-fabric-build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric"
mkdir "$QWEN_RUN_DIR"
exec 9>/tmp/tt-device.lock
echo "Waiting for the allocated Galaxy device lock"
flock --timeout "${QWEN_DEVICE_LOCK_TIMEOUT_SECONDS:-6000}" 9
if [[ -e /tmp/tt-device.dirty ]]; then
    echo "Previous hardware job left the device marked dirty; recovery is required" >&2
    exit 3
fi
touch /tmp/tt-device.dirty
export PATH="$QWEN_TASK_ROOT/python_env/bin:$PATH"
export TT_METAL_HOME="$QWEN_TASK_ROOT/metal"
export TT_METAL_CACHE="$QWEN_TASK_ROOT/jit-cache-fabric-tests"
export LD_LIBRARY_PATH="$QWEN_TASK_ROOT/metal-fabric-build/lib:${LD_LIBRARY_PATH:-}"
unset TT_METAL_SLOW_DISPATCH_MODE TT_METAL_ALLOCATOR_MODE_HYBRID TT_VISIBLE_DEVICES
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_MID_RUN_DUMP TT_METAL_PROFILER_CPP_POST_PROCESS
cd "$TT_METAL_HOME"
# -e and pipefail propagate a failing test binary through the upstream log/tee
# pipeline. The outer persistent unit also bounds the time spent waiting above.
timeout --kill-after=30s 15m /bin/bash -e -o pipefail tools/scaleout/exabox/run_fabric_tests.sh \
    --hosts localhost --image none --config 4x8 --mpi-if none \
    --test-binary "$QWEN_TASK_ROOT/metal-fabric-build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric" \
    --test-config tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_sanity_neighbor_exchange.yaml \
    --filter name.2DTorusXYNeighborExchange --num-packets 1000 --output "$QWEN_RUN_DIR"
# Only a successful test clears our marker. A failure/timeout leaves it for the
# existing safe pytest runner's explicit reset on the next hardware attempt.
rm -f /tmp/tt-device.dirty
