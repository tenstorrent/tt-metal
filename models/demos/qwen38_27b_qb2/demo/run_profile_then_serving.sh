#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Bounded optional diagnostics before the qualified resident server takes the
# device lock. Failed experimental diagnostics never promote model changes.
set -euo pipefail
QWEN_TASK_ROOT=${1:?Provide the isolated Qwen runtime directory}
QWEN_PROFILE_DIR=${2:?Provide a new profile directory}
QWEN_ATTENTION_DIR=${3:?Provide a new attention results directory}
QWEN_SERVING_DIR=${4:?Provide a new serving results directory}
QWEN_G0_UNIT=${5:?Provide the prerequisite qualification unit}
QWEN_G0_RECEIPT=${6:?Provide its full-model receipt}
QWEN_DEMO="$QWEN_TASK_ROOT/metal-galaxy/models/demos/qwen38_27b_qb2/demo"
echo "Waiting for $QWEN_G0_UNIT before bounded diagnostics and serving"
while true; do
    QWEN_STATE=$(systemctl --user show "$QWEN_G0_UNIT" -p ActiveState --value)
    case "$QWEN_STATE" in
        inactive) break ;;
        failed) echo "Qualification failed; refusing to launch serving" >&2; exit 3 ;;
        active|activating|deactivating) sleep 5 ;;
        *) echo "Unrecognized qualification state: $QWEN_STATE" >&2; exit 3 ;;
    esac
done
[[ $(systemctl --user show "$QWEN_G0_UNIT" -p ExecMainStatus --value) == 0 ]]
unset QWEN_WAIT_FOR_UNIT
for QWEN_DIAGNOSTIC in profile attention; do
    if [[ "$QWEN_DIAGNOSTIC" == profile ]]; then
        if [[ "${QWEN_SKIP_PROFILE:-0}" == 1 ]]; then
            echo "Profile explicitly deferred; its P0 gate remains incomplete"
            continue
        fi
        QWEN_SCRIPT=run_galaxy_layer_profile.sh
        QWEN_OUTPUT=$QWEN_PROFILE_DIR
    else
        QWEN_SCRIPT=run_long_context_attention.sh
        QWEN_OUTPUT=$QWEN_ATTENTION_DIR
    fi
    QWEN_STATUS=0
    # A strict 15-minute execution cap plus at most five minutes to terminate;
    # each underlying runner owns the cooperative lock and dirty-device guard.
    timeout --signal=TERM --kill-after=300 900 /bin/bash "$QWEN_DEMO/$QWEN_SCRIPT" \
        "$QWEN_TASK_ROOT" "$QWEN_OUTPUT" > "$QWEN_OUTPUT.log" 2>&1 || QWEN_STATUS=$?
    printf '%s diagnostic exit status: %s; baseline model unchanged\n' "$QWEN_DIAGNOSTIC" "$QWEN_STATUS"
done
# The serving script validates G0, replica placement and exact model hashes
# again, and resets a dirty device under the lock if a diagnostic failed.
exec /bin/bash "$QWEN_DEMO/run_galaxy_serving.sh" \
    "$QWEN_TASK_ROOT" "$QWEN_SERVING_DIR" "$QWEN_G0_RECEIPT"
