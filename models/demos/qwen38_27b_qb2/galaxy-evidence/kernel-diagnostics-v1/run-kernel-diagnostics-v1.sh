#!/bin/bash
set -uo pipefail
QWEN_WAIT_SECONDS=0
while true; do
    QWEN_PROFILE_STATE=$(systemctl --user show qwen38-layer-profile-v5-20261007.service -p ActiveState --value)
    case "$QWEN_PROFILE_STATE" in
        inactive|failed) break ;;
        active|activating|deactivating) ;;
        *) echo "Unexpected profile state: $QWEN_PROFILE_STATE"; exit 3 ;;
    esac
    if (( QWEN_WAIT_SECONDS >= 900 )); then echo "Predecessor wait timed out"; exit 4; fi
    sleep 5
    QWEN_WAIT_SECONDS=$((QWEN_WAIT_SECONDS+5))
done
echo "Attention precision diagnostic starting"
timeout --signal=TERM --kill-after=120s 2100s /bin/bash /home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal-galaxy/models/demos/qwen38_27b_qb2/demo/run_attention_precision.sh /home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006 /home/ttuser/qwen38-artifacts-20261007/attention-precision-v1 > /home/ttuser/qwen38-artifacts-20261007/attention-precision-v1.log 2>&1
QWEN_ATTENTION_EXIT=$?
echo "Attention precision exit: $QWEN_ATTENTION_EXIT"
echo "GDN buffering diagnostic starting"
QWEN_GDN_INPUT_BUFFER_ITEMS=1,2 timeout --signal=TERM --kill-after=120s 2100s /bin/bash /home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal-galaxy/models/demos/qwen38_27b_qb2/demo/run_gdn_step_candidate.sh /home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006 /home/ttuser/qwen38-artifacts-20261007/gdn-step-buffering-v1 > /home/ttuser/qwen38-artifacts-20261007/gdn-step-buffering-v1.log 2>&1
QWEN_GDN_EXIT=$?
echo "GDN buffering exit: $QWEN_GDN_EXIT"
if (( QWEN_ATTENTION_EXIT != 0 || QWEN_GDN_EXIT != 0 )); then exit 1; fi
