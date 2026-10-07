#!/bin/bash
set -euo pipefail
TASK=/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006
DEMO="$TASK/metal-galaxy/models/demos/qwen38_27b_qb2/demo"
G0=qwen38-metal-galaxy-eight-replicas-v3-20261007.service
OLD=qwen38-galaxy-serving-v5-20261007.service
NEW=qwen38-galaxy-serving-v6-20261007.service
[[ $(systemctl --user show "$G0" -p ActiveState --value) == active ]]
[[ $(systemctl --user show "$OLD" -p ActiveState --value) == active ]]
[[ ! -e "$TASK/galaxy-serving-v5" && ! -e "$TASK/galaxy-serving-v6" ]]
[[ $(cat "$TASK/galaxy-serving-v5.log") == "Waiting for $G0 before serving" ]]
systemd-run --user --unit="$NEW" \
    --property=RuntimeMaxSec=172800 --property=MemoryMax=256G --property=CPUQuota=3200% \
    --property=TimeoutStopSec=300 \
    --property="StandardOutput=append:$TASK/galaxy-serving-v6.log" \
    --property="StandardError=append:$TASK/galaxy-serving-v6.log" \
    --setenv="MODEL_WEIGHTS_DIR=/home/ttuser/kimi-prefill.Ubx2wY/weights/qwen38-27b-20261006/checkpoint" \
    /bin/bash "$DEMO/run_profile_then_serving.sh" "$TASK" \
    "$TASK/layer-profile-v4" "$TASK/attention-tuning-v3" "$TASK/galaxy-serving-v6" \
    "$G0" "$TASK/galaxy-eight-replicas-v3/full-model.json"
# Refuse to stop the old service if it has advanced beyond its wait barrier.
if [[ $(systemctl --user show "$G0" -p ActiveState --value) != active || -e "$TASK/galaxy-serving-v5" ]] ||
   [[ $(cat "$TASK/galaxy-serving-v5.log") != "Waiting for $G0 before serving" ]]; then
    systemctl --user stop "$NEW"
    echo "Old serving job advanced; preserved it and cancelled the replacement" >&2
    exit 1
fi
systemctl --user stop "$OLD"
date -u +%FT%TZ >> "$TASK/long-context-queue-transition.txt"
printf '%s\n' 'G0-v3 continues unchanged; replaced waiting serving-v5 with serving-v6: bounded profile-v4 and attention-v3, then qualified baseline serving/GPQA. Each diagnostic limited to 15 minutes plus termination.' >> "$TASK/long-context-queue-transition.txt"
systemctl --user show "$G0" "$NEW" -p Id -p ActiveState -p MainPID
