#!/bin/bash
#SBATCH --job-name=dit-segment
#SBATCH --nodes=1
#SBATCH --time=00:25:00
#SBATCH --nice=10000
#SBATCH --output=logs/segment_%j.log

# Low-priority segmented DiT training: train for SEGMENT_SECONDS, save a
# resume checkpoint, exit, and resubmit this same script. Between segments
# the node returns to the pool, and (with --nice) anyone else's queued job
# runs first — voluntary preemption on clusters without slurm preemption.
#
# Required env (pass with sbatch --export=ALL,VAR=...):
#   REPO      tt-metal root (built, with python_env)
#   CONFIG    training config yaml (absolute path)
#   RUN_NAME  stable run dir name (resume identity)
# Optional:
#   SEGMENT_SECONDS  training budget per segment (default 900)
#   TT_MESH_GRAPH_DESC_PATH  REQUIRED for multi-chip meshes: auto-selection may
#                    pick a descriptor with the wrong eth-channel count
#   SBATCH_EXTRA     extra args for the resubmission (e.g. --partition=...)
set -euo pipefail

: "${REPO:?}" "${CONFIG:?}" "${RUN_NAME:?}"
SEGMENT_SECONDS=${SEGMENT_SECONDS:-900}

export TT_METAL_HOME=$REPO TT_METAL_RUNTIME_ROOT=$REPO PYTHONUNBUFFERED=1
# Optional: STAGING_DIR with fresher example code to copy over the repo copy.
if [ -n "${STAGING_DIR:-}" ]; then cp $STAGING_DIR/*.py $REPO/tt-train/sources/examples/dit/; fi
cd $REPO/tt-train/sources/examples/dit

set +e
$REPO/python_env/bin/python train_dit_cifar.py -c "$CONFIG" \
    --run-name "$RUN_NAME" --resume --max-seconds "$SEGMENT_SECONDS" \
    | tee /tmp/segment_out_$$.log
rc=${PIPESTATUS[0]}
tail=$(grep -aE "SEGMENT_END|TRAINING_COMPLETE" /tmp/segment_out_$$.log | tail -1)
rm -f /tmp/segment_out_$$.log
set -e

# Leave nothing on node-local disk.
rm -rf /home/$USER/.cache/tt-metal-cache 2>/dev/null || true

if [ $rc -ne 0 ]; then
    echo "segment crashed (rc=$rc); chain stopped"
    exit $rc
fi
case "$tail" in
    TRAINING_COMPLETE*) echo "chain finished: $tail" ;;
    SEGMENT_END*)
        echo "resubmitting: $tail"
        sbatch ${SBATCH_EXTRA:-} --export=ALL,REPO=$REPO,CONFIG=$CONFIG,RUN_NAME=$RUN_NAME,SEGMENT_SECONDS=$SEGMENT_SECONDS,STAGING_DIR=${STAGING_DIR:-},SBATCH_EXTRA="${SBATCH_EXTRA:-}" "$0"
        ;;
    *) echo "no end marker (rc=0 but unexpected output); chain stopped" ;;
esac
