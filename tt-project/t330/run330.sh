#!/bin/bash
# t330: one broker job running the unmodified Turbo e2e test (test_pipeline_turbo_minimax_h3.py) on
# the t286 build, 4x8, 4-step, seed 0, fox prompt. Args: <tag> <duration 5|10> <task t2va|fl2va> [keyframe png].
set -o pipefail
F=/var/tmp/fasth3; T=$F/t286; O=$F/t330; W=${T286_W:-/home/smarton/fasth3/t286}
# Run the work in its own process group and kill the whole group on exit or signal, so no pytest
# child outlives the broker job.
if [ -z "$T330_INNER" ]; then
  T330_INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
TAG=${1:?tag}; DUR=${2:?duration}; TASK=${3:?task}; KF=${4:-}
[ "$TASK" = fl2va ] && [ ! -e "$KF" ] && { echo "[t330] fl2va needs a keyframe"; exit 4; }
OUT=$O/out_$TAG
[ -e $OUT/PASS ] && { echo "[t330] $TAG already passed"; exit 0; }
# Stop before any large write if / is too full or our outputs and caches have grown too large.
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le ${T286_ROOT_MAX:-85} ] || { echo "[t330] / at $use%"; exit 5; }
gb=$(timeout 60 du -csxBG $T $O $F/cache/dit-h3hf $F/cache/tt-metal-cache-h3hf 2>/dev/null | tail -1 | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le ${T286_FOOT_MAX:-140} ] || { echo "[t330] footprint ${gb}G"; exit 5; }
mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
# Weights must be on local disk, checked readable before submit (never /mnt or other network fs).
export MINIMAX_H3_MODEL_PATH=$F/models/MiniMax-H3
case "$(readlink -f $MINIMAX_H3_MODEL_PATH)" in /mnt/*) echo "[t330] refusing network-fs weights"; exit 4;; esac
[ -e $MINIMAX_H3_MODEL_PATH/READ_OK ] || { echo "[t330] no READ_OK in $MINIMAX_H3_MODEL_PATH"; exit 4; }
export MINIMAX_H3_TURBO_LORA_PATH=$F/models/lightx2v-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors
export MINIMAX_H3_TURBO_TASK=$TASK MINIMAX_H3_TURBO_POINT=768p MINIMAX_H3_TURBO_NFE=4
# The plugin also reads MINIMAX_H3_TURBO_KEYFRAME to pick VAE canvases, so leave it unset for t2va.
unset MINIMAX_H3_TURBO_KEYFRAME; [ -n "$KF" ] && export MINIMAX_H3_TURBO_KEYFRAME=$KF
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3hf TT_DIT_CACHE_DIR=$F/cache/dit-h3hf
mkdir -p $TT_METAL_CACHE $TT_DIT_CACHE_DIR
cd $W || exit 3
source ${T286_VENV:-/home/smarton/fasth3/tt-metal/python_env}/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools:$T
K="${DUR}s and 4x8 and not 15s and not 4x32 and not WH"
CMD="python -u -m pytest -sv -p no:cacheprovider -p t286_skipvaewarm models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py -k '$K'"
echo "[t330] host=$(hostname) commit=$(git -C $W rev-parse HEAD) tag=$TAG dur=$DUR task=$TASK kf=${KF:-none} job=${TTP_RUNNER_JOB:-} $(date -u '+%F %T') UTC" | tee $OUT/run.log
echo "[t330] cmd: $CMD" | tee -a $OUT/run.log
env | grep -E '^(MINIMAX|TT_)' | sort >> $OUT/run.log
T0=$(date +%s)
eval "$CMD" 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[t330] process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
mv $HOME/h3_turbo_artifacts/${TASK}_turbo_1344x768_${DUR}s_4fwd.mp4 $OUT/ 2>/dev/null
echo "T330_EXIT=$rc" | tee -a $OUT/run.log
[ $rc = 0 ] && touch $OUT/PASS
exit $rc
