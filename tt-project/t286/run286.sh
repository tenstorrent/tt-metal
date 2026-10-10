#!/bin/bash
# t286: one broker job running the unmodified Turbo e2e test (test_pipeline_turbo_minimax_h3.py) on
# ttp/fasth3-hyperflow (e24a2b93d79), fl2va, 4x8. Args: <tag> <duration 5|10|15>.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t286; W=${T286_W:-$F/t284/b}
# Run the work in its own process group and kill the whole group on exit or signal, so no pytest
# child outlives the broker job (job 086 left one holding the device for 2 h).
if [ -z "$T286_INNER" ]; then
  T286_INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
TAG=${1:?tag}; DUR=${2:?duration}
# T286_ONCE=1: the clip already passed in an earlier job, so a queued repeat exits before the device.
[ "${T286_ONCE:-0}" = 1 ] && [ -e $T/out_$TAG/PASS ] && { echo "[t286] $TAG already passed"; exit 0; }
# Stop before any large write if / is too full (blx03 cap 85%, blx01 70%) or our own t286 outputs
# and caches have grown past T286_FOOT_MAX GB.
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le ${T286_ROOT_MAX:-70} ] || { echo "[t286] / at $use%"; exit 5; }
gb=$(timeout 60 du -csxBG $T $F/cache/dit-h3hf $F/cache/tt-metal-cache-h3hf 2>/dev/null | tail -1 | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le ${T286_FOOT_MAX:-120} ] || { echo "[t286] footprint ${gb}G"; exit 5; }
OUT=$T/out_$TAG; mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
# Weights must be on local disk, checked readable before submit (never /mnt or other network fs).
export MINIMAX_H3_MODEL_PATH=${T286_MODEL:-$F/models/MiniMax-H3}
case "$(readlink -f $MINIMAX_H3_MODEL_PATH)" in /mnt/*) echo "[t286] refusing network-fs weights"; exit 4;; esac
[ -e $MINIMAX_H3_MODEL_PATH/READ_OK ] || { echo "[t286] no READ_OK in $MINIMAX_H3_MODEL_PATH"; exit 4; }
export MINIMAX_H3_TURBO_LORA_PATH=$F/models/lightx2v-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors
export MINIMAX_H3_TURBO_KEYFRAME=$F/t209/kf_first.png MINIMAX_H3_TURBO_TASK=fl2va MINIMAX_H3_TURBO_POINT=768p MINIMAX_H3_TURBO_NFE=4
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3hf TT_DIT_CACHE_DIR=$F/cache/dit-h3hf
mkdir -p $TT_METAL_CACHE $TT_DIT_CACHE_DIR
cd $W || exit 3
source ${T286_VENV:-$F/t48/python_env}/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools:$T
# T286_SKIP_VAE_WARM=1: warm only the canvases this test reaches (t286_skipvaewarm.py).
P=""; [ "${T286_SKIP_VAE_WARM:-0}" = 1 ] && P="-p t286_skipvaewarm"
K="${DUR}s and 4x8 and not 15s and not 4x32 and not WH"; [ "$DUR" = 15 ] && K="15s and 4x8 and not 4x32 and not WH"
CMD="python -u -m pytest -sv -p no:cacheprovider $P models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py -k '$K'"
echo "[t286] host=$(hostname) commit=$(git -C $W rev-parse HEAD) tag=$TAG dur=$DUR job=${TTP_RUNNER_JOB:-} $(date -u '+%F %T') UTC" | tee $OUT/run.log
echo "[t286] cmd: $CMD" | tee -a $OUT/run.log
env | grep -E '^(MINIMAX|TT_)' | sort >> $OUT/run.log
T0=$(date +%s)
eval "$CMD" 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[t286] process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
cp -p $HOME/h3_turbo_artifacts/fl2va_turbo_1344x768_${DUR}s_4fwd.mp4 $OUT/ 2>/dev/null
echo "T286_EXIT=$rc" | tee -a $OUT/run.log
[ $rc = 0 ] && touch $OUT/PASS
exit $rc
