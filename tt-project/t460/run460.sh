#!/bin/bash
# t460: one arm of the integrated-vs-t397 A/B: the Turbo e2e test (test_pipeline_turbo_minimax_h3.py),
# 768p generation, fl2va, v1.0 4-step adapter, 4x8, #330 baseline inputs.
# Args: <tag> <base|int> <dur s> <seed>. base = t422/b (12aff07c462: t397 8306d3e9c94 + the t399 fix + the
# test's seed/save knobs; built, warm JIT). int = the same build with the ttp/t460 models/ Python overlay
# ($T/src; the integrated diff over 12aff07c462 is Python only) and MINIMAX_H3_UPSCALE=1920x1080 (device 1080p).
# T408_REPLAYS (default 2) timed replays; T408_SAVE=1 saves the latents and decoded video in the out dir.
# pytest runs in the broker's process group under timeout --foreground -k 10 (no setsid, per the job lint);
# stop() sends TERM to timeout, which relays it to python, then KILLs both; a 90 s no-progress watchdog
# (log output or new cache files) ends it early.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t460; VENV=$T/python_env; W=$F/t422/b
TAG=${1:?tag}; ARM=${2:?base|int}; DUR=${3:?dur}; SEED=${4:?seed}
case $ARM in base) S=$W;; int) S=$T/src;; *) echo "[t460] arm must be base or int"; exit 2;; esac
OUT=$T/out_$TAG
[ -e $OUT/PASS ] && { echo "[t460] $TAG already passed"; exit 0; }
# Stop before any large write if / is too full or the project footprint has grown past blx01's cap.
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 88 ] || { echo "[t460] / at $use%"; exit 5; }
gb=$(timeout 120 du -csxBG $F 2>/dev/null | tail -1 | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 450 ] || { echo "[t460] footprint ${gb}G"; exit 5; }
mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
# Weights must be on local disk, checked readable before submit (never /mnt or other network fs).
export MINIMAX_H3_MODEL_PATH=$F/models/MiniMax-H3
[ -e $MINIMAX_H3_MODEL_PATH/READ_OK ] || { echo "[t460] no READ_OK in $MINIMAX_H3_MODEL_PATH"; exit 4; }
export MINIMAX_H3_TURBO_LORA_PATH=$F/models/lightx2v-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
[ -r $MINIMAX_H3_TURBO_LORA_PATH ] || { echo "[t460] no adapter $MINIMAX_H3_TURBO_LORA_PATH"; exit 4; }
# Matched FastH3 baseline inputs (#330): keyframe kf_t2va_f0.png, CALIBRATED_FOX_PROMPT (fixed in the test).
export MINIMAX_H3_TURBO_KEYFRAME=$F/t408/kf_t2va_f0.png MINIMAX_H3_TURBO_TASK=fl2va MINIMAX_H3_TURBO_POINT=768p MINIMAX_H3_TURBO_NFE=4
export MINIMAX_H3_TURBO_SEED=$SEED MINIMAX_H3_TURBO_REPLAYS=${T408_REPLAYS:-2}
for p in $MINIMAX_H3_MODEL_PATH $MINIMAX_H3_TURBO_LORA_PATH $MINIMAX_H3_TURBO_KEYFRAME; do
  r=$(readlink -f "$p"); fs=$(findmnt -n -o FSTYPE -T "$r" 2>/dev/null)
  case "$r" in /mnt/*) echo "[t460] $p resolves to $r under /mnt"; exit 4;; esac
  case "$fs" in nfs*|cifs|smb*|fuse*|sshfs|lustre|ceph*|9p|"") echo "[t460] $p is on fs '${fs:-unknown}', not local disk"; exit 4;; esac
done
# fl2va 5 s 1344x768 packs to 39773 rows, rung 41984 (t288/t400); the top rung is always warmed too.
export MINIMAX_H3_TURBO_WARM_RUNGS=${T408_RUNGS:-41984} MINIMAX_H3_TURBO_WARM_USED_LAYOUTS=1
unset MINIMAX_H3_TURBO_SAVE_DIR; [ "${T408_SAVE:-0}" = 1 ] && export MINIMAX_H3_TURBO_SAVE_DIR=$OUT
unset MINIMAX_H3_UPSCALE; [ $ARM = int ] && export MINIMAX_H3_UPSCALE=1920x1080
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3hf TT_DIT_CACHE_DIR=$F/cache/dit-h3hf
cd $S || exit 3
[ $ARM = base ] || grep -q MINIMAX_H3_UPSCALE models/tt_dit/pipelines/minimax_h3/pipeline_minimax_h3.py || { echo "[t460] $S lacks the upscale"; exit 3; }
grep -q MINIMAX_H3_TURBO_SAVE_DIR models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py || { echo "[t460] $W lacks the save flag"; exit 3; }
source $VENV/bin/activate || exit 3
export TT_METAL_HOME=$W PYTHONPATH=$S:$W:$W/ttnn:$W/tools:$F/t286
# t286_skipvaewarm: warm only this test's VAE canvases; T286_WARM_DEADLINE_S fails the test cleanly
# (mesh closed) once that many seconds pass inside any progress loop, warmup or denoise.
export T286_WARM_DEADLINE_S=${T286_WARM_DEADLINE_S:-400} T286_WARM_LATE_S=${T286_WARM_LATE_S:-400} T286_SKIP_VAE_WARM=1
MP4=$HOME/h3_turbo_artifacts/fl2va_turbo_1344x768_${DUR}s_4fwd.mp4; rm -f $MP4
CMD="timeout --foreground -k 10 ${T408_PY_TIMEOUT:-570} python -u -m pytest -sv -p no:cacheprovider -p t286_skipvaewarm models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py -k '${DUR}s and 4x8 and not 15s and not 4x32 and not WH'"
L=$OUT/run.log
echo "[t460] host=$(hostname) src=$S upscale=${MINIMAX_H3_UPSCALE:-none} seed=$SEED tag=$TAG arm=$ARM dur=$DUR rung=$MINIMAX_H3_TURBO_WARM_RUNGS replays=$MINIMAX_H3_TURBO_REPLAYS save=${T408_SAVE:-0} job=${TTP_RUNNER_JOB:-} $(date -u '+%F %T') UTC" | tee $L
echo "[t460] cmd: $CMD" | tee -a $L
env | grep -E '^(MINIMAX|TT_|T286_)' | sort >> $L
T0=$(date +%s)
bash -c "exec $CMD" >> $L 2>&1 < /dev/null & PG=$!
stop() {
  [ -n "$PG" ] || return 0
  local kids; kids=$(cat /proc/$PG/task/*/children 2>/dev/null)
  kill -TERM $PG 2>/dev/null; sleep "$1"; kill -KILL $PG $kids 2>/dev/null
}
trap 'stop 5' EXIT
trap 'exit 143' TERM; trap 'exit 130' INT
tail -n +1 -f --pid=$PG $L & TAILPID=$!
STAMP=$T/.wd_$TAG; touch $STAMP; last=$(stat -c %s $L); idle=0
while kill -0 $PG 2>/dev/null; do
  sleep 15
  now=$(stat -c %s $L)
  if [ "$now" != "$last" ]; then last=$now; idle=0; touch $STAMP; continue; fi
  # Silent cold-JIT compiles still write kernels to the caches.
  if [ -n "$(timeout 20 find $TT_METAL_CACHE $TT_DIT_CACHE_DIR -newer $STAMP -print -quit 2>/dev/null)" ]; then idle=0; touch $STAMP; continue; fi
  idle=$((idle + 15))
  if [ $idle -ge 90 ]; then
    echo "[t460] watchdog: no output and no new cache files for $idle s, ending the job" | tee -a $L
    stop 10; break
  fi
done
wait $PG; rc=$?; PG=
wait $TAILPID 2>/dev/null; rm -f $STAMP
echo "[t460] process wall $(( $(date +%s) - T0 )) s" | tee -a $L
mv $MP4 $OUT/ 2>/dev/null && md5sum $OUT/$(basename $MP4) | tee -a $L
echo "T408_EXIT=$rc" | tee -a $L
[ $rc = 0 ] && touch $OUT/PASS
exit $rc
