#!/bin/bash
# t466: e2e A/B of ring SDPA matmul fidelity on the Turbo e2e test (test_pipeline_turbo_minimax_h3.py, unmodified),
# 768p fl2va, v1.0 4-step adapter, device 1080p upscale (t460 int arm), 4x8, #330 inputs.
# Args: <tag> <dur s> <seed> <arms...>; arms run one after another in this job, each its own pytest:
#   base = MINIMAX_H3_SDPA_MM_FIDELITY unset, lofi = MINIMAX_H3_SDPA_MM_FIDELITY=lofi. Same build for both:
#   /var/tmp/fasth3/t466/b (e02052af0ee: t460 dbd6d66048f + #57979/#58032/#58223 + the knob).
# Weights are read from the local copy /var/tmp/fasth3/models/MiniMax-H3 (cp466.sh); never from /mnt.
# Every cache and output stays under /var/tmp/fasth3/t466. The first (cold) job builds the dit-h3hf cache (~115 GB)
# plus JIT (~16 GB) and needs >=381 GB free on / before it starts.
# pytest runs in the broker's process group under timeout --foreground -k 10 (no setsid); a 90 s no-progress
# watchdog (log output or new cache files) ends an arm early; two watchdog hangs end the job.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t466; W=$T/b; VENV=$F/t460/python_env
TAG=${1:?tag}; DUR=${2:?dur}; SEED=${3:?seed}; shift 3; ARMS=${*:?arms}
export TT_METAL_CACHE=$T/jit TT_DIT_CACHE_DIR=$T/cache/dit-h3hf
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
mkdir -p $T/cache $TT_METAL_CACHE $TT_DIT_CACHE_DIR $TMPDIR
# Nothing we write may land on /mnt (weka is read-only for us).
for p in $T $TT_METAL_CACHE $TT_DIT_CACHE_DIR $HOME $TMPDIR; do
  case "$(readlink -f "$p")" in /mnt/*|"") echo "[t466] $p resolves under /mnt or nowhere"; exit 4;; esac
done
# Space: / keeps >=250 GB free after a cold cache build (~131 GB); our total stays <=320 GB (#466/#468 exception).
cold=0; [ -n "$(ls -A $TT_DIT_CACHE_DIR 2>/dev/null)" ] || cold=1
free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); need=$((250 + cold * 131))
[ "$free" -ge $need ] || { echo "[t466] / has ${free}G free, need ${need}G (cold=$cold)"; exit 5; }
gb=$(timeout 120 du -sxBG $F 2>/dev/null | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le $((320 - cold * 131)) ] || { echo "[t466] footprint ${gb}G (cold=$cold)"; exit 5; }
export MINIMAX_H3_MODEL_PATH=$F/models/MiniMax-H3
case "$(readlink -f $MINIMAX_H3_MODEL_PATH)" in $F/*) ;; *) echo "[t466] weights not local"; exit 4;; esac
[ -r $MINIMAX_H3_MODEL_PATH/transformer/config.json ] || { echo "[t466] weights unreadable at $MINIMAX_H3_MODEL_PATH"; exit 4; }
export MINIMAX_H3_TURBO_LORA_PATH=$F/models/lightx2v-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
[ -r $MINIMAX_H3_TURBO_LORA_PATH ] || { echo "[t466] no adapter $MINIMAX_H3_TURBO_LORA_PATH"; exit 4; }
export MINIMAX_H3_TURBO_KEYFRAME=$F/t408/kf_t2va_f0.png MINIMAX_H3_TURBO_TASK=fl2va MINIMAX_H3_TURBO_POINT=768p MINIMAX_H3_TURBO_NFE=4
[ -r $MINIMAX_H3_TURBO_KEYFRAME ] || { echo "[t466] no keyframe"; exit 4; }
export MINIMAX_H3_TURBO_SEED=$SEED MINIMAX_H3_TURBO_REPLAYS=${T466_REPLAYS:-1}
export MINIMAX_H3_TURBO_WARM_RUNGS=${T466_RUNGS:-41984} MINIMAX_H3_TURBO_WARM_USED_LAYOUTS=1 MINIMAX_H3_UPSCALE=1920x1080
export T286_WARM_DEADLINE_S=${T286_WARM_DEADLINE_S:-400} T286_WARM_LATE_S=${T286_WARM_LATE_S:-400} T286_SKIP_VAE_WARM=1
[ -r $F/t286/t286_skipvaewarm.py ] || { echo "[t466] no t286_skipvaewarm plugin"; exit 3; }
cd $W || exit 3
grep -q MINIMAX_H3_SDPA_MM_FIDELITY models/tt_dit/models/transformers/minimax_h3/attention_minimax_h3.py || { echo "[t466] $W lacks the knob"; exit 3; }
source $VENV/bin/activate || exit 3
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools:$F/t286
MP4=$HOME/h3_turbo_artifacts/fl2va_turbo_1344x768_${DUR}s_4fwd.mp4
PG=
stop() {
  [ -n "$PG" ] || return 0
  local kids; kids=$(cat /proc/$PG/task/*/children 2>/dev/null)
  kill -TERM $PG 2>/dev/null; sleep "$1"; kill -KILL $PG $kids 2>/dev/null
}
trap 'stop 5' EXIT
trap 'exit 143' TERM; trap 'exit 130' INT
hangs=0; rc=0; lowdisk=
for ARM in $ARMS; do
  OUT=$T/out_${TAG}_$ARM; L=$OUT/run.log
  [ -e $OUT/PASS ] && { echo "[t466] $TAG $ARM already passed"; continue; }
  mkdir -p $OUT; rm -f $MP4
  case $ARM in base) unset MINIMAX_H3_SDPA_MM_FIDELITY;; lofi) export MINIMAX_H3_SDPA_MM_FIDELITY=lofi;; *) echo "[t466] arm $ARM?"; exit 2;; esac
  export MINIMAX_H3_TURBO_SAVE_DIR=$OUT
  CMD="timeout --foreground -k 10 ${T466_PY_TIMEOUT:-280} python -u -m pytest -sv -p no:cacheprovider -p t286_skipvaewarm models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py -k '${DUR}s and 4x8 and not 15s and not 4x32 and not WH'"
  echo "[t466] host=$(hostname) build=$W tag=$TAG arm=$ARM fidelity=${MINIMAX_H3_SDPA_MM_FIDELITY:-default} dur=$DUR seed=$SEED cold=$cold replays=$MINIMAX_H3_TURBO_REPLAYS job=${TTP_RUNNER_JOB:-} $(date -u '+%F %T') UTC" | tee $L
  echo "[t466] cmd: $CMD" | tee -a $L
  env | grep -E '^(MINIMAX|TT_|T286_)' | sort >> $L
  T0=$(date +%s)
  bash -c "exec $CMD" >> $L 2>&1 < /dev/null & PG=$!
  tail -n +1 -f --pid=$PG $L & TAILPID=$!
  STAMP=$T/.wd_$TAG; touch $STAMP; last=$(stat -c %s $L); idle=0
  while kill -0 $PG 2>/dev/null; do
    sleep 15
    if [ "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" -lt 250 ]; then
      echo "[t466] / under 250G free, ending $ARM (space floor)" | tee -a $L; lowdisk=1; stop 10; break
    fi
    now=$(stat -c %s $L)
    if [ "$now" != "$last" ]; then last=$now; idle=0; touch $STAMP; continue; fi
    # Silent cold compiles and cache conversion still write files.
    if [ -n "$(timeout 20 find $TT_METAL_CACHE $TT_DIT_CACHE_DIR -newer $STAMP -print -quit 2>/dev/null)" ]; then idle=0; touch $STAMP; continue; fi
    idle=$((idle + 15))
    if [ $idle -ge 90 ]; then
      echo "[t466] watchdog: no output and no new cache files for $idle s, ending $ARM" | tee -a $L
      hangs=$((hangs + 1)); stop 10; break
    fi
  done
  wait $PG; arc=$?; PG=
  wait $TAILPID 2>/dev/null; rm -f $STAMP
  echo "[t466] $ARM process wall $(( $(date +%s) - T0 )) s" | tee -a $L
  mv $MP4 $OUT/ 2>/dev/null && md5sum $OUT/$(basename $MP4) | tee -a $L
  echo "T466_EXIT_$ARM=$arc" | tee -a $L
  if [ $arc = 0 ]; then touch $OUT/PASS; else rc=$arc; fi
  [ $hangs -ge 2 ] && { echo "[t466] 2 hangs, stopping"; break; }
  [ -n "$lowdisk" ] && { echo "[t466] space floor hit, stopping"; rc=5; break; }
done
exit $rc
