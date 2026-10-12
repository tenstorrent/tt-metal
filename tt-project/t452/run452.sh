#!/bin/bash
# t452 (from t409/run409b.sh): one blx01 broker job running the Turbo e2e test (test_pipeline_turbo_minimax_h3.py),
# 768p, fl2va, 10 s, 4 steps, 4x8, for each arm given, one pytest process per arm, in order. Tree: the t452/w overlay
# (t409/b build + SLA selection code, see mk452.sh). One arm takes ~316 s warm, so a job runs one arm.
# Args: <tag> <arm>... where arm is dense (v1.0 LoRA), vsa<s> (v1.0 LoRA + VSA sparsity s, e.g. vsa0.9) or
# sla<s> (SLA LoRA v0.1 + MINIMAX_H3_VSA_MODE=sla, sparsity s).
# T409_SEED (default 0), T409_REPLAYS (default 1), T409_SAVE=1 saves latents and video per arm,
# T409_PY_TIMEOUT per-arm pytest limit (default 270). Same inputs as t408/run408.sh (#408 baseline).
# pytest runs in the broker job's process group (no setsid) under timeout --foreground -k 10; the EXIT trap and
# a 90 s no-progress watchdog (log output or new cache files) stop timeout and its python child by pid.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t452; W=${T409_W:-$T/w}; VENV=$F/t48/python_env
TAG=${1:?tag}; shift; [ $# -ge 1 ] || { echo "[t409] no arms"; exit 2; }
for a in "$@"; do case $a in dense|vsa0.[0-9]*|sla0.[0-9]*) ;; *) echo "[t409] bad arm $a"; exit 2;; esac; done
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 88 ] || { echo "[t409] / at $use%"; exit 5; }
gb=$(timeout 120 du -csxBG $F 2>/dev/null | tail -1 | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 450 ] || { echo "[t409] footprint ${gb}G"; exit 5; }
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
export MINIMAX_H3_MODEL_PATH=$F/models/MiniMax-H3
[ -e $MINIMAX_H3_MODEL_PATH/READ_OK ] || { echo "[t409] no READ_OK in $MINIMAX_H3_MODEL_PATH"; exit 4; }
LORA_V10=$F/models/lightx2v-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
LORA_SLA=$F/models/lightx2v-h3-turbo-sla/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_bf16.safetensors
for p in $LORA_V10 $LORA_SLA; do [ -r $p ] || { echo "[t409] no adapter $p"; exit 4; }; done
# Matched FastH3 baseline inputs (#330/#408): keyframe kf_t2va_f0.png, CALIBRATED_FOX_PROMPT (fixed in the test).
export MINIMAX_H3_TURBO_KEYFRAME=$F/t409/kf_t2va_f0.png MINIMAX_H3_TURBO_TASK=fl2va MINIMAX_H3_TURBO_POINT=768p MINIMAX_H3_TURBO_NFE=4
export MINIMAX_H3_TURBO_SEED=${T409_SEED:-0} MINIMAX_H3_TURBO_REPLAYS=${T409_REPLAYS:-1}
for p in $MINIMAX_H3_MODEL_PATH $LORA_V10 $LORA_SLA $MINIMAX_H3_TURBO_KEYFRAME; do
  r=$(readlink -f "$p"); fs=$(findmnt -n -o FSTYPE -T "$r" 2>/dev/null)
  case "$r" in /mnt/*) echo "[t409] $p resolves to $r under /mnt"; exit 4;; esac
  case "$fs" in nfs*|cifs|smb*|fuse*|sshfs|lustre|ceph*|9p|"") echo "[t409] $p is on fs '${fs:-unknown}', not local disk"; exit 4;; esac
done
# 10 s = 243 frames, 75449 packed rows, dense rung 76800 (#408). Under VSA the pipeline uses geometry.padded_len.
export MINIMAX_H3_TURBO_WARM_RUNGS=${T409_RUNGS:-76800} MINIMAX_H3_TURBO_WARM_USED_LAYOUTS=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3hf TT_DIT_CACHE_DIR=$F/cache/dit-h3hf
cd $W || exit 3
grep -q MINIMAX_H3_TURBO_SAVE_DIR models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py || { echo "[t409] $W lacks the save flag"; exit 3; }
grep -q MINIMAX_H3_VSA_SPARSITY models/tt_dit/pipelines/minimax_h3/pipeline_minimax_h3.py || { echo "[t409] $W lacks VSA"; exit 3; }
grep -q MINIMAX_H3_VSA_MODE models/tt_dit/pipelines/minimax_h3/pipeline_minimax_h3.py || { echo "[t409] $W lacks SLA mode"; exit 3; }
source $VENV/bin/activate || exit 3
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools:$F/t286:$F/t409/plug:$T/plug
export T286_WARM_DEADLINE_S=${T286_WARM_DEADLINE_S:-400} T286_WARM_LATE_S=${T286_WARM_LATE_S:-400} T286_SKIP_VAE_WARM=1
MP4=$HOME/h3_turbo_artifacts/fl2va_turbo_1344x768_10s_4fwd.mp4
PG=""
stop_arm() {  # TERM, then KILL, $PG and its direct children
  local kids; kids=$(ps -o pid= --ppid $PG 2>/dev/null)
  kill -TERM $PG $kids 2>/dev/null; sleep ${1:-5}; kill -KILL $PG $kids 2>/dev/null
}
trap '[ -n "$PG" ] && stop_arm' EXIT
trap 'exit 143' TERM; trap 'exit 130' INT
rc=0
for ARM in "$@"; do
  OUT=$T/out_$TAG-$ARM; L=$OUT/run.log; mkdir -p $OUT
  if [ -e $OUT/PASS ]; then echo "[t409] $TAG-$ARM already passed"; continue; fi
  unset MINIMAX_H3_VSA_SPARSITY MINIMAX_H3_VSA_MODE; export MINIMAX_H3_TURBO_LORA_PATH=$LORA_V10
  case $ARM in
    vsa*) export MINIMAX_H3_VSA_SPARSITY=${ARM#vsa};;
    sla*) export MINIMAX_H3_VSA_SPARSITY=${ARM#sla} MINIMAX_H3_VSA_MODE=sla MINIMAX_H3_TURBO_LORA_PATH=$LORA_SLA;;
  esac
  unset MINIMAX_H3_TURBO_SAVE_DIR; [ "${T409_SAVE:-1}" = 1 ] && export MINIMAX_H3_TURBO_SAVE_DIR=$OUT
  rm -f $MP4
  CMD="exec timeout --foreground -k 10 ${T409_PY_TIMEOUT:-270} python -u -m pytest -sv -p no:cacheprovider -p t286_skipvaewarm -p t409_alarm -p t452_audiowarn models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py -k '10s and 4x8 and not 15s and not 4x32 and not WH'"
  echo "[t409] host=$(hostname) commit=$(git -C $W rev-parse HEAD) $(cat $W/OVERLAY) tag=$TAG arm=$ARM seed=$MINIMAX_H3_TURBO_SEED replays=$MINIMAX_H3_TURBO_REPLAYS job=${TTP_RUNNER_JOB:-} $(date -u '+%F %T') UTC" | tee $L
  echo "[t409] cmd: $CMD" | tee -a $L
  env | grep -E '^(MINIMAX|TT_|T286_)' | sort >> $L
  export T409_ALARM_S=${T409_ALARM_S:-$(( ${T409_PY_TIMEOUT:-270} - 35 ))}
  T0=$(date +%s)
  bash -c "$CMD" >> $L 2>&1 < /dev/null & PG=$!
  tail -n +1 -f --pid=$PG $L & TAILPID=$!
  STAMP=$T/.wd_$TAG; touch $STAMP; last=$(stat -c %s $L); idle=0
  while kill -0 $PG 2>/dev/null; do
    sleep 15
    now=$(stat -c %s $L)
    if [ "$now" != "$last" ]; then last=$now; idle=0; touch $STAMP; continue; fi
    # Silent cold-JIT compiles and weight conversion still write to the caches.
    if [ -n "$(timeout 20 find $TT_METAL_CACHE $TT_DIT_CACHE_DIR -newer $STAMP -print -quit 2>/dev/null)" ]; then idle=0; touch $STAMP; continue; fi
    idle=$((idle + 15))
    if [ $idle -ge 90 ]; then
      echo "[t409] watchdog: no output and no new cache files for $idle s, ending the arm" | tee -a $L
      stop_arm 10; break
    fi
  done
  wait $PG; arc=$?; PG=""
  wait $TAILPID 2>/dev/null; rm -f $STAMP
  echo "[t409] process wall $(( $(date +%s) - T0 )) s" | tee -a $L
  mv $MP4 $OUT/ 2>/dev/null
  echo "T409_EXIT=$arc" | tee -a $L
  if [ $arc = 0 ]; then touch $OUT/PASS; else rc=$arc; break; fi
done
exit $rc
