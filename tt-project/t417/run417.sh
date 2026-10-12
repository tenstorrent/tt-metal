#!/bin/bash
# t417/t452 device job on one exabox dit BH Galaxy (full 4x8 mesh). Submit only through submit417.sh, e.g.
#   TLIM=00:10:00 bash submit417.sh run417.sh --export=ALL,TAG=s0-vsa,SEED=0,ARMS="dense vsa0.9"
# Runs the Turbo e2e test (test_pipeline_turbo_minimax_h3.py), 768p fl2va 10 s, 4 steps, one pytest per arm, in
# order. Adapted from t409/run409b.sh (blx01): same inputs as #408/#409 (kf_t2va_f0.png, fixed fox prompt).
# Arms: dense (v1.0 LoRA), vsa<s> (v1.0 LoRA + VSA sparsity s), sla<s> (SLA LoRA + MINIMAX_H3_VSA_MODE=sla, sparsity s),
# fill (v1.0 LoRA, MINIMAX_H3_TURBO_CACHE_ONLY=1: writes the weight caches only).
# T417_PY_TIMEOUT per-arm limit (default 270 s) under timeout --foreground -k 10; a 90 s no-progress watchdog
# (log output or new cache files) and the EXIT trap stop the arm by pid. No local-disk check: /data is NFS here.
#SBATCH -J smarton-fasth3-t417
#SBATCH -N 1
#SBATCH -o /data/smarton/fasth3/t417/slurm-%j.out
set -o pipefail
F=/data/smarton/fasth3; T=$F/t417; W=$T/src; VENV=$F/python_env
RC=$T/job-$SLURM_JOB_ID.rc
TAG=${TAG:?tag}; SEED=${SEED:-0}; read -ra ARMV <<<"${ARMS:?arms}"
for a in "${ARMV[@]}"; do case $a in dense|fill|vsa0.[0-9]*|sla0.[0-9]*) ;; *) echo "[t417] bad arm $a"; exit 2;; esac; done
PG=""
stop_arm() {  # TERM, then KILL, $PG and its direct children
  local kids; kids=$(ps -o pid= --ppid $PG 2>/dev/null)
  kill -TERM $PG $kids 2>/dev/null; sleep ${1:-5}; kill -KILL $PG $kids 2>/dev/null
}
trap 'rc=$?; [ -n "$PG" ] && stop_arm; echo "JOB_RC=$rc $(date -u +%FT%TZ)" > $RC' EXIT
trap 'exit 143' TERM; trap 'exit 130' INT
echo "[t417] start node=$(hostname) job=$SLURM_JOB_ID tag=$TAG seed=$SEED arms=${ARMV[*]} $(date -u '+%F %T') UTC"
avail=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9); echo "[t417] /data free ${avail}G"
[ "$avail" -ge 300 ] || { echo "[t417] /data below 300G"; exit 5; }
nd=$(ls /dev/tenstorrent 2>/dev/null | wc -l); echo "[t417] devices $nd"; [ "$nd" -ge 32 ] || { echo "[t417] <32 devices"; exit 4; }
mkdir -p $T/tmp $T/home
export HOME=$T/home XDG_CACHE_HOME=$T/home/.cache TMPDIR=$T/tmp TORCH_HOME=$T/home/.cache/torch HF_HOME=$T/home/.cache/huggingface HF_HUB_OFFLINE=1
# Same layout and transformer/vae/text_encoder configs as blx01's /var/tmp/fasth3/models/MiniMax-H3 (#408 baseline).
export MINIMAX_H3_MODEL_PATH=$T/models/MiniMax-H3
[ -e $MINIMAX_H3_MODEL_PATH/READ_OK ] || { echo "[t417] no READ_OK in $MINIMAX_H3_MODEL_PATH"; exit 4; }
LORA_V10=$T/models/lightx2v-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
LORA_SLA=$T/models/lightx2v-h3-turbo-sla/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_bf16.safetensors
export MINIMAX_H3_TURBO_KEYFRAME=$T/kf_t2va_f0.png MINIMAX_H3_TURBO_TASK=fl2va MINIMAX_H3_TURBO_POINT=768p MINIMAX_H3_TURBO_NFE=4
export MINIMAX_H3_TURBO_SEED=$SEED MINIMAX_H3_TURBO_REPLAYS=${T417_REPLAYS:-1}
for p in $LORA_V10 $LORA_SLA $MINIMAX_H3_TURBO_KEYFRAME; do [ -r $p ] || { echo "[t417] missing $p"; exit 4; }; done
# 10 s = 243 frames, 75449 packed rows, dense rung 76800 (#408). Under VSA/SLA the pipeline uses geometry.padded_len.
export MINIMAX_H3_TURBO_WARM_RUNGS=${T417_RUNGS:-76800} MINIMAX_H3_TURBO_WARM_USED_LAYOUTS=1
export TT_METAL_CACHE=$T/cache/tt-metal-cache TT_DIT_CACHE_DIR=$T/cache/dit-h3
cd $W || exit 3
grep -q MINIMAX_H3_TURBO_SAVE_DIR models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py || { echo "[t417] $W lacks the save flag"; exit 3; }
grep -q MINIMAX_H3_VSA_MODE models/tt_dit/pipelines/minimax_h3/pipeline_minimax_h3.py || { echo "[t417] $W lacks SLA mode"; exit 3; }
[ -f ttnn/ttnn/_ttnn.so ] || { echo "[t417] $W not built"; exit 3; }
source $VENV/bin/activate || exit 3
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools:$T/pydeps:$T/plug
export T286_WARM_DEADLINE_S=${T286_WARM_DEADLINE_S:-400} T286_WARM_LATE_S=${T286_WARM_LATE_S:-400} T286_SKIP_VAE_WARM=1
MP4=$HOME/h3_turbo_artifacts/fl2va_turbo_1344x768_10s_4fwd.mp4
rc=0
for ARM in "${ARMV[@]}"; do
  OUT=$T/out_$TAG-$ARM; L=$OUT/run.log; mkdir -p $OUT
  if [ -e $OUT/PASS ]; then echo "[t417] $TAG-$ARM already passed"; continue; fi
  unset MINIMAX_H3_VSA_SPARSITY MINIMAX_H3_VSA_MODE MINIMAX_H3_TURBO_CACHE_ONLY MINIMAX_H3_TURBO_SAVE_DIR
  export MINIMAX_H3_TURBO_LORA_PATH=$LORA_V10
  case $ARM in
    vsa*) export MINIMAX_H3_VSA_SPARSITY=${ARM#vsa};;
    sla*) export MINIMAX_H3_VSA_SPARSITY=${ARM#sla} MINIMAX_H3_VSA_MODE=sla MINIMAX_H3_TURBO_LORA_PATH=$LORA_SLA;;
    fill) export MINIMAX_H3_TURBO_CACHE_ONLY=1;;
  esac
  [ $ARM != fill ] && export MINIMAX_H3_TURBO_SAVE_DIR=$OUT
  rm -f $MP4
  PYT=${T417_PY_TIMEOUT:-270}
  CMD="exec timeout --foreground -k 10 $PYT python -u -m pytest -sv -p no:cacheprovider -p t286_skipvaewarm -p t409_alarm models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py -k '10s and 4x8 and not 15s and not 4x32 and not WH'"
  echo "[t417] host=$(hostname) commit=$(git -C $W rev-parse HEAD) tag=$TAG arm=$ARM seed=$SEED job=$SLURM_JOB_ID $(date -u '+%F %T') UTC" | tee $L
  echo "[t417] cmd: $CMD" | tee -a $L
  env | grep -E '^(MINIMAX|TT_|T286_)' | sort >> $L
  export T409_ALARM_S=$(( PYT - 35 ))
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
      echo "[t417] watchdog: no output and no new cache files for $idle s, ending the arm" | tee -a $L
      stop_arm 10; break
    fi
  done
  wait $PG; arc=$?; PG=""
  wait $TAILPID 2>/dev/null; rm -f $STAMP
  echo "[t417] process wall $(( $(date +%s) - T0 )) s" | tee -a $L
  mv $MP4 $OUT/ 2>/dev/null
  echo "T417_EXIT=$arc" | tee -a $L
  if [ $arc = 0 ]; then touch $OUT/PASS; else rc=$arc; break; fi
done
exit $rc
