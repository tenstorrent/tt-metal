#!/bin/bash
# t477/t480 blx01 job: MODE=prof  -> Tracy profile of one warm 10 s 768p MiniMax-H3 block (sp_sim1, 4x8 ring)
#                      MODE=ab    -> ff1 pack-SwiGLU A/B (arms old,new,old in one process), then grep the JIT cache.
# Random weights, no model weights read. pytest runs under timeout --foreground with an output/cache watchdog.
set -o pipefail
MODE=${1:?prof|ab}
F=/var/tmp/fasth3; T=$F/t477; W=$T/b; OUT=$T/out_$MODE; L=$OUT/run.log
free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$free" -ge 260 ] || { echo "[t477] / free ${free}G"; exit 5; }
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 88 ] || { echo "[t477] / at $use%"; exit 5; }
gb=$(timeout 120 du -sxBG $F 2>/dev/null | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 450 ] || { echo "[t477] footprint ${gb}G"; exit 5; }
mkdir -p $OUT $F/tmp $T/jit
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$T/jit TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
source $F/t460/python_env/bin/activate || exit 3
cd $W || exit 3
D=models/tt_dit/tests/models/minimax_h3
K="4x8sp1tp0nl2_ring_is_fsdp0 and not WH"
if [ "$MODE" = prof ]; then
  PY="python -u -m tracy -p -r -v -m pytest -sv -p no:cacheprovider $D/test_transformer_minimax_h3.py::test_minimax_h3_transformer_block_perf -k '$K and 10s_768p and test_prompt_text_tokens and sp_sim1'"
  IDLE=240
else
  PY="python -u -m pytest -sv -p no:cacheprovider $D/test_ff1_swiglu_pack_minimax_h3.py::test_ff1_swiglu_pack_block_10s -k '$K'"
  IDLE=120
fi
CMD="timeout --foreground -k 10 ${T477_PY_TIMEOUT:-560} $PY"
echo "[t477] mode=$MODE host=$(hostname) job=${TTP_RUNNER_JOB:-} $(date -u '+%F %T') UTC" | tee $L
echo "[t477] cmd: $CMD" | tee -a $L
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
STAMP=$T/.wd_$MODE; touch $STAMP; last=$(stat -c %s $L); idle=0
while kill -0 $PG 2>/dev/null; do
  sleep 15
  now=$(stat -c %s $L)
  if [ "$now" != "$last" ]; then last=$now; idle=0; touch $STAMP; continue; fi
  if [ -n "$(timeout 20 find $TT_METAL_CACHE $W/generated -newer $STAMP -print -quit 2>/dev/null)" ]; then idle=0; touch $STAMP; continue; fi
  idle=$((idle + 15))
  if [ $idle -ge $IDLE ]; then
    echo "[t477] watchdog: no output and no new cache/report files for $idle s, ending the job" | tee -a $L
    stop 10; break
  fi
done
wait $PG; rc=$?; PG=
wait $TAILPID 2>/dev/null; rm -f $STAMP
echo "[t477] process wall $(( $(date +%s) - T0 )) s" | tee -a $L
if [ "$MODE" = ab ]; then
  n=$(timeout 60 grep -rl SWIGLU_PACK_THREAD $TT_METAL_CACHE 2>/dev/null | wc -l)
  echo "[t477] JIT cache files naming SWIGLU_PACK_THREAD: $n" | tee -a $L
  timeout 60 grep -rl SWIGLU_PACK_THREAD $TT_METAL_CACHE 2>/dev/null | head -5 | tee -a $L
else
  r=$(ls -td $W/generated/profiler/reports/*/ 2>/dev/null | head -1); echo "[t477] report dir: $r" | tee -a $L
  [ -n "$r" ] && cp $r/ops_perf_results_*.csv $OUT/ 2>/dev/null
fi
echo "T477_EXIT=$rc" | tee -a $L
exit $rc
