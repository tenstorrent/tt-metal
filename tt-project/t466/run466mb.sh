#!/bin/bash
# t466: one-layer ring SDPA microbench, HiFi2 vs LoFi matmuls interleaved in one process (random weights,
# 10 s fl2va shape, full 4x8 mesh). Build: /var/tmp/fasth3/t466/b (e02052af0ee). Needs no model weights.
# pytest runs in the broker's process group under timeout --foreground -k 10 with a 90 s output watchdog.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t466; W=$T/b; OUT=$T/out_mb; L=$OUT/run.log
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 88 ] || { echo "[t466] / at $use%"; exit 5; }
gb=$(timeout 120 du -sxBG $F 2>/dev/null | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 450 ] || { echo "[t466] footprint ${gb}G"; exit 5; }
mkdir -p $OUT $F/tmp $T/jit
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$T/jit TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
source $T/../t460/python_env/bin/activate || exit 3
cd $W || exit 3
CMD="timeout --foreground -k 10 ${T466_PY_TIMEOUT:-560} python -u -m pytest -sv -p no:cacheprovider models/tt_dit/tests/models/minimax_h3/test_sdpa_fidelity_timing_minimax_h3.py -k '4x8sp1tp0nl2_ring_is_fsdp0 and not WH'"
echo "[t466] host=$(hostname) job=${TTP_RUNNER_JOB:-} $(date -u '+%F %T') UTC rounds=${T466_ROUNDS:-3}" | tee $L
echo "[t466] cmd: $CMD" | tee -a $L
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
STAMP=$T/.wd_mb; touch $STAMP; last=$(stat -c %s $L); idle=0
while kill -0 $PG 2>/dev/null; do
  sleep 15
  now=$(stat -c %s $L)
  if [ "$now" != "$last" ]; then last=$now; idle=0; touch $STAMP; continue; fi
  if [ -n "$(timeout 20 find $TT_METAL_CACHE -newer $STAMP -print -quit 2>/dev/null)" ]; then idle=0; touch $STAMP; continue; fi
  idle=$((idle + 15))
  if [ $idle -ge 90 ]; then
    echo "[t466] watchdog: no output and no new cache files for $idle s, ending the job" | tee -a $L
    stop 10; break
  fi
done
wait $PG; rc=$?; PG=
wait $TAILPID 2>/dev/null; rm -f $STAMP
echo "[t466] process wall $(( $(date +%s) - T0 )) s" | tee -a $L
echo "T466_EXIT=$rc" | tee -a $L
exit $rc
