#!/bin/bash
# t455: one blx01 broker job on the full 4x8 mesh running test_sla_device_minimax_h3.py (SLA coarse stage,
# topk_large_indices and vsa_sdpa at the 10 s 768p geometry, random Q/K/V) in /var/tmp/fasth3/t455/w, a symlink
# overlay of the t452/w tree (the one the SLA e2e arms ran) plus that test file (mk455.sh).
# pytest runs in the broker job's process group under timeout --foreground -k 10; a 90 s no-progress watchdog
# (no log output and no new cache files) stops it by pid.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t455; W=$T/w; VENV=$F/t48/python_env; L=$T/run.log
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 88 ] || { echo "[t455] / at $use%"; exit 5; }
gb=$(timeout 120 du -sxBG /var/tmp/fasth3 | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 450 ] || { echo "[t455] /var/tmp/fasth3 at ${gb}G"; exit 5; }
mkdir -p $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3hf TT_DIT_CACHE_DIR=$F/cache/dit-h3hf
cd $W || exit 3
[ -e models/tt_dit/tests/models/minimax_h3/test_sla_device_minimax_h3.py ] || { echo "[t455] no test in $W"; exit 3; }
source $VENV/bin/activate || exit 3
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools:$F/t409/pydeps
CMD="exec timeout --foreground -k 10 ${T455_PY_TIMEOUT:-540} python -u -m pytest -sv -p no:cacheprovider models/tt_dit/tests/models/minimax_h3/test_sla_device_minimax_h3.py -k 4x8sp1tp0nl2_ring"
echo "[t455] host=$(hostname) $(cat $W/OVERLAY) job=${TTP_RUNNER_JOB:-} $(date -u '+%F %T') UTC" | tee $L
echo "[t455] cmd: $CMD" | tee -a $L
PG=""
stop_py() { local kids; kids=$(ps -o pid= --ppid $PG 2>/dev/null); kill -TERM $PG $kids 2>/dev/null; sleep 10; kill -KILL $PG $kids 2>/dev/null; }
trap '[ -n "$PG" ] && stop_py' EXIT
trap 'exit 143' TERM; trap 'exit 130' INT
bash -c "$CMD" >> $L 2>&1 < /dev/null & PG=$!
tail -n +1 -f --pid=$PG $L & TAILPID=$!
STAMP=$T/.wd; touch $STAMP; last=$(stat -c %s $L); idle=0
while kill -0 $PG 2>/dev/null; do
  sleep 15
  now=$(stat -c %s $L)
  if [ "$now" != "$last" ]; then last=$now; idle=0; touch $STAMP; continue; fi
  if [ -n "$(timeout 20 find $TT_METAL_CACHE -newer $STAMP -print -quit 2>/dev/null)" ]; then idle=0; touch $STAMP; continue; fi
  idle=$((idle + 15))
  if [ $idle -ge 90 ]; then echo "[t455] watchdog: no progress for $idle s" | tee -a $L; stop_py; break; fi
done
wait $PG; rc=$?; PG=""
wait $TAILPID 2>/dev/null; rm -f $STAMP
echo "T455_EXIT=$rc" | tee -a $L
exit $rc
