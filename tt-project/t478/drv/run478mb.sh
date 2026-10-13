#!/bin/bash
# t478: one-layer attention microbench A/B (base ae9534f5fb2 vs bundle 56df249a1f6), one broker job:
# arm base, then arm bundle (each its own TT_METAL_HOME and TT_METAL_CACHE, T466_ROUNDS=10, full 4x8 mesh,
# random weights), then the host compare test. Each pytest runs under timeout --foreground with a 90 s
# output watchdog.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t478; OUT=$T/out_mb; L=$OUT/run.log
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 88 ] || { echo "[t478] / at $use%"; exit 5; }
gb=$(timeout 120 du -sxBG $F 2>/dev/null | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 450 ] || { echo "[t478] footprint ${gb}G"; exit 5; }
mkdir -p $OUT $F/tmp $T/jit_base $T/jit_bundle
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp HF_HUB_OFFLINE=1 T466_ROUNDS=10
source $F/t460/python_env/bin/activate || exit 3
TEST=models/tt_dit/tests/models/minimax_h3/test_sdpa_fidelity_timing_minimax_h3.py
echo "[t478] host=$(hostname) job=${TTP_RUNNER_JOB:-} $(date -u '+%F %T') UTC rounds=$T466_ROUNDS" | tee $L
PG=
stop() {
  [ -n "$PG" ] || return 0
  local kids; kids=$(cat /proc/$PG/task/*/children 2>/dev/null)
  kill -TERM $PG 2>/dev/null; sleep "$1"; kill -KILL $PG $kids 2>/dev/null
}
trap 'stop 5' EXIT
trap 'exit 143' TERM; trap 'exit 130' INT

# run <name> <tree> <cache> <timeout_s> <pytest args...>
run() {
  local name=$1 W=$T/$2 cache=$3 to=$4; shift 4
  export TT_METAL_HOME=$W TT_METAL_CACHE=$cache PYTHONPATH=$W:$W/ttnn:$W/tools
  cd $W || return 3
  local CMD="timeout --foreground -k 10 $to python -u -m pytest -sv -p no:cacheprovider $*"
  echo "[t478] $name cmd: $CMD" | tee -a $L
  local T0=$(date +%s) last idle=0 STAMP=$T/.wd_mb rc
  bash -c "exec $CMD" >> $L 2>&1 < /dev/null & PG=$!
  touch $STAMP; last=$(stat -c %s $L)
  while kill -0 $PG 2>/dev/null; do
    sleep 15
    now=$(stat -c %s $L)
    if [ "$now" != "$last" ]; then last=$now; idle=0; touch $STAMP; continue; fi
    if [ -n "$(timeout 20 find $cache -newer $STAMP -print -quit 2>/dev/null)" ]; then idle=0; touch $STAMP; continue; fi
    idle=$((idle + 15))
    if [ $idle -ge 90 ]; then
      echo "[t478] watchdog: $name no output and no new cache files for $idle s, ending it" | tee -a $L
      stop 10; break
    fi
  done
  wait $PG; rc=$?; PG=
  rm -f $STAMP
  echo "[t478] $name process wall $(( $(date +%s) - T0 )) s rc=$rc" | tee -a $L
  return $rc
}

tail -n +1 -f --pid=$$ $L &
K="test_attention_layer_dump_10s and 4x8sp1tp0nl2_ring_is_fsdp0 and not WH"
rm -f $OUT/base.pt $OUT/bundle.pt
T478_DUMP=$OUT/base.pt run base base $T/jit_base 260 "$TEST -k '$K'"; rb=$?
T478_DUMP=$OUT/bundle.pt run bundle bundle $T/jit_bundle 260 "$TEST -k '$K'"; rn=$?
rc=0
if [ $rb = 0 ] && [ $rn = 0 ] && [ -s $OUT/base.pt ] && [ -s $OUT/bundle.pt ]; then
  T478_BASE_DUMP=$OUT/base.pt T478_BUNDLE_DUMP=$OUT/bundle.pt \
    run compare bundle $T/jit_bundle 60 "$TEST::test_attention_layer_dumps_bit_identical"; rc=$?
else
  echo "[t478] arm failed: base=$rb bundle=$rn; compare skipped" | tee -a $L; rc=1
fi
echo "T478_EXIT=$rc base=$rb bundle=$rn" | tee -a $L
exit $rc
