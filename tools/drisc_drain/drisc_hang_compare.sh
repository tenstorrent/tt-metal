#!/bin/bash
# Instrumented DRISC hang harness.
#
# Scores every run on FOUR axes, because today three separate errors came from scoring on exit code alone:
#   1. DURATION   -- a 9x slowdown (45s vs 5s) hid in ~280 runs that all exited 0
#   2. CARD STATE -- "hang" pooled two failure modes: endpoint wedge vs teardown wait on a healthy card
#   3. MASKED     -- an armed run that exits 0 after a CAUGHT teardown timeout is NOT a clean run
#   4. rc
#
# Classification (card state is authoritative, never exit code):
#   WEDGE    card reads Unknown|63 (all-ones config space) -> genuine PCIe endpoint wedge
#   TEARDOWN run failed or logged the core-wait timeout, card HEALTHY -> wait_until_cores_done, not a wedge
#   MASKED   rc=0 but log has "Continuing with cleanup" -> teardown timeout caught; would hang if unarmed
#   OTHER    failed, card healthy, no teardown signature
#   CLEAN    rc=0, no masked signature
#
# Usage: DELAY=125 N=60 ARMED=0 ./harness.sh

H=${TT_HOST:-yyzo-bh-05}; P=${TT_PORT:-42756}; R=${TT_REMOTE:-/localdev/$LOGNAME/tt-metal}
BIN=./build_Release/programming_examples/test_streaming_profiler_zones
TAG=${TAG:-run}
OUT=${OUT_DIR:-${TMPDIR:-/tmp}}/harn_$TAG; mkdir -p $OUT
SUM=$OUT/summary.txt; CSV=$OUT/runs.csv; : > $SUM
echo "k,delay,armed,rc,dur_s,card,class" > $CSV
DELAY=${DELAY:-125}; N=${N:-40}   # N = runs PER ARM
STOP_ON_WEDGE=0
REPEAT=${REPEAT:-0}
REPX=""
[ "$REPEAT" != "0" ] && REPX="TT_METAL_STREAMING_PROFILER_SHIP_REPEAT=$REPEAT"

# randomized 2-arm schedule (never alternate: a boundary-straddling failure would bias attribution)
sched=()
for a in 0 1; do for i in $(seq 1 $N); do sched+=($a); done; done
tot=${#sched[@]}
for ((i=tot-1; i>0; i--)); do j=$((RANDOM % (i+1))); t=${sched[i]}; sched[i]=${sched[j]}; sched[j]=$t; done

card_state(){ ssh -o ConnectTimeout=10 $H 'D=/sys/bus/pci/devices/0000:01:00.0; echo "$(cat $D/current_link_speed 2>&1)|$(cat $D/max_link_width 2>&1)"' 2>/dev/null; }

recover(){ ssh -p $P -o ConnectTimeout=10 $H 'tt-smi -r' >/dev/null 2>&1; sleep 8
  st=$(card_state)
  if [ "${st%%|*}" = "Unknown" ]; then echo "  [recover] reset insufficient -> reboot" | tee -a $SUM
    ssh -o ConnectTimeout=10 $H 'sudo reboot' >/dev/null 2>&1; sleep 30
    until ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $H 'true' 2>/dev/null; do sleep 15; done
    until ssh -p $P -o ConnectTimeout=5 $H 'true' 2>/dev/null; do sleep 15; done; sleep 10; st=$(card_state); fi
  echo "  [recover] card=$st" | tee -a $SUM; }

echo "=== HARNESS: delay=$DELAY armed=${ARMED:-mixed} N=$N repeat=$REPEAT ===" | tee -a $SUM
echo "class rules: WEDGE=card Unknown|63 | TEARDOWN=core-wait, card healthy | MASKED=rc0 but caught timeout" | tee -a $SUM
echo "start card=$(card_state)" | tee -a $SUM

nclean=0; nwedge=0; nteardown=0; nmasked=0; nother=0
k=0
for ARMED in "${sched[@]}"; do
  k=$((k+1))
  ENVX=""
  [ "$ARMED" = "1" ] && ENVX="TT_METAL_OPERATION_TIMEOUT_SECONDS=45"
  log=$OUT/${k}_a${ARMED}.log
  t0=$(python3 -c 'import time;print(time.time())')
  ssh -p $P -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=6 $H "cd $R && timeout -k 15 300 env \
    TT_METAL_STREAMING_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_STREAMING_PROFILER_NO_DECODE=1 $ENVX $REPX \
    $BIN --gx 0 --gy 0 --iters 500 --delay $DELAY" > $log 2>&1 &
  sshpid=$!
  waited=0
  while kill -0 $sshpid 2>/dev/null; do
    sleep 2; waited=$((waited+2))
    if [ $waited -ge 360 ]; then
      echo "  [watchdog] local ssh stuck ${waited}s -- killing client" | tee -a $SUM
      kill -9 $sshpid 2>/dev/null; sleep 1; break
    fi
  done
  wait $sshpid 2>/dev/null; rc=$?
  t1=$(python3 -c 'import time;print(time.time())')
  dur=$(python3 -c "print(f'{$t1-$t0:.1f}')")

  card="-"; class="CLEAN"
  teardown_sig=0
  grep -q "waiting for physical cores to finish\|Continuing with cleanup" $log 2>/dev/null && teardown_sig=1
  # How far the log got is the ROBUST discriminator. The teardown-timeout message is only emitted
  # when the timeout is ARMED; an unarmed run hangs at the same place silently, so keying TEARDOWN
  # off the signature alone dumps every unarmed teardown hang into OTHER.
  reached_end=0
  grep -q "Cluster destructor completed" $log 2>/dev/null && reached_end=1

  if [ $rc -ne 0 ]; then
    card=$(card_state)
    if [ "${card%%|*}" = "Unknown" ]; then class="WEDGE"; nwedge=$((nwedge+1))
    elif [ $teardown_sig -eq 1 ] || [ $reached_end -eq 0 ]; then class="TEARDOWN"; nteardown=$((nteardown+1))
    else class="OTHER"; nother=$((nother+1)); fi
  elif [ $teardown_sig -eq 1 ]; then
    class="MASKED"; nmasked=$((nmasked+1)); card=$(card_state)
  else
    nclean=$((nclean+1))
  fi

  echo "$k,$DELAY,$ARMED,$rc,$dur,$card,$class" >> $CSV
  if [ "$class" != "CLEAN" ]; then
    echo "k=$k  rc=$rc  dur=${dur}s  card=$card  -> $class" | tee -a $SUM
  fi
  if [ "$class" = "WEDGE" ] || [ "$class" = "TEARDOWN" ] || [ "$class" = "OTHER" ]; then
    recover
    [ "$STOP_ON_WEDGE" = "1" ] && [ "$class" = "WEDGE" ] && { echo "=== STOPPING: wedge captured ===" | tee -a $SUM; break; }
  fi
  [ $((k % 20)) -eq 0 ] && echo "progress k=$k/$tot clean=$nclean wedge=$nwedge teardown=$nteardown masked=$nmasked other=$nother" | tee -a $SUM
done

echo "=== DONE ===" | tee -a $SUM
echo "clean=$nclean wedge=$nwedge teardown=$nteardown masked=$nmasked other=$nother" | tee -a $SUM
python3 - "$CSV" <<'PY2'
import sys,csv
rows=list(csv.DictReader(open(sys.argv[1])))
d=[float(r['dur_s']) for r in rows]
if d:
    d2=sorted(d)
    print(f"duration: min={d2[0]:.1f}s median={d2[len(d2)//2]:.1f}s max={d2[-1]:.1f}s  n={len(d2)}")
PY2

echo "======== PER-ARM BREAKDOWN ========" | tee -a $SUM
python3 - "$CSV" <<'PYX' | tee -a $SUM
import sys,csv,collections
rows=list(csv.DictReader(open(sys.argv[1])))
for armed in ('0','1'):
    r=[x for x in rows if x['armed']==armed]
    c=collections.Counter(x['class'] for x in r)
    d=sorted(float(x['dur_s']) for x in r)
    lbl='UNARMED' if armed=='0' else 'ARMED  '
    med=d[len(d)//2] if d else 0
    print(f"{lbl} n={len(r):3d}  WEDGE={c['WEDGE']:2d}  TEARDOWN={c['TEARDOWN']:2d}  MASKED={c['MASKED']:2d}  CLEAN={c['CLEAN']:2d}  median_dur={med:.1f}s")
PYX
