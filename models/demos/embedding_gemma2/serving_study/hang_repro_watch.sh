#!/bin/bash
# Run emb_hang_repro.py (args passed through) on chip 86:00.0; if no progress line for 60 s after "model ready",
# run tt-triage against the hung chip (process still alive), then kill it. Output: $E/results/hang_repro_<UTC>_<open>/
source "$(dirname "${BASH_SOURCE[0]}")/env_emb.sh"
OPEN=$(echo "$@" | sed -n 's/.*--open \([a-z]*\).*/\1/p')
R=$E/results/hang_repro_$(date -u +%Y%m%dT%H%M)_$OPEN; mkdir -p $R; LOG=$R/repro.log
[ -z "$(fuser $EMB_DEV 2>/dev/null)" ] || { echo "chip busy"; exit 3; }
# triage needs this process's inspector logs and RPC (port 50051 must be free) and --dev=all under TT_VISIBLE_DEVICES
triage() { (cd $ROOT && TT_METAL_INSPECTOR_LOG_PATH=$ROOT/generated/inspector timeout 900 python3 tools/tt-triage.py --disable-progress --dev=all --triage-summary-path=$R/triage_summary.txt > $R/triage.txt 2>&1); echo "triage exit $? (summary: $(grep -c pass $R/triage_summary.txt) pass, $(grep -c FAIL $R/triage_summary.txt) fail)"; }
ss -ltn | grep -q ":50051 " && { echo "port 50051 in use: inspector RPC would fail"; exit 4; }
cd $EMB_SCRIPTS && python3 emb_hang_repro.py "$@" > $LOG 2>&1 &
PID=$!
last=$(date +%s); lines=0; state=build
while kill -0 $PID 2>/dev/null; do
  sleep 5
  n=$(grep -cE "model ready|forwards|phase|PASS" $LOG)
  if [ "$n" != "$lines" ]; then lines=$n; last=$(date +%s); grep -q "model ready" $LOG && state=run; continue; fi
  limit=60; [ $state = build ] && limit=900
  if [ $(( $(date +%s) - last )) -gt $limit ]; then
    echo "STALL detected at $(date -u +%H:%M:%SZ) after: $(grep -E 'forwards|phase' $LOG | tail -1)"
    ${PY_SPY:-/home/ttuser/.local/bin/py-spy} dump --pid $PID > $R/host_stack.txt 2>&1
    triage
    kill -TERM $PID; sleep 10; kill -KILL $PID 2>/dev/null; break
  fi
done
wait $PID 2>/dev/null; rc=$?; echo "repro exit $rc"
if [ $rc -ne 0 ] && [ ! -f $R/triage.txt ]; then echo "repro failed: running triage on its device state"; grep -iE "timeout|TT_THROW|error" $LOG | tail -5; triage; fi; grep -E "model ready|phase|PASS|forwards" $LOG | tail -4
sleep 3; fuser $EMB_DEV 2>/dev/null && echo "chip 31 still held" || echo "chip 31 released"; echo "-> $R"
