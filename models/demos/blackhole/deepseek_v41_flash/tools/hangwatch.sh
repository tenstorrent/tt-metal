#!/bin/bash
# usage (run ON the host that runs the job): hangwatch.sh <pid> <logfile> [stall_minutes=10] [report_dir=/mnt/tt-data/ssinghal/dsv4-logs/triage]
# Detects a stalled device job: log silent for stall_minutes. Then runs tt-triage IN PARALLEL with the live process (it needs the
# running Metal runtime / inspector logs), saves the report, kills the pid, and exits 3 so the caller can retry. Exits 0 if the pid ended by itself.
pid=$1; log=$2; stall=${3:-10}; out=${4:-/mnt/tt-data/ssinghal/dsv4-logs/triage}; mkdir -p $out
M=/mnt/tt-data/ssinghal/tests/tt-metal
while kill -0 $pid 2>/dev/null; do
  age=$(( ( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || date +%s) ) / 60 ))
  if [ $age -ge $stall ]; then
    rep=$out/hang_$(hostname -s | tr -dc 0-9 | tail -c 2)_${pid}_$(date +%H%M).txt
    echo "[hangwatch] pid $pid log silent ${age} min -> triage -> $rep" | tee -a "$log"
    ( cd $M && source python_env/bin/activate && export TT_METAL_HOME=$M PYTHONPATH=$M && \
      timeout 300 python tools/tt-triage.py --skip-version-check --disable-progress --disable-colors --llm-output-path=$rep > $rep.console 2>&1 )
    echo "[hangwatch] killing pid $pid (report: $rep)" | tee -a "$log"
    kill $pid; sleep 5; kill -9 $pid 2>/dev/null
    # a killed hung job leaves the device dirty (next job hangs on its first op): reset under the device lock before anyone else starts
    source $M/python_env/bin/activate
    # when started from INSIDE the locked command we already hold the lock (inherited fd): taking it again would deadlock
    if ls -l /proc/$$/fd 2>/dev/null | grep -q dsv4_dev.lock; then tt-smi -glx_reset >> "$log" 2>&1; else flock -w 1200 /tmp/dsv4_dev.lock tt-smi -glx_reset >> "$log" 2>&1; fi; echo "[hangwatch] reset rc=$?" | tee -a "$log"
    exit 3
  fi
  sleep 30
done
exit 0
