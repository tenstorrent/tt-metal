#!/bin/bash
# Detached on blx03: queue the VAE A/B job, then the block A/B job only after it exits 0. One project job
# at a time (submit.sh refuses while one is live; retried every 2 min for 3 h). Writes DRIVE82_DONE <status> when finished.
V=/var/tmp/fasth3/t82; S=/home/smarton/fasth3/tt-metal/tmp/blx03/submit.sh
cd /home/smarton/fasth3/tt-metal
wait_job() {  # broker job id -> exit code, polling the broker every 30 s
  while tt-device-mcp status -j $1 2>&1 | grep -qiE "^Status: *(running|queued|pending)"; do sleep 30; done
  tt-device-mcp status -j $1 2>&1 | tee -a $V/drive82.log | sed -nE 's/^Exit( code)?: *([0-9-]+).*/\2/p' | head -1
}
for mode in vae block; do
  for try in $(seq 90); do out=$($S 1500 bash /home/smarton/fasth3/t48/tmp/t82/run82.sh $mode); r=$?; [ $r = 75 ] || break; sleep 120; done
  id=$(echo "$out" | tail -1); echo "[drive82] $mode submit rc=$r job=$id" >> $V/drive82.log
  [ $r = 0 ] || { echo "DRIVE82_DONE submit_failed_$mode" >> $V/drive82.log; exit 1; }
  ec=$(wait_job $id); echo "[drive82] $mode job=$id exit=$ec" >> $V/drive82.log
  [ "$ec" = 0 ] || { echo "DRIVE82_DONE fail_$mode job=$id exit=$ec" >> $V/drive82.log; exit 1; }
done
echo "DRIVE82_DONE ok" >> $V/drive82.log
