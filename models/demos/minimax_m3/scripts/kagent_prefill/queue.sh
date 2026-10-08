#!/bin/bash
# Sequential device-job queue for the prefill partition. Each line of $1 is a command (run-bench.sh / run-prof.sh /
# tt-partition-run ...). Stops on a HANG (75) or dirty flag (76) -- never retries.
cd /mnt/data/kernel-agent/dev/prefill
while read -r line; do
  [ -z "$line" ] && continue; [[ "$line" == \#* ]] && continue
  echo "[queue] $(date +%T) start: $line"
  bash -c "$line"; rc=$?
  echo "[queue] $(date +%T) rc=$rc: $line"
  if [ $rc -eq 75 ] || [ $rc -eq 76 ]; then echo "[queue] HANG/DIRTY -> stopping all silicon work"; exit $rc; fi
done < "$1"
echo "[queue] done"
