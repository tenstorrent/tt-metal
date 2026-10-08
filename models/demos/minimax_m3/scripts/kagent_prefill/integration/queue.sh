#!/bin/bash
# Sequential device-job queue for the prefill partition (integration agent). Each line of $1 is a wrapper command
# "<wrapper> <tag> <timeout> ...". Stops on HANG (75) / dirty (76), and on the UMD warning signs the device rules name
# (a CHIP_IN_USE lock wait or a hugepage NOC-address mismatch in the job log) -- never retries.
cd /mnt/data/kernel-agent/dev/prefill-best
while read -r line; do
  [ -z "$line" ] && continue; [[ "$line" == \#* ]] && continue
  [ -e STOP-QUEUE ] && { echo "[queue] STOP-QUEUE present -> stopping"; exit 0; }
  echo "[queue] $(date +%T) start: $line"
  bash -c "$line"; rc=$?
  echo "[queue] $(date +%T) rc=$rc: $line"
  if [ $rc -eq 75 ] || [ $rc -eq 76 ]; then echo "[queue] HANG/DIRTY -> stopping all silicon work"; exit $rc; fi
  tag=$(echo "$line" | awk '{print $2}')
  if grep -qE "NOC address of a hugepage does not match|Waiting for lock 'CHIP_IN_USE" runs/$tag/log.txt 2>/dev/null; then
    echo "[queue] UMD lock-wait / hugepage warning in runs/$tag/log.txt -> stopping (report)"; exit 90; fi
done < "$1"
echo "[queue] done"
