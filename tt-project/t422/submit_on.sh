#!/bin/bash
# t422: queue the on arm behind other smarton jobs on blx01 (blx01-submit.sh refuses with 75 while one runs),
# wait for it to end, then copy its result lines into $D/on.log. Writes T422ON_DONE <rc> last.
set -eo pipefail
cd /home/smarton/fasth3/tt-metal
S=${T422_SPECS:?}; D=${T422_STATE:?}; L=$D/on.log
trap 'rc=$?; echo "$(date -u +%FT%TZ) T422ON_DONE $rc" >> $L' EXIT
B1="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"
# Up to 2 jobs: the second runs only if the first stopped at the warm deadline (cold JIT, cache still filling).
for try in 1 2; do
  for i in $(seq 150); do
    rc=0; out=$(tt-project/t288/blx01-submit.sh $S/on.txt 2>&1) || rc=$?
    # 75: another smarton job holds the queue. A broker hold or unreachable box (chip drop, recovery) also waits.
    echo "$out" >> $L; [ $rc = 75 ] || echo "$out" | grep -qE 'broker holds the device|broker inactive|unreachable' || break
    sleep 120
  done
  [ $rc = 0 ] || exit $rc
  id=$(echo "$out" | sed -n 's/^job=//p'); echo "$(date -u +%FT%TZ) on: try $try job $id submitted" >> $L
  for i in $(seq 120); do
    st=$($B1 "tt-device-mcp status -j $id" 2>&1 | sed -n 's/^Status: *//p' | head -1 | tr A-Z a-z) || true
    case "$st" in running|queued|pending|"") sleep 30 ;; *) break ;; esac
  done
  $B1 "tt-device-mcp status -j $id" >> $L 2>&1 || true
  [ "$st" = completed ] && break
  $B1 "grep -qE '\\[t286\\] (warm deadline|warmup ended)' /var/tmp/fasth3/t422/out_on/run.log && cp /var/tmp/fasth3/t422/out_on/run.log /var/tmp/fasth3/t422/out_on/run.fill$try.log" || break
  echo "$(date -u +%FT%TZ) on: try $try stopped at the warm deadline" >> $L
done
$B1 "tail -20 /var/tmp/fasth3/t422/out_on/run.log; grep -h 'Denoising (per step)\|Total Pipeline\|process wall\|mp4\|T408_EXIT' /var/tmp/fasth3/t422/out_off/run.log /var/tmp/fasth3/t422/out_on/run.log" >> $L 2>&1 || true
[ "$st" = completed ]
