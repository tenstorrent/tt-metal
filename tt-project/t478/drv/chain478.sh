#!/bin/bash
# t478 (run by t481 on g15blx02): wait for the blx01 builds, submit the microbench A/B job through
# blx01-submit.sh (retrying while blx01 is busy or unhealthy), then wait for the job to end.
# Writes the job id to $D/job and the first non-zero step code (or 0) to $D/marker.
set -eo pipefail
D=${1:?state dir}; SPEC=${2:?spec}
R=/home/smarton/fasth3/tt-metal
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"
trap 'echo $? > "$D/marker"' EXIT
step() { echo "[chain] $(date -u '+%F %T') $*"; }
step "wait build"
for i in $(seq 1 60); do
  rc=0; $R/tt-project/harness/bin/ttp detach --check --host g15blx01 /var/tmp/fasth3/t478/detach/build478 || rc=$?
  [ $rc = 0 ] && break
  [ $i = 60 ] && { step "build not done after 60 min"; exit 20; }
  sleep 60
done
brc=$($SSH cat /var/tmp/fasth3/t478/build.rc 2>/dev/null || echo x)
step "build rc=$brc"
[ "$brc" = 0 ] || exit 21
step "submit"
for i in $(seq 1 48); do
  rc=0; out=$(bash $R/tt-project/t288/blx01-submit.sh "$SPEC" 2>&1) || rc=$?
  echo "$out" | tail -5
  id=$(echo "$out" | sed -n 's/^job=//p')
  [ $rc = 0 ] && [ -n "$id" ] && break
  [ $i = 48 ] && { step "submit failed 48 times, last rc=$rc"; exit 22; }
  sleep 300
done
echo "$id" > "$D/job"
step "job $id submitted"
while :; do
  st=$($SSH "tt-device-mcp status -j $id" 2>/dev/null | sed -n 's/^Status: *//p' | head -1) || st=
  case "$st" in running*|queued*|pending*|"") sleep 60 ;; *) break ;; esac
done
step "job $id ended: $st"
