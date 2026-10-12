#!/bin/bash
# t449 (detached on g15blx02): waits for the blx01 incremental build (build449.rc), then runs the 71-exempt unit
# spec and, if it passes, the 10 s VSA-0.9 arm, one broker job at a time via chain449a.sh. Last line of $T449_LOG:
# "T449C_DONE <rc> <step>".
set -eo pipefail
P=/home/smarton/fasth3/tt-metal; D=$P/tt-project/worktrees/t449/tt-project; L=${T449_LOG:?log}; STEP=build
trap 'rc=$?; echo "$(date -u +%FT%TZ) T449C_DONE $rc $STEP" >> $L' EXIT
B1="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"
for i in $(seq 120); do
  r=$($B1 "cat /var/tmp/fasth3/t409/build449.rc 2>/dev/null" || true)
  [ -n "$r" ] && break; sleep 30
done
echo "$(date -u +%FT%TZ) build449 rc=$r" >> $L
[ "$r" = 0 ]
STEP=unit
T449_OUT=out_u449 bash $D/t409/chain449a.sh $D/t449/u449.txt
STEP=ab10
T449_OUT=out_ab10-vsa0.9 bash $D/t409/chain449a.sh $D/t409/ab10vsa.txt
STEP=done
