#!/bin/bash
# t286 one-shot submit on blx01: health gate, then exactly one broker job. No waiting, no loop.
# Args: <tag> <duration>. Prints "SUBMITTED <tag> JOB=<id>" or "NOT_READY <reason>".
T=/var/tmp/fasth3/t286; W=/var/tmp/fasth3/t284/b
FSM=/var/lib/tt-device-broker/health/fsm.json; INC=/var/lib/tt-device-broker/health/incidents
[ "$(systemctl is-active tt-device-broker)" = active ] || { echo "NOT_READY broker inactive"; exit 1; }
ps -eo cmd | grep -q '[/]opt/tt-device-broker/autoupdate.sh' && { echo "NOT_READY broker upgrade"; exit 1; }
s=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' $FSM 2> /dev/null)
[ "$s" = healthy ] || { echo "NOT_READY fsm=$s"; exit 1; }
st=$(timeout 60 tt-device-mcp status 1 2>&1) || { echo "NOT_READY status failed"; exit 1; }
echo "$st" | sed -n '/^RUNNING/,/^RECENT/p' | grep -q smarton && { echo "NOT_READY smarton job running/queued"; exit 1; }
last=$(ls $INC 2> /dev/null | sort | tail -1)
if [ -n "$last" ]; then
  lt=$(date -u -d "$(echo $last | sed -E 's/^(....)(..)(..)T(..)(..)(..)Z.*/\1-\2-\3 \4:\5:\6/')" +%s 2> /dev/null || echo 0)
  [ $(($(date -u +%s) - lt)) -ge 600 ] || { echo "NOT_READY incident $last < 10 min old"; exit 1; }
fi
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 70 ] || { echo "NOT_READY root fs ${use}%"; exit 1; }
[ -e $T/out_$1 ] && mv $T/out_$1 $T/out_$1_prev$(date +%H%M%S)
out=$(timeout 120 tt-device-mcp run-bg "bash $T/run286.sh $1 $2" -w $W -e $T/env286.yaml -t 600 2>&1)
JOB=$(echo "$out" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p' | head -1)
echo "$(date -u '+%F %T') submit $1: $(echo "$out" | tr '\n' ' ' | cut -c1-200)" >> $T/sub286.log
[ -n "$JOB" ] && echo "SUBMITTED $1 JOB=$JOB" || { echo "SUBMIT_FAILED $out"; exit 2; }
