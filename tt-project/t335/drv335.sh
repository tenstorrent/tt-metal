#!/bin/bash
# t335 driver on blx01: one broker job per A/B pair, in order (default+HiFi2, then default+LoFi), each submitted only
# after the broker health check passes. A job that ends without its T335_EXIT line counts as a drop: wait for
# recovery and rerun it once; a second drop skips that pair. Then cmp335.py on the host.
# Marker: t335/drv335.done (first line = outcome). Log: t335/drv335.log. Never resets, never touches other jobs.
set -o pipefail
F=/var/tmp/fasth3; D=$F/t335; M=$D/drv335.done; L=$D/drv335.log
INC=/var/lib/tt-device-broker/health/incidents; FSM=/var/lib/tt-device-broker/health/fsm.json
PAIRS=${PAIRS:-"hifi2:default,HiFi2 lofi:default,LoFi"}
RES=""
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc$RES" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
fsm() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' $FSM 2> /dev/null; }
health() {
  [ "$(systemctl is-active tt-device-broker 2> /dev/null)" = active ] || { log "health: broker inactive"; return 1; }
  ps -eo args= | grep -q '^[^ ]*bash /opt/tt-device-broker/autoupdate.sh' && { log "health: broker upgrade running"; return 1; }
  [ "$(fsm)" = healthy ] || { log "health: fsm=$(fsm)"; return 1; }
  st=$(timeout 60 tt-device-mcp status 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | sed -n '/^RUNNING/,/^RECENT/p' | grep -qi upgrade && { log "health: upgrade"; return 1; }
  echo "$st" | sed -n '/^RUNNING/,/^RECENT/p' | grep -qiE 'smarton|hold|health|reset|fabric-check' && { log "health: busy"; return 1; }
  last=$(ls $INC 2> /dev/null | sort | tail -1)
  if [ -n "$last" ]; then
    lt=$(date -u -d "$(echo $last | sed -E 's/^(....)(..)(..)T(..)(..)(..)Z.*/\1-\2-\3 \4:\5:\6/')" +%s 2> /dev/null || echo 0)
    [ $(($(date -u +%s) - lt)) -ge 600 ] || { log "health: incident $last < 10 min old"; return 1; }
  fi
}
wait_health() { local ok=0; for i in $(seq 360); do health && ok=$((ok + 1)) || ok=0; [ $ok -ge $1 ] && return 0; sleep 30; done; return 1; }
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
# Up to 6 h: a job queued behind other tenants is never treated as a drop (no duplicate submit).
waitjob() { for i in $(seq 1440); do s=$(st $1); case $s in running|queued|pending|"") sleep 15;; *) echo $s; return;; esac; done; echo stillrunning; }
log "driver start; $(df -h / | tail -1)"
need=1
for P in $PAIRS; do
  TAG=${P%%:*}; ARMS=${P#*:}
  for att in 1 2; do
    [ $att = 1 ] && echo " $ADOPT " | grep -q " $TAG=" || wait_health $need || { log "broker never healthy"; RES="$RES $TAG=nohealth"; break 2; }
    # ADOPT="<tag>=<job>": a restarted driver waits on that already-submitted job instead of submitting again.
    J=$(echo " $ADOPT " | grep -oE " $TAG=[0-9]+ " | cut -d= -f2 | tr -d ' ')
    if [ $att = 1 ] && [ -n "$J" ]; then log "adopting job $J for $TAG"; else
    rm -f $D/out_$TAG/run.log
    sub=$(tt-device-mcp run-bg "env T335_ATTEMPT=$att bash $D/run335.sh $TAG ${ARMS//,/ }" -w $D -e $D/env.yaml -t 600 2>&1)
    echo "$sub" >> $L; J=$(echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'); fi
    [ -n "$J" ] || { RES="$RES $TAG=notsubmitted"; break; }
    log "pair=$TAG attempt=$att job=$J"
    s=$(waitjob $J); ex=$(grep -E '^T335_EXIT=' $D/out_$TAG/run.log 2>/dev/null | tail -1)
    cp $D/out_$TAG/run.log $D/run_${TAG}_job$J.log 2>/dev/null
    sleep 10; log "pair=$TAG job=$J status=$s $ex leftover: $(ps -u $(id -u) -o pid=,args= | grep -E 'dec335|run335' | grep -v grep | tr '\n' ';')"
    RES="$RES $TAG=job$J:$s:${ex:-noexit}"
    [ "$s" = stillrunning ] && { log "job $J still not finished after 6 h; stop"; break 2; }
    [ "$s" != completed ] && [ -z "$ex" ] || break
    log "pair=$TAG job=$J DROP? status=$s, no exit line; waiting for recovery, then rerun"; need=2
  done
done
source $F/t48/python_env/bin/activate
for P in $PAIRS; do
  TAG=${P%%:*}; ARMS=${P#*:}
  [ -d $D/out_$TAG/default ] && timeout 900 python $D/cmp335.py $D/out_$TAG default ${ARMS#default,} >> $D/cmp.txt 2>&1
done
log "cmp: $(grep -c CMP $D/cmp.txt 2>/dev/null) lines"
echo "DONE$RES" > $M
