#!/bin/bash
# t133 (#135) driver on blx01: two broker jobs in order, j1 (A B) then j2 (B A), each after the broker health check,
# one at a time. j1 -t 600 (unmeasured: it also records the halo-off reference); j2 -t = j1's measured time +50%,
# within 600. On a drop: wait for two clean health passes and rerun that job once; a second drop on it stops.
# ADOPT_J1=<broker job id> follows an already submitted j1 instead of submitting it.
# Never resets, never touches other jobs. Final marker: $T/driver.marker "T133_DRIVER_DONE stage=.. rc=.. jobs=.."
# rc: 0 ok, 1 job failed (not a drop), 7 submit failed, 8 broker never healthy, 9 two drops on one job.
F=/var/tmp/fasth3; T=$F/t133; L=$T/driver.log; M=$T/driver.marker
INC=/var/lib/tt-device-broker/health/incidents; FSM=/var/lib/tt-device-broker/health/fsm.json
STAGE=start; JOBS=
trap 'rc=$?; echo "T133_DRIVER_DONE stage=$STAGE rc=$rc jobs=${JOBS# }" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
fsm() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' $FSM 2> /dev/null; }
health() {
  [ "$(systemctl is-active tt-device-broker 2> /dev/null)" = active ] || { log "health: broker inactive"; return 1; }
  pgrep -f /opt/tt-device-broker/autoupdate.sh > /dev/null && { log "health: broker upgrade running"; return 1; }
  [ "$(fsm)" = healthy ] || { log "health: fsm=$(fsm)"; return 1; }
  st=$(timeout 60 tt-device-mcp status 1 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | sed -n '/^RUNNING/,/^RECENT/p' | grep -q smarton && { log "health: a smarton job is running/queued"; return 1; }
  last=$(ls $INC 2> /dev/null | sort | tail -1)
  if [ -n "$last" ]; then
    lt=$(date -u -d "$(echo $last | sed -E 's/^(....)(..)(..)T(..)(..)(..)Z.*/\1-\2-\3 \4:\5:\6/')" +%s 2> /dev/null || echo 0)
    [ $(($(date -u +%s) - lt)) -ge 600 ] || { log "health: incident $last < 10 min old"; return 1; }
  fi
  return 0
}
wait_health() {  # $1 = passes needed in a row
  local ok=0
  for i in $(seq 240); do
    if health; then ok=$((ok + 1)); [ $ok -ge $1 ] && return 0; else ok=0; fi
    sleep 60
  done
  return 1
}
TLIM=600
for spec in "j1 A B" "j2 B A"; do
  set -- $spec; J=$1
  need=1
  for a in 1 2; do
    STAGE=health_${J}_$a
    [ $J = j1 ] && [ $a = 1 ] && [ -n "$ADOPT_J1" ] || wait_health $need || { log "broker never healthy"; exit 8; }
    STAGE=${J}_$a
    inc0=$(ls $INC 2> /dev/null | sort | tail -1)
    t0=$(date -u '+%F %T')
    if [ $J = j1 ] && [ $a = 1 ] && [ -n "$ADOPT_J1" ]; then
      out="Job $ADOPT_J1 queued (adopted: already submitted by the first driver start)"
    else
      out=$(timeout 120 tt-device-mcp run-bg "bash $T/run.sh $spec" -w $F/t48 -e $F/t159/env.yaml -t $TLIM 2>&1)
    fi
    JOB=$(echo "$out" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p' | head -1)
    log "$J attempt $a submit -t $TLIM: $(echo "$out" | tr '\n' ' ' | cut -c1-300) JOB=$JOB"
    [ -n "$JOB" ] || exit 7
    JOBS="$JOBS $J:$JOB"
    while :; do
      st=$(timeout 60 tt-device-mcp status -j $JOB 2>&1)
      s=$(echo "$st" | sed -n 's/^Status: *//p' | head -1)
      case "$s" in queued | running | pending | "") sleep 30 ;; *) break ;; esac
    done
    log "$J attempt $a job $JOB status=$s $(echo "$st" | grep -iE '^(Exit|Cause|Runtime)' | tr '\n' ' ' | cut -c1-200)"
    sleep 120  # the broker's post-job gate
    new=$(ls $INC 2> /dev/null | sort | awk -v z="$inc0" '$0 > z' | paste -sd' ' -)
    jrc=$(grep -oE 'T133_EXIT=[0-9]+' $T/run133_$J.log 2> /dev/null | tail -1 | cut -d= -f2)
    case "$s" in
      broker-kill | power-cycle | reboot | interrupted | abandoned) drop=1 ;;
      completed) [ "$jrc" = 0 ] && drop=0 || { [ -n "$new" ] && drop=1 || drop=0; } ;;
      *) [ -n "$new" ] && drop=1 || drop=0 ;;
    esac
    if [ $drop = 0 ]; then
      [ "$s" = completed ] && [ "$jrc" = 0 ] || { log "$J FAILED (not a drop) status=$s T133_EXIT=$jrc"; exit 1; }
      log "$J OK (job $JOB) new_incidents=[$new]"
      if [ $J = j1 ]; then
        s0=$(sed -n 's/.*start .* epoch=\([0-9]*\).*/\1/p' $T/run133_j1.log); s1=$(sed -n 's/.*end_epoch=\([0-9]*\).*/\1/p' $T/run133_j1.log)
        TLIM=$(( (s1 - s0) * 3 / 2 )); [ $TLIM -gt 600 ] && TLIM=600; [ $TLIM -lt 240 ] && TLIM=240
        log "j1 took $((s1 - s0)) s; j2 -t $TLIM"
      fi
      break
    fi
    for f in $new; do
      log "DROP-INCIDENT $f: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("label"), d.get("evidence"), "job=", d.get("job"), "present=", d.get("chips_present_at_capture"))' $INC/$f/incident.json 2>&1 | head -c 600)"
    done
    log "DROP utc=$t0..$(date -u '+%T') box=g15blx01 job=$JOB status=$s whose=smarton(t133 $J) incidents=[$new]"
    mkdir -p $T/drop_${J}_$a; mv $T/out_?$J $T/run133_$J.log $T/drop_${J}_$a/ 2> /dev/null
    need=2
    [ $a = 2 ] && { log "two drops in a row on $J: stop"; exit 9; }
  done
done
STAGE=cmp
source $F/t48/python_env/bin/activate
python $T/cmp133.py $T > $T/cmp.txt 2>&1; log "cmp rc=$? $(tr '\n' ' ' < $T/cmp.txt | cut -c1-600)"
exit 0
