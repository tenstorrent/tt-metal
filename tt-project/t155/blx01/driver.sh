#!/bin/bash
# t155 (#155) driver on blx01: CPU setup + fresh build of the t152 fix (setup155.sh), then two broker jobs in order,
# j1 = 64,128,5,4,4 then j2 = 64,128,7,4,4 (only if j1 passed), each after the broker health check and with no other
# smarton job running/queued (so it never overlaps #127). j1 -t 60 (job 777 took 34 s); j2 -t = j1 measured +50% (60..600).
# On a drop: wait for two clean health passes and rerun that job once; a second drop on it skips that config.
# Never resets, never touches other jobs. At the end removes B (worktree + build dir) and the JIT cache, keeps logs.
# Final marker: $T/driver.marker "T155_DRIVER_DONE stage=.. rc=.. jobs=.. results=.."
# rc: 0 both ran (see results), 1 j1 failed (not a drop), 6 setup/build failed, 7 submit failed, 8 broker never healthy.
F=/var/tmp/fasth3; T=$F/t155; L=$T/driver.log; M=$T/driver.marker; A=$F/t48; B=$T/b
INC=/var/lib/tt-device-broker/health/incidents; FSM=/var/lib/tt-device-broker/health/fsm.json
STAGE=start; JOBS=; RES=
cleanup() {
  [ -n "$KEEP" ] && return
  git -C $A worktree remove --force $B >> $L 2>&1 || rm -rf $B
  git -C $A worktree prune; rm -rf $T/jit
  log "cleanup: B and jit removed; t48 HEAD=$(git -C $A rev-parse --short=11 HEAD)"
}
trap 'rc=$?; cleanup; echo "T155_DRIVER_DONE stage=$STAGE rc=$rc jobs=${JOBS# } results=${RES# }" > $M' EXIT
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
  for i in $(seq 480); do
    if health; then ok=$((ok + 1)); [ $ok -ge $1 ] && return 0; else ok=0; fi
    sleep 60
  done
  return 1
}
STAGE=setup
log "start; t48 HEAD=$(git -C $A rev-parse --short=11 HEAD)"
bash $T/setup155.sh > $T/setup.log 2>&1 || { log "setup/build failed: $(tail -3 $T/setup.log | tr '\n' ' ')"; exit 6; }
log "$(grep BUILD155_DONE $T/setup.log)"
TLIM=60
for spec in "j1 64,128,5,4,4" "j2 64,128,7,4,4"; do
  set -- $spec; J=$1; BLK=$2
  if [ $J = j2 ] && ! grep -q '^T155_PASS' $T/j1.log 2> /dev/null; then log "j2 not run: j1 did not pass"; RES="$RES j2:not_run"; break; fi
  need=1
  for a in 1 2; do
    STAGE=health_${J}_$a
    wait_health $need || { log "broker never healthy"; exit 8; }
    STAGE=${J}_$a
    inc0=$(ls $INC 2> /dev/null | sort | tail -1)
    t0=$(date -u '+%F %T')
    out=$(timeout 120 tt-device-mcp run-bg "bash $T/run155.sh $J $BLK $((TLIM - 8))" -w $F/t48 -e $F/t159/env.yaml -t $TLIM 2>&1)
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
    jrc=$(grep -oE 'T155_EXIT=[0-9]+' $T/$J.log 2> /dev/null | tail -1 | cut -d= -f2)
    case "$s" in
      broker-kill | power-cycle | reboot | interrupted | abandoned) drop=1 ;;
      completed) [ "$jrc" = 0 ] && drop=0 || { [ -n "$new" ] && drop=1 || drop=0; } ;;
      *) [ -n "$new" ] && drop=1 || drop=0 ;;
    esac
    # A hang (pytest timeout) is a result, not a drop, even if the broker logs an incident for it.
    grep -qE 'Timeout|timed out' $T/$J.log 2> /dev/null && [ -z "$(grep -E 'hugepage|Failed to pin' $T/$J.log)" ] && drop=0
    if [ $drop = 0 ]; then
      r=$(grep -E '^T155_(PASS|FAIL)' $T/$J.log | tail -1)
      log "$J RESULT (job $JOB) status=$s T155_EXIT=$jrc $r new_incidents=[$new]"
      RES="$RES $J:$JOB:${r:-none}"
      for f in $new; do log "POST-JOB-INCIDENT $f"; done
      if [ $J = j1 ]; then
        s0=$(sed -n 's/.* epoch=\([0-9]*\)$/\1/p' $T/j1.log | head -1); s1=$(sed -n 's/.*end_epoch=\([0-9]*\).*/\1/p' $T/j1.log)
        TLIM=$(( (s1 - s0) * 3 / 2 )); [ $TLIM -gt 600 ] && TLIM=600; [ $TLIM -lt 60 ] && TLIM=60
        log "j1 took $((s1 - s0)) s; j2 -t $TLIM"
      fi
      break
    fi
    for f in $new; do
      log "DROP-INCIDENT $f: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("label"), d.get("evidence"), "job=", d.get("job"), "present=", d.get("chips_present_at_capture"))' $INC/$f/incident.json 2>&1 | head -c 600)"
    done
    log "DROP utc=$t0..$(date -u '+%T') box=g15blx01 job=$JOB status=$s whose=smarton(t155 $J) incidents=[$new]"
    mkdir -p $T/drop_${J}_$a; mv $T/$J.log $T/results_$J $T/drop_${J}_$a/ 2> /dev/null
    need=2
    [ $a = 2 ] && { log "two drops in a row on $J ($BLK): skipped"; RES="$RES $J:skipped_2drops"; }
  done
done
STAGE=end
exit 0
