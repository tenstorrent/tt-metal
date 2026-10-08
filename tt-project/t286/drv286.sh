#!/bin/bash
# t286 driver on blx01 (g15blx01): FastH3 Turbo 4-step fl2va baseline as <=600 s broker jobs:
# fill (cold weight + kernel cache, 5 s clip; retried while it times out) -> time 5 s -> time 10 s.
# Health and drop logic from /var/tmp/fasth3/t161/driver.sh: a drop reruns that step after two clean
# health passes; a second drop skips it. Never resets, never touches other jobs.
# Marker: $T/drv286.marker "T286_DRIVER_DONE stage=.. rc=.. jobs=.."
F=/var/tmp/fasth3; T=$F/t286; L=$T/drv286.log; M=$T/drv286.marker; W=$F/t284/b
INC=/var/lib/tt-device-broker/health/incidents; FSM=/var/lib/tt-device-broker/health/fsm.json
STAGE=start; JOBS=
trap 'rc=$?; echo "T286_DRIVER_DONE stage=$STAGE rc=$rc jobs=${JOBS# }" > $M' EXIT
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
wait_health() {
  local ok=0
  for i in $(seq 240); do
    if health; then ok=$((ok + 1)); [ $ok -ge $1 ] && return 0; else ok=0; fi
    sleep 60
  done
  return 1
}
# job <tag> <duration>: one broker job with one rerun on a drop. Sets S (status) and JRC (script exit).
job() {
  local need=1 a
  for a in 1 2; do
    STAGE=$1.health$a
    wait_health $need || { log "broker never healthy"; exit 8; }
    STAGE=$1.job$a
    inc0=$(ls $INC 2> /dev/null | sort | tail -1); t0=$(date -u '+%F %T')
    out=$(timeout 120 tt-device-mcp run-bg "bash $T/run286.sh $1 $2" -w $W -e $T/env286.yaml -t 600 2>&1)
    JOB=$(echo "$out" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p' | head -1)
    log "$1 attempt $a submit: $(echo "$out" | tr '\n' ' ' | cut -c1-200) JOB=$JOB"
    [ -n "$JOB" ] || exit 7
    JOBS="$JOBS $1:$JOB"
    while :; do
      st=$(timeout 60 tt-device-mcp status -j $JOB 2>&1)
      S=$(echo "$st" | sed -n 's/^Status: *//p' | head -1)
      case "$S" in queued | running | pending | "") sleep 30 ;; *) break ;; esac
    done
    sleep 120  # the broker's post-job gate
    new=$(ls $INC 2> /dev/null | sort | awk -v z="$inc0" '$0 > z' | paste -sd' ' -)
    JRC=$(grep -oE 'T286_EXIT=[0-9]+' $T/out_$1/run.log 2> /dev/null | tail -1 | cut -d= -f2)
    log "$1 job $JOB status=$S T286_EXIT=$JRC new_incidents=[$new] $(echo "$st" | grep -iE '^(Exit|Cause|Runtime)' | tr '\n' ' ' | cut -c1-200)"
    case "$S" in
      broker-kill | power-cycle | reboot | interrupted | abandoned) drop=1 ;;
      *) [ -n "$new" ] && [ "$JRC" != 0 ] && drop=1 || drop=0 ;;
    esac
    [ $drop = 0 ] && return 0
    for f in $new; do
      log "DROP-INCIDENT $f: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("label"), d.get("evidence"), "job=", d.get("job"), "present=", d.get("chips_present_at_capture"))' $INC/$f/incident.json 2>&1 | head -c 600)"
    done
    log "DROP utc=$t0..$(date -u '+%T') box=g15blx01 job=$JOB status=$S whose=smarton(t286 $1) incidents=[$new]"
    mv $T/out_$1 $T/out_$1_drop$a 2> /dev/null
    need=2
  done
  log "$1: two drops in a row, skipped"; S=skipped; JRC=
}
log "start; disk: $(df -h / | tail -1)"
for n in 1 2 3; do
  [ -f $T/fill.ok ] && break
  job fill$n 5
  [ "$JRC" = 0 ] && { touch $T/fill.ok; break; }
  log "fill$n not done (status=$S rc=$JRC); dit cache $(du -sh $F/cache/dit-h3hf | cut -f1), kernel cache $(du -sh $F/cache/tt-metal-cache-h3hf | cut -f1)"
done
[ -f $T/fill.ok ] || { STAGE=fill; log "fill never completed"; exit 14; }
job t10 10
log "t10 status=$S rc=$JRC"
job t5 5
log "t5 status=$S rc=$JRC"
STAGE=done
log "done; disk: $(df -h / | tail -1)"
