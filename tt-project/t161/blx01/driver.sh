#!/bin/bash
# t161 driver on blx01 (g15blx01): H3 fl2va timing as <=600 s broker jobs (local hook cap):
# capture (dispatch off, kernel manifest) -> offline kernel_prewarm -> fill (weight cache) -> time 6 s -> time 10 s.
# Health logic copied from /var/tmp/fasth3/t159/driver.sh. A drop reruns that step once after two clean health
# passes; a second drop skips it. Never resets, never touches other jobs.
# Marker: $T/driver.marker "T161_DRIVER_DONE stage=.. rc=.. jobs=.."
F=/var/tmp/fasth3; T=$F/t161; L=$T/driver.log; M=$T/driver.marker; W=$F/t48
INC=/var/lib/tt-device-broker/health/incidents; FSM=/var/lib/tt-device-broker/health/fsm.json
MAN=$F/cache/tt-metal-cache-h3/kernel_prewarm.manifest
STAGE=start; JOBS=
trap 'rc=$?; echo "T161_DRIVER_DONE stage=$STAGE rc=$rc jobs=${JOBS# }" > $M' EXIT
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
# job <tag> <mode> <seconds>: one broker job with one rerun on a drop. Sets S (status) and JRC (script exit).
job() {
  local need=1 a
  for a in 1 2; do
    STAGE=$1.health$a
    wait_health $need || { log "broker never healthy"; exit 8; }
    STAGE=$1.job$a
    inc0=$(ls $INC 2> /dev/null | sort | tail -1); t0=$(date -u '+%F %T')
    out=$(timeout 120 tt-device-mcp run-bg "env T161_MODE=$2 T161_SECONDS=$3 T161_TAG=$1 bash $T/run_t161.sh" -w $W -e $F/t159/env.yaml -t 600 2>&1)
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
    JRC=$(grep -oE 'T161_EXIT=[0-9]+' $T/out_$1/run.log 2> /dev/null | tail -1 | cut -d= -f2)
    log "$1 job $JOB status=$S T161_EXIT=$JRC new_incidents=[$new] $(echo "$st" | grep -iE '^(Exit|Cause)' | tr '\n' ' ' | cut -c1-200)"
    case "$S" in
      broker-kill | power-cycle | reboot | interrupted | abandoned) drop=1 ;;
      *) [ -n "$new" ] && [ "$JRC" != 0 ] && drop=1 || drop=0 ;;
    esac
    [ $drop = 0 ] && return 0
    for f in $new; do
      log "DROP-INCIDENT $f: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("label"), d.get("evidence"), "job=", d.get("job"), "present=", d.get("chips_present_at_capture"))' $INC/$f/incident.json 2>&1 | head -c 600)"
    done
    log "DROP utc=$t0..$(date -u '+%T') box=g15blx01 job=$JOB status=$S whose=smarton(t161 $1) incidents=[$new]"
    mv $T/out_$1 $T/out_$1_drop$a 2> /dev/null
    need=2
  done
  log "$1: two drops in a row, skipped"; S=skipped; JRC=
}
log "start; disk: $(df -h / | tail -1)"
if [ ! -s $MAN ]; then
  # Capture-only traverses on garbage tensors, so its exit code is not checked; the manifest is the artifact.
  job capture capture 6,10
  [ -s $MAN ] || { log "capture wrote no manifest"; exit 12; }
fi
STAGE=prewarm
log "offline compile start ($(wc -l < $MAN) manifest lines)"
(cd $W && env TT_METAL_KERNEL_PREWARM=1 TT_METAL_CACHE=$F/cache/tt-metal-cache-h3 TT_METAL_HOME=$W build_Release/tools/kernel_prewarm) >> $T/prewarm.log 2>&1 \
  || { log "kernel_prewarm rc=$?"; exit 13; }
log "offline compile done; $(du -sh $F/cache/tt-metal-cache-h3 | cut -f1)"
if ! ls $F/cache/dit-h3/* > /dev/null 2>&1 || ! grep -q 'T161_EXIT=0' $T/out_fill/run.log 2> /dev/null; then
  job fill fill 6
  [ "$JRC" = 0 ] || { log "fill failed status=$S rc=$JRC"; exit 14; }
fi
log "dit cache $(du -sh $F/cache/dit-h3 | cut -f1)"
job time6 time 6; R6="$S/$JRC"
job time10 time 10; R10="$S/$JRC"
STAGE=done
log "results time6=$R6 time10=$R10"
[ "${R6#*/}" = 0 ] && [ "${R10#*/}" = 0 ] && exit 0
exit 1
