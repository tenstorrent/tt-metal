#!/bin/bash
# t262 driver on blx01: ONE broker job: run262.sh SRC "ARMS" OUT "PROFILE_ARMS" (one python process per arm),
# then score each arm vs the #214 host-noise refs (diffvae/ref) and vs the first arm.
# Usage: driver262.sh TAG SRC "ARMS" "PROFILE_ARMS" TIMEOUT
# Health/drop logic from t244's driver: a drop reruns the job after two clean health passes, a second drop skips it.
# Never resets, never touches other jobs. Marker: $T/drv/driver.marker "T262_DRIVER_DONE stage=.. rc=.. jobs=.."
F=/var/tmp/fasth3; T=$F/t262; L=$T/drv/driver.log; M=$T/drv/driver.marker; W=$T
INC=/var/lib/tt-device-broker/health/incidents; FSM=/var/lib/tt-device-broker/health/fsm.json
PY=$F/t48/python_env/bin/python; CODE=34a571c5f47
TAG=$1; SRC=$2; ARMS=$3; PARMS=$4; TMO=$5; M=$T/drv/driver_$TAG.marker
STAGE=start; JOBS=
trap 'rc=$?; echo "T262_DRIVER_DONE stage=$STAGE rc=$rc jobs=${JOBS# }" > $M' EXIT
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
  for i in $(seq 360); do
    if health; then ok=$((ok + 1)); [ $ok -ge $1 ] && return 0; else ok=0; fi
    sleep 60
  done
  return 1
}
# job <tag> <arms> <outdir> <timeout>: one broker job; one rerun on a drop. Sets S and JRC.
job() {
  local tag=$1 arms=$2 O=$3 tmo=$4 need=1 a
  for a in 1 2; do
    STAGE=$tag.health$a
    wait_health $need || { log "broker never healthy"; exit 8; }
    STAGE=$tag.job$a
    inc0=$(ls $INC 2> /dev/null | sort | tail -1); t0=$(date -u '+%F %T')
    out=$(timeout 120 tt-device-mcp run-bg "bash $T/drv/run262.sh $SRC '$arms' $O '$PARMS'" -w $W -e $T/drv/env.yaml -t $tmo 2>&1)
    JOB=$(echo "$out" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p' | head -1)
    log "$tag attempt $a submit: $(echo "$out" | tr '\n' ' ' | cut -c1-200) JOB=$JOB"
    [ -n "$JOB" ] || exit 7
    JOBS="$JOBS $tag:$JOB"
    while :; do
      st=$(timeout 60 tt-device-mcp status -j $JOB 2>&1)
      S=$(echo "$st" | sed -n 's/^Status: *//p' | head -1)
      case "$S" in queued | running | pending | "") sleep 30 ;; *) break ;; esac
    done
    sleep 120  # the broker's post-job gate
    new=$(ls $INC 2> /dev/null | sort | awk -v z="$inc0" '$0 > z' | paste -sd' ' -)
    JRC=$(grep -oE 'T262_EXIT=[0-9]+' $O/run.log 2> /dev/null | tail -1 | cut -d= -f2)
    log "$tag job $JOB status=$S T262_EXIT=$JRC new_incidents=[$new] $(echo "$st" | grep -iE '^(Exit|Cause|Runtime)' | tr '\n' ' ' | cut -c1-200)"
    case "$S" in
      broker-kill | power-cycle | reboot | interrupted | abandoned) drop=1 ;;
      *) [ -n "$new" ] && [ "$JRC" != 0 ] && drop=1 || drop=0 ;;
    esac
    if [ $drop = 1 ] && grep -q '^Traceback' $O/run.log 2> /dev/null; then
      log "$tag: job $JOB failed with a traceback; incidents [$new] came after it, not counted as a drop"; drop=0
    fi
    if [ $drop = 0 ]; then
      return 0
    fi
    for f in $new; do
      log "DROP-INCIDENT $f: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("label"), d.get("evidence"), "job=", d.get("job"), "present=", d.get("chips_present_at_capture"))' $INC/$f/incident.json 2>&1 | head -c 600)"
    done
    log "DROP utc=$t0..$(date -u '+%T') box=g15blx01 job=$JOB status=$S whose=smarton(t262 $tag) incidents=[$new]"
    mv $O ${O}_drop$a 2> /dev/null
    need=2
  done
  log "$tag: two drops (or drop after cold timeout), skipped"; S=skipped; JRC=
}
log "start $TAG src=$SRC arms=[$ARMS] profile=[$PARMS] -t $TMO; disk: $(df -h / | tail -1)"
STAGE=build
[ "$(git -C $F/t238/b rev-parse --short=11 HEAD)" = $CODE ] || { log "t238/b is not at $CODE; refusing"; exit 10; }
O=$T/out_$TAG
job $TAG "$ARMS" $O $TMO
log "$TAG: status=$S rc=$JRC $(grep -hE 'DECODE_MEAN|process wall' $O/run.log 2> /dev/null | tr '\n' ' ')"
[ "$S" = skipped ] && exit 11
STAGE=score
first=
for a in $ARMS; do
  arm=${a%%:*}
  [ -e $O/$arm/decode_times.json ] && $PY $T/drv/cmp241.py $F/diffvae/ref $O/$arm $O/cmp_$arm.json 0,1 >> $L 2>&1
  if [ -z "$first" ]; then first=$arm; continue; fi
  for s in 0 1; do
    x=$(md5sum < $O/$first/ref_dvx_seed$s.yuv | cut -c1-32); y=$(md5sum < $O/$arm/ref_dvx_seed$s.yuv | cut -c1-32)
    log "MD5 host-noise seed $s: $arm vs $first: $x $y"
  done
  [ -e $O/$arm/decode_times.json ] && $PY $T/drv/cmp241.py $O/$first $O/$arm $O/cmp_${arm}_vs_$first.json 0,1 >> $L 2>&1
done
grep -q 'T262_EXIT=0' $O/run.log || exit 12
STAGE=done
exit 0
