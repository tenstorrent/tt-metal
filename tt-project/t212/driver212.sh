#!/bin/bash
# t212 driver on blx01: wait for the t212 build, make treeA, then two <=600 s broker jobs, one at a time:
# A (unported t48 DiffVAE decode) then B (the t212 port), then the CPU PCC/PSNR compare.
# Health/drop logic copied from /var/tmp/fasth3/t209/driver.sh: a drop reruns that arm after two clean health passes,
# a second drop skips it. A timeout without a drop (cold JIT cache) reruns once. Never resets, never touches other jobs.
# Marker: $T/driver.marker "T212_DRIVER_DONE stage=.. rc=.. jobs=.."
F=/var/tmp/fasth3; T=$F/t212; L=$T/driver.log; M=$T/driver.marker; W=$T
INC=/var/lib/tt-device-broker/health/incidents; FSM=/var/lib/tt-device-broker/health/fsm.json
STAGE=start; JOBS=
trap 'rc=$?; echo "T212_DRIVER_DONE stage=$STAGE rc=$rc jobs=${JOBS# }" > $M' EXIT
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
# job <tag> <env...>: one broker job with one rerun on a drop. Sets S (status) and JRC (script exit).
job() {
  local tag=$1 need=1 a; shift
  for a in 1 2; do
    STAGE=$tag.health$a
    wait_health $need || { log "broker never healthy"; exit 8; }
    STAGE=$tag.job$a
    inc0=$(ls $INC 2> /dev/null | sort | tail -1); t0=$(date -u '+%F %T')
    out=$(timeout 120 tt-device-mcp run-bg "env T212_TAG=$tag $* bash $T/run212.sh" -w $W -e $T/env.yaml -t 600 2>&1)
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
    JRC=$(grep -oE 'T212_EXIT=[0-9]+' $T/out_$tag/run.log 2> /dev/null | tail -1 | cut -d= -f2)
    log "$tag job $JOB status=$S T212_EXIT=$JRC new_incidents=[$new] $(echo "$st" | grep -iE '^(Exit|Cause|Runtime)' | tr '\n' ' ' | cut -c1-200)"
    case "$S" in
      broker-kill | power-cycle | reboot | interrupted | abandoned) drop=1 ;;
      *) [ -n "$new" ] && [ "$JRC" != 0 ] && drop=1 || drop=0 ;;
    esac
    [ $drop = 0 ] && return 0
    for f in $new; do
      log "DROP-INCIDENT $f: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("label"), d.get("evidence"), "job=", d.get("job"), "present=", d.get("chips_present_at_capture"))' $INC/$f/incident.json 2>&1 | head -c 600)"
    done
    log "DROP utc=$t0..$(date -u '+%T') box=g15blx01 job=$JOB status=$S whose=smarton(t212 $tag) incidents=[$new]"
    mv $T/out_$tag $T/out_${tag}_drop$a 2> /dev/null
    need=2
  done
  log "$tag: two drops in a row, skipped"; S=skipped; JRC=
}
log "start; disk: $(df -h / | tail -1)"
STAGE=build
while [ ! -f $T/build.rc ]; do sleep 60; done
[ "$(cat $T/build.rc)" = 0 ] && grep -q BUILD212_DONE $T/build.log || { log "build failed rc=$(cat $T/build.rc)"; exit 11; }
log "build done; tree $(du -sh $T/b | cut -f1)"
STAGE=treeA
bash $T/mktreeA.sh >> $L 2>&1 || exit 12
for tag in A B; do
  for i in 1 2; do
    [ -f $T/out_$tag/px.pt ] && grep -q 'T212_EXIT=0' $T/out_$tag/run.log 2> /dev/null && break
    [ -d $T/out_$tag ] && mv $T/out_$tag $T/out_${tag}_try$i
    job $tag
    log "$tag: status=$S rc=$JRC $(grep -hE '^\[decode' $T/out_$tag/run.log 2> /dev/null | tr '\n' ' ')"
    [ "$S" = skipped ] && break
  done
done
STAGE=cmp
[ -f $T/out_A/px.pt ] && [ -f $T/out_B/px.pt ] || { log "missing a pixel dump"; exit 15; }
$F/t48/python_env/bin/python $T/cmp212.py $T/out_A/px.pt $T/out_B/px.pt $T/cmp.json >> $L 2>&1 || exit 16
log "cmp $(cat $T/cmp.json)"
STAGE=done
exit 0
