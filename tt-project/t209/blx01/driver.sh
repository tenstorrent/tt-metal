#!/bin/bash
# t209 driver on blx01: build fasth3-opt H3 tree, then <=600 s broker jobs: fill (weight cache, load only, retried
# while it makes progress) -> jit (2-step cold run, fills the JIT cache) -> time (2-step warmup + seeds 0,1 at 50 steps).
# Health/drop logic from /var/tmp/fasth3/t161/driver.sh. A drop reruns that step after two clean health passes; a second
# drop skips it. Never resets, never touches other jobs. Marker: $T/driver.marker "T209_DRIVER_DONE stage=.. rc=.. jobs=.."
F=/var/tmp/fasth3; T=$F/t209; L=$T/driver.log; M=$T/driver.marker; W=$T/b
INC=/var/lib/tt-device-broker/health/incidents; FSM=/var/lib/tt-device-broker/health/fsm.json
STAGE=start; JOBS=
trap 'rc=$?; echo "T209_DRIVER_DONE stage=$STAGE rc=$rc jobs=${JOBS# }" > $M' EXIT
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
    out=$(timeout 120 tt-device-mcp run-bg "env T209_TAG=$tag $* bash $T/run209.sh" -w $W -e $T/env.yaml -t 600 2>&1)
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
    JRC=$(grep -oE 'T209_EXIT=[0-9]+' $T/out_$tag/run.log 2> /dev/null | tail -1 | cut -d= -f2)
    log "$tag job $JOB status=$S T209_EXIT=$JRC new_incidents=[$new] $(echo "$st" | grep -iE '^(Exit|Cause|Runtime)' | tr '\n' ' ' | cut -c1-200)"
    case "$S" in
      broker-kill | power-cycle | reboot | interrupted | abandoned) drop=1 ;;
      *) [ -n "$new" ] && [ "$JRC" != 0 ] && drop=1 || drop=0 ;;
    esac
    [ $drop = 0 ] && return 0
    for f in $new; do
      log "DROP-INCIDENT $f: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("label"), d.get("evidence"), "job=", d.get("job"), "present=", d.get("chips_present_at_capture"))' $INC/$f/incident.json 2>&1 | head -c 600)"
    done
    log "DROP utc=$t0..$(date -u '+%T') box=g15blx01 job=$JOB status=$S whose=smarton(t209 $tag) incidents=[$new]"
    mv $T/out_$tag $T/out_${tag}_drop$a 2> /dev/null
    need=2
  done
  log "$tag: two drops in a row, skipped"; S=skipped; JRC=
}
log "start; disk: $(df -h / | tail -1)"
STAGE=build
if ! grep -q BUILD209_DONE $T/build.log 2> /dev/null; then
  log "build start"
  bash $T/setup209.sh > $T/build.log 2>&1 || { log "build rc=$?"; exit 11; }
  log "build done; tree $(du -sh $W | cut -f1)"
fi
for i in 1 2 3; do
  grep -q 'T209_EXIT=0' $T/out_fill/run.log 2> /dev/null && break
  [ -d $T/out_fill ] && mv $T/out_fill $T/out_fill_try$i
  job fill T209_LOAD_ONLY=1
  log "dit cache $(du -sh $F/cache/dit-h3opt | cut -f1)"
done
grep -q 'T209_EXIT=0' $T/out_fill/run.log 2> /dev/null || { log "fill never finished"; exit 14; }
[ -d $T/out_jit ] && grep -q 'T209_EXIT=0' $T/out_jit/run.log || job jit T209_STEPS=2 T209_WARM=0 T209_SEEDS=0
log "jit status=$S rc=$JRC; jit cache $(du -sh $F/cache/tt-metal-cache-h3opt | cut -f1)"
job time T209_STEPS=50 T209_WARM=2 T209_SEEDS=0,1; R="$S/$JRC"
STAGE=done
log "results time=$R"
ls $T/out_time/seed*_timings.json > /dev/null 2>&1 && exit 0
exit 1
