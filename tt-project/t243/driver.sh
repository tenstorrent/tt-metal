#!/bin/bash
# t242 driver on blx01: check out the code commit in the t238 Release build tree (python-only change, same C++),
# then ONE broker job (run.sh: arms def and packed (DIFFVAE_S5_PACKED_LANES=1), each warm-up + 2 timed seeds + deep profile + host-noise seeds 0,1),
# then score: packed vs def bit-identity, both vs the #214 host-noise ref (diffvae/ref).
# Health/drop logic from t235's driver: a drop reruns the job after two clean health passes, a second drop skips it.
# Never resets, never touches other jobs. Marker: $T/drv/driver.marker "T242_DRIVER_DONE stage=.. rc=.. jobs=.."
F=/var/tmp/fasth3; T=$F/t242; L=$T/drv/driver.log; M=$T/drv/driver.marker; W=$T
INC=/var/lib/tt-device-broker/health/incidents; FSM=/var/lib/tt-device-broker/health/fsm.json
PY=$F/t48/python_env/bin/python; CODE=b8c2b3403f8
STAGE=start; JOBS=
trap 'rc=$?; echo "T242_DRIVER_DONE stage=$STAGE rc=$rc jobs=${JOBS# }" > $M' EXIT
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
# job <tag> <script> <outdir>: one broker job; one rerun on a drop. Sets S and JRC.
job() {
  local tag=$1 script=$2 O=$3 need=1 a tmo=0
  for a in 1 2; do
    STAGE=$tag.health$a
    wait_health $need || { log "broker never healthy"; exit 8; }
    STAGE=$tag.job$a
    inc0=$(ls $INC 2> /dev/null | sort | tail -1); t0=$(date -u '+%F %T')
    out=$(timeout 120 tt-device-mcp run-bg "bash $T/drv/$script" -w $W -e $T/drv/env.yaml -t 330 2>&1)
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
    JRC=$(grep -oE 'T242_EXIT=[0-9]+' $O/run.log 2> /dev/null | tail -1 | cut -d= -f2)
    log "$tag job $JOB status=$S T242_EXIT=$JRC new_incidents=[$new] $(echo "$st" | grep -iE '^(Exit|Cause|Runtime)' | tr '\n' ' ' | cut -c1-200)"
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
    log "DROP utc=$t0..$(date -u '+%T') box=g15blx01 job=$JOB status=$S whose=smarton(t242 $tag) incidents=[$new]"
    mv $O ${O}_drop$a 2> /dev/null
    need=2
  done
  log "$tag: two drops (or drop after cold timeout), skipped"; S=skipped; JRC=
}
log "start; disk: $(df -h / | tail -1)"
STAGE=build
git -C $F/t48 fetch -q $T/drv/t242.bundle ttp/t242-diffvae-next-lever-fused-stage-5-blocks-:refs/t242/code || exit 10
git -C $F/t238/b checkout -q --detach $CODE || exit 10
log "b at $(git -C $F/t238/b rev-parse --short=11 HEAD)"
job AB run.sh $T/out
log "AB: status=$S rc=$JRC $(grep -hE 'DECODE|host-noise|rc=' $T/out/run.log 2> /dev/null | tr '\n' ' ')"
[ "$S" = skipped ] && exit 11
STAGE=score
for s in 0 1; do
  a=$(md5sum < $T/out/def/ref_dvx_seed$s.yuv | cut -c1-32); b=$(md5sum < $T/out/packed/ref_dvx_seed$s.yuv | cut -c1-32)
  [ "$a" = "$b" ] && log "score host-noise seed $s: packed identical to def ($a)" || log "score host-noise seed $s: packed DIFFERS from def ($a vs $b)"
done
log "device-noise md5: $(grep -hE 'DECODE seed' $T/out/run.log | sed -E 's/.*\] (\w+) DECODE seed ([0-9]).*md5=(\w+)/\1:\2:\3/' | tr '\n' ' ')"
for arm in def packed; do
  [ -e $T/out/$arm/decode_times.json ] && $PY $T/drv/cmp241.py $F/diffvae/ref $T/out/$arm $T/out/cmp_$arm.json 0,1 >> $L 2>&1
done
[ -e $T/out/packed/decode_times.json ] && $PY $T/drv/cmp241.py $T/out/def $T/out/packed $T/out/cmp_packed_vs_def.json 0,1 >> $L 2>&1
grep -q 'T242_EXIT=0' $T/out/run.log || exit 12
STAGE=done
exit 0
