#!/bin/bash
# t225 driver on blx01: one broker job (run.sh: 1d then 2d production decodes, seeds 0-4, yuv saved), then CPU post
# (post.py cmp + stills, mp4s). Health/drop logic from t222's driverE.sh: a drop reruns after two clean health passes,
# a second drop in a row skips (rc 11). A non-drop failure that still wrote new seeds reruns (the job resumes per seed).
# At most 3 attempts. Never resets, never touches other jobs.
# Marker: $T/drv/driver.marker "T225_DRIVER_DONE stage=.. rc=.. jobs=.."
F=/var/tmp/fasth3; T=$F/t225; O=$T/out; L=$T/drv/driver.log; M=$T/drv/driver.marker
INC=/var/lib/tt-device-broker/health/incidents; FSM=/var/lib/tt-device-broker/health/fsm.json
STAGE=start; JOBS=
trap 'rc=$?; echo "T225_DRIVER_DONE stage=$STAGE rc=$rc jobs=${JOBS# }" > $M' EXIT
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
mkdir -p $O
log "start; disk: $(df -h /var/tmp | tail -1)"
need=1; drops=0; ok=0
for a in 1 2 3; do
  STAGE=health$a
  wait_health $need || { log "broker never healthy"; exit 8; }
  STAGE=job$a
  [ -f $O/run.log ] && mv $O/run.log $O/run.log.a$((a - 1))
  inc0=$(ls $INC 2> /dev/null | sort | tail -1); t0=$(date -u '+%F %T'); n0=$(ls $O/*.yuv 2> /dev/null | wc -l)
  out=$(timeout 120 tt-device-mcp run-bg "env T225_ATTEMPT=$a bash $T/drv/run.sh" -w $T -e $T/drv/env.yaml -t ${TMO:-300} 2>&1)
  JOB=$(echo "$out" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p' | head -1)
  log "attempt $a submit -t ${TMO:-300}: $(echo "$out" | tr '\n' ' ' | cut -c1-200) JOB=$JOB"
  [ -n "$JOB" ] || exit 7
  JOBS="$JOBS $JOB"
  while :; do
    st=$(timeout 60 tt-device-mcp status -j $JOB 2>&1)
    S=$(echo "$st" | sed -n 's/^Status: *//p' | head -1)
    case "$S" in queued | running | pending | "") sleep 30 ;; *) break ;; esac
  done
  sleep 120  # the broker's post-job gate
  new=$(ls $INC 2> /dev/null | sort | awk -v z="$inc0" '$0 > z' | paste -sd' ' -)
  JRC=$(grep -oE 'T225_EXIT=[0-9]+' $O/run.log 2> /dev/null | tail -1 | cut -d= -f2)
  log "job $JOB status=$S T225_EXIT=$JRC new_incidents=[$new] $(echo "$st" | grep -iE '^(Exit|Cause|Runtime)' | tr '\n' ' ' | cut -c1-200)"
  [ "$S" = completed ] && [ "$JRC" = 0 ] && { ok=1; break; }
  case "$S" in
    broker-kill | power-cycle | reboot | interrupted | abandoned) drop=1 ;;
    *) [ -n "$new" ] && drop=1 || drop=0 ;;
  esac
  # A Python traceback is our own failure; a chip lost in the post-job gate after it is not a drop of this job.
  if [ $drop = 1 ] && grep -q '^Traceback' $O/run.log 2> /dev/null; then
    log "job $JOB failed with a traceback; incidents [$new] came after it, not counted as a drop"; drop=0
  fi
  if [ $drop = 0 ]; then
    n1=$(ls $O/*.yuv 2> /dev/null | wc -l)
    log "job $JOB FAILED (not a drop) status=$S T225_EXIT=$JRC yuv $n0 -> $n1"
    [ $n1 -gt $n0 ] || exit 9
    need=1; drops=0; continue
  fi
  for f in $new; do
    log "DROP-INCIDENT $f: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("label"), d.get("evidence"), "job=", d.get("job"), "present=", d.get("chips_present_at_capture"))' $INC/$f/incident.json 2>&1 | head -c 600)"
  done
  log "DROP utc=$t0..$(date -u '+%T') box=g15blx01 job=$JOB status=$S whose=smarton(t225) incidents=[$new]"
  drops=$((drops + 1)); need=2
  [ $drops -ge 2 ] && { log "two drops in a row: skipped on blx01"; exit 11; }
done
[ $ok = 1 ] || exit 12
log "decode: $(grep -hE 'DECODE_MEAN|brick' $O/run.log | tr '\n' ' ')"
STAGE=post
bash -c "source $F/t48/python_env/bin/activate && python -u $T/drv/post.py $O $F/diffvae/ref" > $O/post.log 2>&1 || { log "post failed"; exit 4; }
STAGE=media
for s in 0 1 2 3 4; do
  for a in 1d 2d; do
    ffmpeg -loglevel error -y -f rawvideo -pix_fmt yuv420p -s 1920x1088 -r 24 -i $O/${a}_seed$s.yuv -c:v libx264 -crf 12 -pix_fmt yuv420p $O/${a}_seed$s.mp4 >> $L 2>&1 || exit 5
  done
done
(cd $O && md5sum *.yuv > MD5SUMS)
log "media done; $(ls $O | tr '\n' ' ')"
STAGE=done
exit 0
