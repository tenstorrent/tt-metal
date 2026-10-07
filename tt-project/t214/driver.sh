#!/bin/bash
# t214 driver on blx01: job A (pipeline latents), map latents, job B (DiffVAE reference decodes), mp4 + stills.
# One broker job at a time, each after the broker health check (t159's gate). A drop waits for two clean
# passes and reruns; two drops in a row on one job stop the driver (rc 9). Job B resumes per seed, so a
# non-drop failure that still wrote new seeds is retried too (at most 3 attempts per job).
# Final marker: $D/driver.marker "T214_DRIVER_DONE stage=.. rc=.. jobs=.."
F=/var/tmp/fasth3; D=$F/diffvae; S=$D/scripts; L=$D/driver.log; M=$D/driver.marker
INC=/var/lib/tt-device-broker/health/incidents; FSM=/var/lib/tt-device-broker/health/fsm.json
STAGE=start; JOBS=
trap 'rc=$?; echo "T214_DRIVER_DONE stage=$STAGE rc=$rc jobs=${JOBS# }" > $M' EXIT
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
# run_job <name> <script> <timeout_s> <exit-tag> <log-glob>; returns 0 ok, 1 failed, 9 two drops in a row
run_job() {
  local name=$1 script=$2 tmo=$3 tag=$4 glob=$5 need=1 drops=0 a JOB st s new jrc drop t0 inc0 nyuv0
  for a in 1 2 3; do
    STAGE=$name-health$a
    wait_health $need || { log "$name: broker never healthy"; return 8; }
    STAGE=$name-job$a
    inc0=$(ls $INC 2> /dev/null | sort | tail -1); nyuv0=$(ls $D/ref/*.yuv 2> /dev/null | wc -l)
    t0=$(date -u '+%F %T')
    out=$(timeout 120 tt-device-mcp run-bg "env T214_ATTEMPT=$a bash $script" -w $F/t48 -e $S/env.yaml -t $tmo 2>&1)
    JOB=$(echo "$out" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p' | head -1)
    log "$name attempt $a submit -t $tmo: $(echo "$out" | tr '\n' ' ' | cut -c1-200) JOB=$JOB"
    [ -n "$JOB" ] || return 7
    JOBS="$JOBS $name:$JOB"
    while :; do
      st=$(timeout 60 tt-device-mcp status -j $JOB 2>&1)
      s=$(echo "$st" | sed -n 's/^Status: *//p' | head -1)
      case "$s" in queued | running | pending | "") sleep 30 ;; *) break ;; esac
    done
    log "$name attempt $a job $JOB status=$s $(echo "$st" | grep -iE '^(Exit|Cause|Runtime)' | tr '\n' ' ' | cut -c1-200)"
    sleep 120  # the broker's post-job gate
    new=$(ls $INC 2> /dev/null | sort | awk -v z="$inc0" '$0 > z' | paste -sd' ' -)
    jrc=$(cat $glob 2> /dev/null | grep -oE "$tag=[0-9]+" | tail -1 | cut -d= -f2)
    case "$s" in
      broker-kill | power-cycle | reboot | interrupted | abandoned) drop=1 ;;
      completed) [ "$jrc" = 0 ] && drop=0 || { [ -n "$new" ] && drop=1 || drop=0; } ;;
      *) [ -n "$new" ] && drop=1 || drop=0 ;;
    esac
    if [ $drop = 0 ]; then
      [ "$s" = completed ] && [ "$jrc" = 0 ] && { log "$name attempt $a OK (job $JOB)"; return 0; }
      log "$name attempt $a FAILED (not a drop) status=$s $tag=$jrc"
      [ "$name" = decode ] && [ $(ls $D/ref/*.yuv 2> /dev/null | wc -l) -gt $nyuv0 ] && { drops=0; need=1; continue; }
      return 1
    fi
    for f in $new; do
      log "DROP-INCIDENT $f: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("label"), d.get("evidence"), "job=", d.get("job"), "present=", d.get("chips_present_at_capture"))' $INC/$f/incident.json 2>&1 | head -c 600)"
    done
    log "DROP utc=$t0..$(date -u '+%T') box=g15blx01 job=$JOB status=$s whose=smarton(t214 $name) incidents=[$new]"
    drops=$((drops + 1)); need=2
    [ $drops -ge 2 ] && { log "$name: two drops in a row: skip on blx01"; return 9; }
  done
  return 1
}
log "driver start; disk: $(df -h /var/tmp | tail -1)"
if [ ! -e $D/latents/seed4.pt ]; then
  run_job latents $S/run_lat.sh ${TMO_A:-600} T214A_EXIT "$D/pipeline/run.log" || exit $?
  STAGE=map
  bash -c "source $F/t48/python_env/bin/activate && python $S/map_latents.py $D/pipeline/run.log $D/latents_raw $D/latents" >> $L 2>&1 || { log "map failed"; exit 4; }
  rm -rf $D/latents_raw
fi
run_job decode $S/run_dec.sh ${TMO_B:-600} T214B_EXIT "$D/ref/run_dec.*.log" || exit $?
STAGE=media
fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 $(ls $D/pipeline/*.mp4 | head -1) 2> /dev/null)
fps=${fps:-24}
for s in 0 1 2 3 4; do
  y=$D/ref/ref_dvx_seed$s.yuv
  ffmpeg -loglevel error -y -f rawvideo -pix_fmt yuv420p -s 1920x1088 -r $fps -i $y -c:v libx264 -crf 12 -pix_fmt yuv420p $D/ref/ref_dvx_seed$s.mp4 >> $L 2>&1
  ffmpeg -loglevel error -y -ss 3 -i $D/ref/ref_dvx_seed$s.mp4 -frames:v 1 -q:v 2 $D/ref/ref_dvx_seed${s}_t3s.jpg >> $L 2>&1
done
for mp4 in $D/pipeline/*.mp4; do ffmpeg -loglevel error -y -ss 3 -i $mp4 -frames:v 1 -q:v 2 ${mp4%.mp4}_t3s.jpg >> $L 2>&1; done
(cd $D && md5sum latents/*.pt ref/*.yuv > MD5SUMS)
log "media done fps=$fps; $(ls $D/ref | tr '\n' ' ')"
exit 0
