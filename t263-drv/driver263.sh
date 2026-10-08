#!/bin/bash
# t263 driver on blx01: wait until no other project driver runs, build $F/t252/b (reused, as t260 did) at REV
# (t48 8a99339c424 + DIFFVAE_NA_ABLATE diagnostic, incremental from t260's 89d04d2ab71), then TWO short broker
# jobs, one arm per process, all profiled: A = def + reads (DIFFVAE_NA_ABLATE=reads), B = math.
# The stage trees give neighborhood-sdpa ms/block per arm. Ablated outputs are garbage: no scoring.
# Health/drop logic from t260's driver. Never resets, never touches other jobs.
# Marker: $T/drv/driver.marker "T263_DRIVER_DONE stage=.. rc=.. jobs=.."
F=/var/tmp/fasth3; T=$F/t263; L=$T/drv/driver.log; M=$T/drv/driver.marker; W=$T
INC=/var/lib/tt-device-broker/health/incidents; FSM=/var/lib/tt-device-broker/health/fsm.json
PY=$F/t48/python_env/bin/python; REV=6a23f8dfe10; A=$F/t48; B=$F/t252/b
PARMS="def reads math"
STAGE=start; JOBS=
trap 'rc=$?; echo "T263_DRIVER_DONE stage=$STAGE rc=$rc jobs=${JOBS# }" > $M' EXIT
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
    out=$(timeout 120 tt-device-mcp run-bg "bash $T/drv/run263.sh '$arms' $O '$PARMS'" -w $W -e $T/drv/env.yaml -t $tmo 2>&1)
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
    JRC=$(grep -oE 'T263_EXIT=[0-9]+' $O/run.log 2> /dev/null | tail -1 | cut -d= -f2)
    log "$tag job $JOB status=$S T263_EXIT=$JRC new_incidents=[$new] $(echo "$st" | grep -iE '^(Exit|Cause|Runtime)' | tr '\n' ' ' | cut -c1-200)"
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
    log "DROP utc=$t0..$(date -u '+%T') box=g15blx01 job=$JOB status=$S whose=smarton(t263 $tag) incidents=[$new]"
    mv $O ${O}_drop$a 2> /dev/null
    need=2
  done
  log "$tag: two drops (or drop after cold timeout), skipped"; S=skipped; JRC=
}
log "start; disk: $(df -h / | tail -1)"
STAGE=wait_drivers
for i in $(seq 480); do
  o=$(pgrep -af 'driver2[0-9][0-9]\.sh' | grep -v driver263 | cut -d' ' -f1 | paste -sd' ' -)
  [ -z "$o" ] && break
  [ $((i % 10)) = 1 ] && log "waiting for other project drivers: $o"
  sleep 60
done
[ -z "$o" ] || { log "other drivers still running after 8 h"; exit 6; }
STAGE=build
(
  set -eux
  git -C $A cat-file -e $REV^{commit} 2> /dev/null || git -C $A fetch -q $T/drv/t263.bundle "ttp/t263-r1b-widen-diffvae-na-k-v-l1-ring-bf8-k-o:refs/t263/base"
  cd $B
  git checkout -q --detach $REV
  test "$(git rev-parse --short=11 HEAD)" = $REV
  git submodule update --init tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors tt_metal/third_party/umd
  ./build_metal.sh --build-type Release
  test -f ttnn/ttnn/_ttnn.so
) > $T/build_$REV.log 2>&1 || { log "build failed: $(tail -3 $T/build_$REV.log | tr '\n' ' ')"; exit 20; }
log "built $B @ $(git -C $B rev-parse --short=11 HEAD)"
summ() { log "$1: status=$S rc=$JRC $(grep -hE 'DECODE seed|DECODE_MEAN|process wall|kv ring' $2/run.log 2> /dev/null | sort -u | tr '\n' ' ' | cut -c1-900)"
  for f in $2/stage_tree_*.txt; do log "$(basename $f) NA: $(grep -E 'neighborhood-sdpa|stage5 TOTAL|DECODE TREE' $f | awk '{$1=$1};1' | sort | uniq -c | tr '\n' ' ' | cut -c1-600)"; done; }
job A "def reads:DIFFVAE_NA_ABLATE=reads" $T/out_A 480
summ A $T/out_A
STAGE=jobB
job B "math:DIFFVAE_NA_ABLATE=math" $T/out_B 300
summ B $T/out_B
grep -q 'T263_EXIT=0' $T/out_A/run.log && grep -q 'T263_EXIT=0' $T/out_B/run.log || exit 12
STAGE=done
exit 0
