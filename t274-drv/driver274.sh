#!/bin/bash
# t274 driver on blx01 (no device work itself): incremental build of $F/t263/b (t263 is closed; reused as
# t260 reused t252/b) at REV, then ONE broker job if the broker is healthy and idle of smarton jobs.
# Writes the job id to job.id and a marker; the device job is polled by the broker id, not by this driver.
F=/var/tmp/fasth3; T=$F/t274; D=$T/drv; L=$D/driver.log; M=$D/driver.marker
A=$F/t48; B=$F/t263/b; REV=$1; ARMS=$2; O=$3; TMO=$4; PARMS=$5
FSM=/var/lib/tt-device-broker/health/fsm.json
STAGE=start
trap 'rc=$?; echo "T274_DRIVER_DONE stage=$STAGE rc=$rc job=$(cat $D/job.id 2>/dev/null)" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache
rm -f $M $D/job.id
log "start REV=$REV arms=$ARMS out=$O t=$TMO"
STAGE=build
bash $D/build274.sh $REV > $T/build_$REV.log 2>&1 || { log "build failed: $(tail -3 $T/build_$REV.log | tr '\n' ' ')"; exit 20; }
log "built $B @ $(git -C $B rev-parse --short=11 HEAD)"
STAGE=health
[ "$(systemctl is-active tt-device-broker 2> /dev/null)" = active ] || { log "broker inactive"; exit 30; }
pgrep -f /opt/tt-device-broker/autoupdate.sh > /dev/null && { log "broker upgrade running"; exit 31; }
st=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' $FSM 2> /dev/null)
[ "$st" = healthy ] || { log "fsm=$st"; exit 32; }
STAGE=submit
out=$(timeout 120 tt-device-mcp run-bg "bash $D/run274.sh '$ARMS' $O '$PARMS'" -w $T -e $D/env.yaml -t $TMO 2>&1)
JOB=$(echo "$out" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p' | head -1)
log "submit: $(echo "$out" | tr '\n' ' ' | cut -c1-200) JOB=$JOB"
[ -n "$JOB" ] || exit 40
echo $JOB > $D/job.id
STAGE=submitted
exit 0
