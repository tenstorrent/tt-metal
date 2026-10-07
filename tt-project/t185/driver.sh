#!/bin/bash
# t185 driver on g15blx02 (copy of t186's): run each S2 step-cut arm as its own blx01 broker job, one after another,
# then score it. Before each submit it waits for a healthy, project-free blx01 (no broker-owned job,
# not HELD, no other smarton job running or queued; two passes 60 s apart). A broker kill (drop) is
# logged and rerun once; a second drop of the same arm skips it. A job the broker re-queued keeps its
# id, so watching that id never submits a duplicate.
# Usage: bash driver.sh <label>=<sigmas> ...     Marker: $D/DRIVER.done = "<first non-zero rc> <reason>"
set -o pipefail
D=/home/smarton/fasth3/tt-metal/tt-project/t185
H=g15blx01; RUN=/var/tmp/fasth3/t185/run_cfg.sh; WS=/var/tmp/fasth3/t48
TMO=240; READY_WAIT_S=21600
first_rc=0; reason=ok; posts=()
trap 'echo "$first_rc $reason" > $D/DRIVER.done' EXIT
log() { echo "$(date -u '+%F %T') $*" | tee -a $D/driver.log; }
bst() { ssh -o ConnectTimeout=20 $H "tt-device-mcp status $*" 2>&1; }

ready_once() {
  local s run queued
  s=$(bst) || return 1
  echo "$s" | grep -q '^RUNNING' || return 1
  run=$(echo "$s" | sed -n '/^RUNNING/,/^QUEUED/p')
  queued=$(echo "$s" | sed -n '/^QUEUED/,/^RECENT/p')
  echo "$run" | grep -qE 'HELD|🔧' && return 1
  echo "$run$queued" | grep -q 'smarton' && return 1
  return 0
}

wait_ready() {
  local t0=$(date +%s)
  while :; do
    if ready_once; then sleep 60; ready_once && return 0; fi
    (( $(date +%s) - t0 > READY_WAIT_S )) && return 1
    sleep 60
  done
}

# Submit and watch one job under the blx01-device lock; prints "<status> <exit> <job>".
one_job() {
  local label=$1 sig=$2 out job st ex
  out=$(ssh $H "cd $WS && tt-device-mcp run-bg 'env PYTEST_S=220 bash $RUN $label LTX_FRESH_PROMPTS=0 LTX_E2E_SEEDS=0,1,2,3,4 LTX_E2E_EXTRA_REPLAYS=0 LTX_S2_SIGMAS=$sig' -w $WS -e /var/tmp/fasth3/t159/env.yaml -t $TMO" 2>&1)
  job=$(echo "$out" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p' | head -1)
  [ -n "$job" ] || { log "$label: submit failed: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"; echo "submitfail - -"; return; }
  log "$label: submitted job $job ($sig)"
  while :; do
    st=$(bst -j $job | sed -n 's/^Status: *//p')
    echo "$st" | grep -qiE 'running|queued' || break
    sleep 20
  done
  ex=$(bst -j $job | sed -n 's/^Exit: *//p')
  bst -j $job > $D/job$job.status
  echo "$st ${ex:--} $job"
}

for arm in "$@"; do
  label=${arm%%=*}; sig=${arm#*=}; drops=${DROPS0:-0}
  while :; do
    log "$label: waiting for blx01 ready"
    wait_ready || { log "$label: blx01 not ready after ${READY_WAIT_S}s"; [ $first_rc -eq 0 ] && first_rc=75; reason="notready-$label"; break 2; }
    res=$(ttp lock --timeout 600 blx01-device -- bash -c "$(declare -f log bst one_job); D=$D H=$H RUN=$RUN WS=$WS TMO=$TMO; one_job $label $sig" | tail -1)
    lrc=$?
    [ $lrc -eq 75 ] && { log "$label: blx01-device lock busy, retrying"; continue; }
    read -r st ex job <<< "$res"
    log "$label: job $job ended status=$st exit=$ex"
    if [ "$st" = completed ] && [ "$ex" = 0 ]; then
      bash $D/post.sh $label > $D/post_$label.log 2>&1 &
      posts+=("$!:$label")
      break
    fi
    if echo "$st" | grep -qiE 'killed|abandoned|interrupted|power|reboot'; then
      drops=$((drops + 1))
      log "DROP $label: job $job status=$st (drop $drops for this arm); cause: $(grep -m1 '^Cause' $D/job$job.status)"
      [ $drops -ge 2 ] && { log "$label: dropped twice on blx01, skipped"; [ $first_rc -eq 0 ] && { first_rc=86; reason="skip2drops-$label"; }; break; }
      continue
    fi
    [ $first_rc -eq 0 ] && { first_rc=1; reason="fail-$label-$st-$ex"; }
    break
  done
done
for p in "${posts[@]}"; do
  wait ${p%%:*}; prc=$?
  log "${p#*:}: post rc=$prc"
  [ $prc -ne 0 ] && [ $first_rc -eq 0 ] && { first_rc=$prc; reason="post-${p#*:}"; }
done
