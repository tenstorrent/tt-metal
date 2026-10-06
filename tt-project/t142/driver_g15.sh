#!/bin/bash
# t142 on g15blx02: wait until the broker is healthy (3 checks in a row) and no other smarton job runs or
# queues, then ONE 5-seed job (t142/job_g15.sh: t166 tree, warmup cuts, LTX_FRESH_PROMPTS=0), then score
# each seed off-device against ref_dv145/seed<N>.mp4. A broker-kill (chips left PCIe) is a drop: wait for
# recovery and rerun all 5 seeds (the job is ~150 s); two drops in a row skip. Marker: $V/DONE.
set -o pipefail
P=/home/smarton/fasth3/tt-metal/tt-project; W=$P/worktrees/t158; V=$P/data/g15/t142; mkdir -p $V
REF=$P/baselines/ltx25_1080p_6s/ref_dv145; M=models.tt_dit.tests.models.ltx.tools.ltx_eval
SL=/var/log/tt-device-broker/server.log; D=$V/driver.log; BOOT0=$(uptime -s); TO=300
OKRE=': OK|device healthy; no reset needed|heartbeat: HEALTHY'
BADRE='[|] ERROR [|]|ESCALATE|RECOVER'
log() { echo "[$(date -u '+%F %T')] $*" | tee -a $D; }
done_() { log "DONE rc=$1 $2"; echo "rc=$1 $2" > $V/DONE; exit 0; }
health() {
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  st=$(tt-device-mcp status 1 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | sed -n '/^RUNNING/,/^QUEUED/p' | grep -qiE '🔧|health-gate|fabric-check|recover|upgrade|bridge-reset|power-cycle' && { log "health: broker gate/recovery running"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
wait_ready() {
  ok=0
  for i in $(seq 360); do
    if health; then ok=$((ok + 1)); else ok=0; fi
    if [ $ok -ge 3 ]; then
      active=$(tt-device-mcp status 1 2>&1 | sed -n "/^RUNNING/,/^RECENT/p" | grep -w smarton)
      [ -z "$active" ] && return 0
      log "busy: $(echo $active | cut -c1-160)"; ok=0
    fi
    sleep 60
  done
  done_ 8 "broker not healthy or free for 6 h"
}
drops=0; k=0; prev=
while :; do
  wait_ready; k=$((k + 1)); tag=r$k; OUT=$V/out/$tag
  out=$(cd $W && tt-device-mcp run-bg "env TAG=$tag PYTEST_S=$((TO - 30)) bash $P/t142/job_g15.sh" -w $W -e $W/tmp/t158/env.yaml -t $TO 2>&1); src=$?
  log "$tag submit rc=$src -t $TO: $(echo "$out" | tr '\n' ' ' | cut -c1-200)"
  [ $src = 0 ] || done_ 7 "submit failed"
  JOB=$(echo "$out" | grep -oE '[0-9]+' | tail -1); echo "$tag $JOB $(date -u '+%F %T')" >> $V/jobs.txt
  while :; do
    ST=$(tt-device-mcp status -j $JOB 2>&1)
    echo "$ST" | grep -qiE "^Status: *(running|queued|pending)" || break
    [ "$(uptime -s)" = "$BOOT0" ] || done_ 9 "host reboot during job $JOB"
    sleep 30
  done
  log "job $JOB: $(echo "$ST" | grep -iE '^(Status|Exit|Runtime|Cause)' | tr -s ' ' | tr '\n' ' ' | cut -c1-300)"
  grep -q 'T142_EXIT=0' $OUT/run.log 2>/dev/null && break
  if echo "$ST" | grep -qiE 'broker-kill|left PCIe'; then
    drops=$((drops + 1)); prev="$prev $JOB"; log "DROP job $JOB (ours, t142) $(date -u '+%F %T') UTC"
    [ $drops -ge 2 ] && done_ 6 "skipped: two drops in a row (jobs$prev)"
    continue
  fi
  done_ 5 "job $JOB failed without a drop; see $OUT/run.log"
done
grep -E 'E2E_WALL_S|encode|done in|process wall|OVERLAY|overlay=' $OUT/run.log > $V/summary.txt
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
for i in 0 1 2 3 4; do
  c=$OUT/ltx_av_fast_1920x1088_$((i + 1)).mp4; r=$REF/seed$i.mp4
  [ -e $c ] || { log "missing $c"; continue; }
  extra=; [ $i = 0 ] && extra=--vbench-ref
  (cd $W && timeout 2400 python -m $M video --ref $r --cand $c --out $V/eval/seed$i $extra) >> $V/eval.log 2>&1
  log "eval seed$i rc=$?"
done
grep -h '^QUALITY' $V/eval.log > $V/quality.txt
done_ 0 "ok job $JOB tag $tag drops=$drops"
