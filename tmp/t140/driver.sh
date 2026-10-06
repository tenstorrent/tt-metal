#!/bin/bash
# t140 detached driver on blx03: runs configs.txt one broker job at a time (via tmp/blx03/submit.sh).
# Waits for the t136/t141 drivers to finish, then per config: health gate -> submit -> watch -> post-job gate.
# A drop/reboot/broker kill during our job is logged and the config is rerun after the health gate passes;
# after 2 drops in a row the config is skipped. Configs whose run.log has T140_EXIT[<cfg>]=0 are skipped,
# so a relaunch resumes. CONFIGS=<file> picks another list (the 5-seed phase).
# Final marker in $D: "T140_DRIVER_DONE <summary>".
V=/var/tmp/fasth3/t140; D=$V/driver.log; SL=/var/log/tt-device-broker/server.log
R=/home/smarton/fasth3/tt-metal; S=$V/src
CONFIGS=${CONFIGS:-$S/tmp/t140/configs.txt}
BOOT0=$(uptime -s)
mkdir -p $V
log() { echo "$(date -u '+%F %T') $*" >> $D; }
now() { date -u '+%F %T'; }
OKRE=': OK|device healthy; no reset needed|heartbeat: HEALTHY'
BADRE='[|] ERROR [|]|ESCALATE|RECOVER'
errors_since() { awk -v s="$1" -v ok="$OKRE" -v bad="$BADRE" 'substr($0,1,19) > s && ($0 ~ bad || (/HEALTH-GATE/ && $0 !~ ok))' $SL; }
health() {
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  st=$(tt-device-mcp status 1 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | sed -n '/^RUNNING/,/^QUEUED/p' | grep -qiE 'health-gate|fabric-check|recover|upgrade|reset|HELD' && { log "health: broker gate/recovery running"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
wait_health() { for i in $(seq 360); do health && return 0; sleep 60; done; return 1; }
others() { pgrep -f 'bash /var/tmp/fasth3/t136/driver.sh|bash /home/smarton/fasth3/t141drv/driver.sh' > /dev/null; }
# The t136 A/B and t141 5-seed e2e go first: wait until both driver logs end in a DONE line, or until
# neither driver nor any smarton job has been seen for 2 h (they were not relaunched).
pair_done() { for t in t136 t141; do tail -1 /var/tmp/fasth3/$t/driver.log 2>/dev/null | grep -q _DRIVER_DONE || return 1; done; }
smarton_job() { tt-device-mcp status 1 2>&1 | sed -n '/^RUNNING/,/^RECENT/p' | grep -qw smarton; }
wait_pair() {
  local idle=0
  until pair_done; do
    if others || smarton_job; then idle=0; else idle=$((idle+2)); fi
    [ $idle -ge 120 ] && { log "e2e pair: no driver or job for 2 h, going ahead"; return; }
    sleep 120
  done
  log "e2e pair done"
}
our_live_job() {  # a t140 job of ours the broker still holds (e.g. re-queued after a drop)
  tt-device-mcp status 1 2>&1 | sed -n '/^RUNNING/,/^RECENT/p' | grep -w smarton | grep "t140/src/tmp/t140/run_cfg.sh $1" | grep -oE '^ *[0-9]+' | head -1 | tr -d ' '
}
watch() {  # $1 cfg, $2 job, $3 T0. 0 ok, 1 job failed (no drop), 9 drop/reboot/kill
  local st
  while :; do
    st=$(tt-device-mcp status -j $2 2>&1)
    echo "$st" | grep -qiE "^Status: *(running|queued|pending)" || break
    [ "$(uptime -s)" = "$BOOT0" ] || { log "$1 job $2: host rebooted"; return 9; }
    sleep 30
  done
  log "$1 job $2 $(echo "$st" | grep -iE '^(Status|Exit)' | tr '\n' ' ')"
  sleep 90   # the broker's post-job gate
  local e; e=$(errors_since "$3")
  awk -v s="$3" 'substr($0,1,19) >= s' $SL > $V/$1/broker_slice_$2.log 2>/dev/null
  if [ -n "$e" ] || echo "$st" | grep -qiE 'broker-kill|abandoned|killed'; then
    log "$1 job $2 DROP: $(echo "$e" | grep -oiE 'chip\(?s?\)? [0-9, ]+|tray [0-9]+|PCIe[^|]{0,80}|fabric[^|]{0,80}' | sort -u | tr '\n' ';' | cut -c1-300)"
    log "$1 job $2 first error: $(echo "$e" | head -2 | tr '\n' ' ' | cut -c1-400)"
    return 9
  fi
  grep -q "T140_EXIT\[$1\]=0" $V/$1/run.log && return 0
  return 1
}
log "start boot=$BOOT0 configs=$CONFIGS src=$(cat $S/REV)"
summary=""
wait_pair
while read -r cfg flags <&3; do
  if grep -qs "T140_EXIT\[$cfg\]=0" $V/$cfg/run.log; then log "$cfg already done"; continue; fi
  drops=0; result=""
  while [ -z "$result" ]; do
    while others; do sleep 120; done
    wait_health || { log "broker never healthy (6 h)"; log "T140_DRIVER_DONE health_timeout $summary"; exit 0; }
    job=$(our_live_job $cfg); T0=$(now)
    if [ -n "$job" ]; then
      log "$cfg attaching to live job $job (no resubmit)"
    else
      mkdir -p $V/$cfg; [ -s $V/$cfg/run.log ] && mv $V/$cfg/run.log $V/$cfg/run.log.$(date +%s)
      for i in $(seq 360); do
        out=$(cd $R && flock /var/tmp/fasth3/.submit.lock tmp/blx03/submit.sh 2400 bash $S/tmp/t140/run_cfg.sh $cfg $flags 2>&1); src=$?
        [ $src = 75 ] && { sleep 60; continue; }; break
      done
      log "$cfg submit rc=$src: $(echo "$out" | tr '\n' ' ' | cut -c1-200)"
      [ $src = 0 ] || { result="submit_rc$src"; break; }
      job=$(echo "$out" | tail -1)
    fi
    echo "$cfg $job $(now)" >> $V/jobs.txt
    watch $cfg $job "$T0"; rc=$?
    if [ $rc = 9 ]; then
      drops=$((drops+1)); log "$cfg drop #$drops (job $job)"
      [ $drops -ge 2 ] && result="skipped_2drops"
      [ "$(uptime -s)" = "$BOOT0" ] || BOOT0=$(uptime -s)
    elif [ $rc = 0 ]; then result=ok
    else result="failed_job$job"
    fi
  done
  log "$cfg -> $result"; summary="$summary $cfg=$result"
done 3< <(grep -vE '^\s*(#|$)' $CONFIGS)
log "T140_DRIVER_DONE$summary"
