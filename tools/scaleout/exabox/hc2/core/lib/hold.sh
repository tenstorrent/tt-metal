# shellcheck shell=bash
# lib/hold.sh: run.sh --hold. Functions only; run.sh sources it after say/OUT/RK_MAP exist.
# On a detected hang where every chip is still on PCIe (a link/CCL hang, an all-gather stall, a chip whose NOC hangs),
# after the normal automatic capture: keep this run's hung job alive and do not reset, print one banner (host, the hung
# chips by BDF and tray 1-4, the PIDs, the tt-triage / tt-exalens commands to attach, the STOP file, the deadline), and
# wait until the STOP file exists or the hold timeout passes. Then the caller stops its own job and the run goes on
# (reset, rest of the run) as without --hold. One hold per run: a later hang is captured and handled as today.
# Not for drops (a chip off PCIe), numeric faults or chips lost at a reset: those never reach hold_point, and a hang
# with a chip off PCIe is not held.
#   HOLD=1               --hold
#   HOLD_TIMEOUT_MIN     --hold-timeout (default 240)
#   HOLD_STOP            the STOP file (default <run dir>/STOP)
#   HOLD_POLL_S          STOP/timeout poll (default 5); HOLD_SAY_S: one "holding" line this often (default 20: a job
#                        runner reaps a job silent for about a minute); HOLD_BROKER_POLL_S broker poll (default 30)
# Refused under Slurm (SLURM_JOB_ID): a shared-cluster job never waits on a human. Under a tt-device broker the hold
# ends as soon as another broker job is queued, or the broker cannot be read (broker_collision.py queued).
HOLD=${HOLD:-0}; HOLD_TIMEOUT_MIN=${HOLD_TIMEOUT_MIN:-240}; HOLD_ACTIVE=0; HOLD_USED=""   # run.sh sets HOLD and the timeout before its options
HOLD_ROOT=${HOLD_ROOT:-$PWD}

hold_init() {  # after the options and OUT: validate, refuse under Slurm, arm
  [[ $HOLD_TIMEOUT_MIN =~ ^[1-9][0-9]*$ ]] || { echo "--hold-timeout takes whole minutes > 0 (got '$HOLD_TIMEOUT_MIN')" >&2; exit 1; }
  [ "$HOLD" = 1 ] || return 0
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    say "--hold refused: under Slurm (job $SLURM_JOB_ID) a shared-cluster job must never wait on a human or keep a node idle; the run goes on without --hold"
    HOLD=0; return 0
  fi
  HOLD_STOP=${HOLD_STOP:-$OUT/STOP}; rm -f "$HOLD_STOP"   # a STOP left from before would end the hold at once
  HOLD_ACTIVE=1
  # a hold capture reads every chip (tt-triage --dev=all: NOC, ERISC, ARC, callstacks; the default in_use skips all once the job ended) and keeps the run's Inspector log
  export HOLD_ACTIVE HOLD_INSPECTOR=${HOLD_INSPECTOR:-$OUT/inspector} SNAP_TRIAGE_ARGS=${SNAP_TRIAGE_ARGS---disable-progress --skip-version-check --dev=all}
  say "--hold on: a hang with every chip on PCIe is held after its capture (no kill, no reset) until $HOLD_STOP exists or $HOLD_TIMEOUT_MIN min pass$(hold_broker && echo ', or another broker job queues')"
}
hold_broker() {  # 0 under a tt-device broker (run.sh's test, plus our own ttdev-job cgroup)
  [ -n "${TRAY_REPRO_BROKER:-$(env | grep -m1 -o '^TT_DEVICE_BROKER[A-Z_]*')}" ] && return 0
  grep -q 'ttdev-job-[0-9]*\.scope' "${BROKER_CGROUP:-/proc/self/cgroup}" 2>/dev/null
}
hold_triage() {  # the triage command noc_snapshot.sh would use
  if [ -n "${SNAP_TRIAGE:-${HEAVY_TRIAGE:-}}" ]; then echo "${SNAP_TRIAGE:-$HEAVY_TRIAGE}"
  elif [ -f "${TT_METAL_HOME:-/nonexistent}/tools/triage/triage.py" ]; then
    # run it with a python that has ttexalens (the run's venv), not a bare system python3
    local vp; vp=$(tools_venv_python "${HEAVY_PY:-${PY:-python3}}") || vp=""
    [ -n "$vp" ] && echo "$vp $TT_METAL_HOME/tools/triage/triage.py" || echo "python3 $TT_METAL_HOME/tools/triage/triage.py"
  elif command -v tt-triage > /dev/null; then echo tt-triage
  else echo "tt-triage (not found: set TT_METAL_HOME or put tt-triage on PATH)"; fi
}
hold_tray() { declare -F rk_tray > /dev/null && { rk_tray "$1"; return; }; local b=${1#0000:}; case ${b:0:1} in 0|1|2|3) echo 1 ;; 4|5|6|7) echo 2 ;; c|d|e|f) echo 3 ;; 8|9|a|b) echo 4 ;; *) echo '?' ;; esac; }
hold_banner() {  # <what> <pids> <bdf...>
  local what=$1 pids=$2 b p as="${HEAVY_AS:-}${HEAVY_AS:+ }" tri py=${HEAVY_EXALENS_PY:-${HEAVY_PY:-python3}}; shift 2
  tri=$(hold_triage)
  echo "==================== HOLD: hung chips kept as they are (run.sh --hold) ===================="
  echo "host:     $H"
  echo "hang:     $what"
  echo "chips:    $(for b in "$@"; do printf '%s (tray %s)\n' "${b#0000:}" "$(hold_tray "$b")"; done | paste -sd, - | sed 's/,/, /g')"
  echo "pids:     ${pids:-none: the job ended by itself after its snapshot; the chips stay as the hang left them, not reset}"
  for p in $pids; do ps -o pid=,pgid=,etimes=,cmd= -g "$p" 2>/dev/null | cut -c1-160 | sed 's/^/          /'; done
  echo "inspector: ${HOLD_INSPECTOR:-$OUT/inspector} (the run's Inspector log; TT_METAL_INSPECTOR_LOG_PATH for the attach lines)"
  echo "attach (read-only; NOC0 then NOC1):"
  echo "  $as$tri --disable-progress --skip-version-check --dev=all --noc-id=0"
  echo "  $as$tri --disable-progress --skip-version-check --dev=all --noc-id=1"
  for b in "${@:1:2}"; do for p in 0 1; do   # two chips at most: the rest take the same line with their own BDF
    echo "  $as$py $HOLD_ROOT/lib/exalens_probe.py --noc $p --bdf $b --unique-id none --out $OUT/hold_exalens_noc${p}_${b//:/_}.json"
  done; done
  [ $# -gt 2 ] && echo "  (the other $(( $# - 2 )) chips above: the same two lines with their own --bdf)"
  command -v tt-exalens > /dev/null && echo "  ${as}tt-exalens   (interactive)"
  echo "capture:  $OUT (already taken, as without --hold)"
  echo "stop:     touch $HOLD_STOP"
  echo "deadline: $(date -d "@$HOLD_END" '+%F %H:%M %Z') ($HOLD_TIMEOUT_MIN min)$(hold_broker && echo '; ends sooner if another broker job queues')"
  echo "then:     this run stops its hung job, resets and finishes as without --hold"
  echo "=========================================================================================="
}
hold_point() {  # <what> <pids, space separated; may be empty> <bdf...> (none: every chip): hold here, return when it ends
  [ "${HOLD_ACTIVE:-0}" = 1 ] || return 0
  local what=$1 pids=$2 lost why="" q rc k=0 t0; shift 2
  if [ -n "$HOLD_USED" ]; then say "--hold: not holding $what: this run already held once ($HOLD_USED)"; return 0; fi
  lost=$(rk_snap | awk '$3 != "ok" { print $1 }')
  if [ -n "$lost" ]; then
    say "--hold: not holding $what: $(for b in $lost; do printf '%s (tray %s) ' "${b#0000:}" "$(hold_tray "$b")"; done)off PCIe (a drop, not a hang)"
    HOLD_USED="not held: $what (drop)"; return 0
  fi
  [ $# -gt 0 ] || set -- $(echo "$RK_MAP" | awk 'NF { print $1 }')
  t0=$(date +%s); HOLD_END=$(( t0 + ${HOLD_TIMEOUT_S:-$(( HOLD_TIMEOUT_MIN * 60 ))} ))   # HOLD_TIMEOUT_S: tests only
  HOLD_USED=$what
  hold_banner "$what" "$pids" "$@" | tee -a "$OUT/hold.txt"
  while :; do
    [ -e "$HOLD_STOP" ] && { why="STOP file $HOLD_STOP"; break; }
    [ "$(date +%s)" -ge "$HOLD_END" ] && { why="hold timeout ($HOLD_TIMEOUT_MIN min)"; break; }
    if hold_broker && [ $(( k % ${HOLD_BROKER_POLL_S:-30} )) = 0 ]; then
      q=$(timeout 60 python3 "$HOLD_ROOT/broker_collision.py" queued 2>/dev/null); rc=$?
      [ $rc = 0 ] && { why="another broker job waits: ${q:-queued}"; break; }
      [ $rc = 1 ] || { why="broker status unreadable (rc $rc): ended so no broker job is kept waiting"; break; }
    fi
    [ $k -gt 0 ] && [ $(( k % ${HOLD_SAY_S:-20} )) = 0 ] && say "--hold: holding $what ($(( k / 60 )) min); touch $HOLD_STOP to end, deadline $(date -d "@$HOLD_END" '+%H:%M %Z')"
    sleep "${HOLD_POLL_S:-5}"; k=$(( k + ${HOLD_POLL_S:-5} ))
  done
  say "--hold ended after $(( ($(date +%s) - t0) / 60 )) min: $why; stopping the held job, then the reset and the rest of the run" | tee -a "$OUT/hold.txt"
  return 0
}
hold_done() {  # after the caller stopped its held job: the exalens probe heavy_logs_on_hang left for later (device busy)
  [ -n "${HL_EXA_LATER:-}" ] || return 0
  local h; read -r -a h <<< "$HL_EXA_LATER"; HL_EXA_LATER=""
  declare -F hl_exalens > /dev/null && hl_exalens "${h[@]}" > "${h[0]}/exalens_after_hold.log" 2>&1
  return 0
}
