# shellcheck shell=bash
# broker yield: under a tt-device-mcp broker job, run.sh gives the device up within 30 s
# of a queued job of another owner. by_start (at fl_start, so setup is watched too) starts a watcher that reads the broker queue
# (broker_collision.py queued: read only; ours are this job's cgroup id and TRAY_BROKER_OWN_FILE) every
# TRAY_BROKER_YIELD_POLL_S s (default 10) and writes $FL_OUT/yield.txt ('<epoch> <id> (<owner>)') on the first foreign
# queued job. The watcher signals nothing: fl_watch reads yield.txt every 1 s and stops this run's own load jobs on its
# own kill path, and by_check sets FL_STOP, so later stages skip, C9 (glx_reset, rescan, recount) still runs (fl_end), and
# fl_verdict prints RESULT INCOMPLETE 'yielded the device to broker job <id> (<owner>)': never a fault, never a pass
# Off a broker (hold_broker false) nothing starts. by_stop ends the watcher (after the load stages,
# and in the EXIT trap). An unreadable queue is one ERROR line, then the watcher keeps reading.
# Ours: this job's id, TRAY_BROKER_OWN_FILE's ids, and queued jobs whose command matches BY_OWN_RE (our own
# tray-test run.sh; TRAY_BROKER_OWN_RE overrides it). Other jobs of the same user stay foreign.
BY_PID=""; FL_YIELD=""; FL_YIELD_AT=""; BY_OWN_RE='(^|[ /])run[.]sh( |$)'
by_file() { echo "${FL_OUT:-${OUT:-.}}/yield.txt"; }
by_on() { [ -s "$(by_file)" ]; }
by_start() {
  [ -z "$BY_PID" ] || return 0
  declare -F hold_broker > /dev/null && hold_broker || return 0
  local f main=$$ poll=${TRAY_BROKER_YIELD_POLL_S:-10} py=${TT_DIR:-.}/broker_collision.py
  f=$(by_file)
  say "broker yield: watching the broker queue every ${poll} s; a queued job of another owner ends the load"
  (
    trap - EXIT; local q rc err=""   # the watcher never runs run.sh's EXIT trap
    while kill -0 "$main" 2> /dev/null; do
      rc=0; q=$(timeout 60 python3 "$py" queued --own-re "${TRAY_BROKER_OWN_RE:-$BY_OWN_RE}" ${TRAY_BROKER_OWN_FILE:+--own-file "$TRAY_BROKER_OWN_FILE"} 2> /dev/null) || rc=$?
      if [ "$rc" = 0 ] && [ -n "$q" ]; then
        echo "$(date +%s) ${q%%,*}" > "$f.tmp" && mv "$f.tmp" "$f"
        echo "broker yield: broker job ${q%%,*} queued: stopping the load (C9 reset follows)" >&2
        exit 0
      elif [ "$rc" != 1 ] && [ -z "$err" ]; then
        err=1; echo "ERROR broker yield: broker queue unreadable (broker_collision.py queued rc $rc); still reading it" >&2
      fi
      sleep "$poll"
    done
  ) &
  BY_PID=$!
}
by_stop() {  # the watcher and its children (its sleep or 'timeout 60 python3 broker_collision.py queued'), by pid
  [ -n "$BY_PID" ] || return 0; local c; c=$(ps -o pid= --ppid "$BY_PID" 2> /dev/null)
  kill "$BY_PID" 2> /dev/null; for c in $c; do kill "$c" $(ps -o pid= --ppid "$c" 2> /dev/null) 2> /dev/null; done; wait "$BY_PID" 2> /dev/null; BY_PID=""; by_check; return 0; }
by_check() {  # main shell: once yield.txt is in, FL_YIELD and FL_STOP (the stage reached from phases.txt)
  [ -z "$FL_YIELD" ] && by_on || return 0
  local l; l=$(head -1 "$(by_file)")
  if [ -n "${FL_STOP:-}" ] || [ "${seen:-0}" = 1 ]; then   # the load had already stopped on its own: that run's verdict stands
    [ -n "${BY_LATE:-}" ] || { BY_LATE=1; say "broker yield: broker job ${l#* } queued after the load had stopped (${FL_STOP:-chips lost}); the run goes on to its end"; }
    return 0
  fi
  FL_YIELD="yielded the device to broker job ${l#* }"
  FL_YIELD_AT=$(awk 'END { print $1 }' "${FL_OUT:-.}/phases.txt" 2> /dev/null); FL_YIELD_AT=${FL_YIELD_AT:-setup}
  say "$FL_YIELD in stage $FL_YIELD_AT ($(( $(date +%s) - ${l%% *} )) s after the queue read); the C9 reset follows"
  FL_STOP=${FL_STOP:-$FL_YIELD}
  return 0
}
