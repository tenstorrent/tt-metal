#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/psu_check.sh: the box's PSUs, read in-band before the first load (the health step, fl_start) and
# after each load stage (fl_loads). Sourced by lib/fleet.sh: functions only. Why: on one box run.sh logged box
# kW as the PSU0-3 sum for 7.4 h with PSU2 at 0 W and PSU_Redundancy Lost and never warned; one PSU read
# 4120 W of its 4000 W.
#   psu_read <prefix>: `sudo -n ipmitool sdr type 'Power Supply'` -> <prefix>.sdr, then `ipmitool sensor reading
#     Power_PSU<n>...` -> <prefix>.watts. Sensor reads only (no raw, no chassis). rc 2 with the why on stdout when ipmitool
#     or sudo -n is not there or the read fails.
#   fl_psu <when> [<stage power.txt>]: fleet/psu_check.py judges the read (run.sh reads in the background: fl_psu_bg/fl_psu_join). The per-PSU W line goes to the log and to
#     load.txt beside the stage's LOAD line. A PSU Device Absent, Critical or failed, or PSU_Redundancy lost: a
#     "PSU FINDING" and FL_PSU_STOP ("PSU redundancy lost (PSU2)"): fl_plan and fl_loads start no more load, fl_verdict
#     makes the run INCOMPLETE (rc 9), fl_report's clean trays INCOMPLETE. A PSU above PSU_WARN_PCT (90) % of
#     TRAY_REPRO_PSU_RATED_W (4000 W), or the box above 14 kW: a "PSU WARN" flag only (goal 8 asks for 12-14 kW).
#     No read possible: "PSU WARN PSU check not run: <why>", never a FAIL.
# timeout --foreground: timeout and ipmitool stay in the background read's process group, so fl_psu_kill ends them too
# (plain timeout moves itself to a new process group).
# TRAY_REPRO_PSU_IPMITOOL: the ipmitool command to use as is (tests: a fake); unset: ipmitool as root, else sudo -n ipmitool when sudo -n -l allows that exact read.
FL_PSU_STOP=""; FL_PSU_AT=""
psu_read() {
  local ip=() n s
  if [ -n "${TRAY_REPRO_PSU_IPMITOOL:-}" ]; then read -ra ip <<< "$TRAY_REPRO_PSU_IPMITOOL"
    command -v "${ip[0]}" > /dev/null 2>&1 || { echo "ipmitool not found (${ip[0]})"; return 2; }
  elif ! command -v ipmitool > /dev/null 2>&1; then echo "ipmitool not found"; return 2
  elif [ "$(id -u)" = 0 ]; then ip=(ipmitool)
  elif timeout --foreground -k 2 5 sudo -n -l ipmitool sdr type 'Power Supply' > /dev/null 2>&1; then ip=(sudo -n ipmitool)   # the exact read, never sudo -n true
  else echo "no root, and sudo -n -l does not allow ipmitool sdr type 'Power Supply'"; return 2; fi
  timeout --foreground -k 5 "${PSU_READ_S:-30}" "${ip[@]}" sdr type 'Power Supply' > "$1.sdr" 2> "$1.err"; s=$?
  if [ $s != 0 ]; then echo "ipmitool sdr type 'Power Supply' rc $s$([ -s "$1.err" ] && echo ": $(head -c 200 "$1.err" | tr '\n' ' ')")"; return 2; fi
  mapfile -t n < <(grep -oE '^PSU[0-9]+' "$1.sdr" | sort -u | sed 's/^PSU/Power_PSU/')
  : > "$1.watts"; [ ${#n[@]} = 0 ] || timeout --foreground -k 5 "${PSU_READ_S:-30}" "${ip[@]}" sensor reading "${n[@]}" > "$1.watts" 2>> "$1.err"
  return 0
}
fl_psu_get() {  # $1 when, $2 file prefix, [$3 power.txt]: the read and psu_check.py's lines on stdout (no state changed)
  local why o rc
  if ! why=$(psu_read "$2"); then
    case $why in "no root, and sudo -n -l does not allow"*) [ -e "$(fl_psu_f nosudo)" ] && return 0; : > "$(fl_psu_f nosudo)" ;; esac   # once per run (the bg read is a subshell: a file, not a variable)
    echo "PSU WARN PSU check not run: $why ($1)"; return 0; fi
  o=$(python3 "$FL_DIR/psu_check.py" "$1" "$2.sdr" "$2.watts" ${3:+"$3"} 2>&1); rc=$?
  if [ $rc != 0 ]; then echo "PSU WARN PSU check not run: psu_check.py rc $rc: $(echo "$o" | tail -n1 | cut -c1-200) ($1)"; return 0; fi
  echo "$o"
}
fl_psu_apply() {  # $1 when, [$2 text added to the W line], stdin fl_psu_get's lines: findings, the W line (log, load.txt), FL_PSU_STOP
  local l had=$FL_PSU_STOP
  while IFS= read -r l; do case $l in
    "STOP "*) [ -n "$FL_PSU_STOP" ] || { FL_PSU_STOP=${l#STOP }; FL_PSU_AT=$1; } ;;
    "PSU FINDING "*|"PSU WARN "*) fl_find "$l" ;;
    ?*) case $l in "PSU $1:"*) l=$l${2:-} ;; esac; say "$l"; echo "$l" >> "$FL_OUT/load.txt" ;;
  esac; done
  [ -n "$FL_PSU_STOP" ] && [ -z "$had" ] && say "INCOMPLETE: $FL_PSU_STOP: $([ "$1" = "before load" ] && echo "no load starts" || echo "no more load after this ($1)")"
  return 0
}
fl_psu_f() { local d=$FL_OUT/psu; mkdir -p "$d"; echo "$d/$(echo "$1" | tr -c 'A-Za-z0-9\n' _)"; }
fl_psu() {  # $1 when, [$2 power.txt]: read and judge now (before load: it decides whether load starts)
  local f; f=$(fl_psu_f "$1"); fl_psu_apply "$1" < <(fl_psu_get "$1" "$f" "${2:-}")
}
# fl_psu_bg <when> [<power.txt>]: the same read in the background (two ipmitool calls, 8-11 s each read,
# sdr up to 41 s on good boxes; three of them cut a heavy-log step on one box). fl_psu_join applies
# them in order, waiting (1) up to PSU_JOIN_S (75 s; psu_read's own timeouts bound it). Stage 2 design:
# the before-load read starts at P1 and is joined at fl_start's end, after C2, before any load; after a
# stage, at the next load stage's start; the last stage's read runs over C9 and fl_verdict joins it (fl_end only applies
# it if done). No new cap. A lost PSU still stops every load stage not yet started and makes the run INCOMPLETE; no read
# is dropped. fl_psu_reap (run.sh's EXIT trap): an exit with a read pending (fl_start's exit 6) ends it.
# Each read runs in its own process group (process group id = the subshell's pid; set -m around the start;
# not handed to pg_reap: it would count the finished, not yet waited subshell as live), and fl_psu_reap and
# fl_psu_join's overtime path end that whole group (fl_psu_kill), so no timeout or ipmitool of ours outlives run.sh. A read that ended after C9 started (its .done time against C9 in
# phases.txt) gets its W line marked: its watts may include the C9 glx_reset. Label only; stop and verdict unchanged.
FL_PSU_BG=()
fl_psu_bg() {
  local f m=; f=$(fl_psu_f "$1"); rm -f "$f.out" "$f.done"
  case $- in *m*) m=1 ;; esac; set -m   # its own process group (no tty needed); back as it was right after
  ( trap - USR1 INT TERM EXIT; fl_psu_get "$1" "$f" "${2:-}" > "$f.out" 2>&1; echo $? > "$f.done" ) < /dev/null > /dev/null 2>&1 &
  FL_PSU_BG+=("$!|$f|$1"); [ -n "$m" ] || set +m
  return 0
}
fl_psu_live() {  # a live (not zombie) process in process group $1, from /proc/<pid>/stat (state, parent, process group after the name)
  local s r a; for s in /proc/[0-9]*/stat; do read -r r 2> /dev/null < "$s" || continue; read -ra a <<< "${r##*) }"; [ "${a[2]:-}" = "$1" ] && [ "${a[0]}" != Z ] && return 0; done; return 1
}
fl_psu_kill() {  # <pid>...: our own read subshells' process groups, by process group id (fl_psu_bg): TERM (sudo passes it on to a root ipmitool), up to 3 s, then KILL
  local p g=() t
  for p; do fl_psu_live "$p" && { g+=("$p"); kill -TERM -- "-$p" 2> /dev/null; }; done
  t=$(( $(date +%s) + 3 ))
  for p in "${g[@]}"; do while fl_psu_live "$p" && [ "$(date +%s)" -lt "$t" ]; do sleep 0.2; done; kill -KILL -- "-$p" 2> /dev/null; done
  return 0
}
fl_psu_join() {  # $1 1: wait (bounded) for every pending read; 0: apply only those done, in order
  local e pid f w t c9 x
  while [ ${#FL_PSU_BG[@]} -gt 0 ]; do
    e=${FL_PSU_BG[0]}; pid=${e%%|*}; f=${e#*|}; w=${f#*|}; f=${f%%|*}
    if [ ! -e "$f.done" ]; then
      [ "$1" = 1 ] || return 0
      t=$(( $(date +%s) + ${PSU_JOIN_S:-75} ))
      while [ ! -e "$f.done" ] && kill -0 "$pid" 2> /dev/null && [ "$(date +%s)" -lt "$t" ]; do sleep 1; done
      if [ ! -e "$f.done" ]; then fl_psu_kill "$pid"   # our own read's whole process group: subshell, timeout and ipmitool
        echo "PSU WARN PSU check not run: the read did not end in ${PSU_JOIN_S:-75} s ($w)" > "$f.out"; fi
    fi
    wait "$pid" 2> /dev/null; FL_PSU_BG=("${FL_PSU_BG[@]:1}")
    c9=$(awk '$1 == "C9" { t = $2 } END { print t }' "$FL_OUT/phases.txt" 2> /dev/null); x=
    [ -n "$c9" ] && [ -e "$f.done" ] && [ "$(stat -c %Y "$f.done" 2> /dev/null || echo 0)" -ge "$c9" ] && x=" (read ended after C9 started: watts may include the C9 glx_reset)"
    fl_psu_apply "$w" "$x" < "$f.out"
  done
  return 0
}
fl_psu_reap() {  # run.sh's EXIT trap: our own pending reads' whole process groups (subshell, timeout, ipmitool), by process group id
  local e p=(); for e in "${FL_PSU_BG[@]}"; do p+=("${e%%|*}"); done
  [ ${#p[@]} = 0 ] || fl_psu_kill "${p[@]}"; FL_PSU_BG=(); return 0
}
fl_psu_where() { [ "$FL_PSU_AT" = "before load" ] && echo "no load ran" || echo "load stopped $FL_PSU_AT"; }   # for the RESULT and tray lines
