#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/pgroups.sh: reap the process groups run.sh's setsid children start. The tt-device-mcp broker
# preempts a job with TERM to its process group, then KILL 10 s later; a setsid child has left that group, so without
# this it would keep its device open after run.sh is gone. Each setsid launch records "pgid starttime" in
# $TRAY_PGID_FILE (pg_track; ub0_repro.py's snapshot writes the same line to the file its environment names: under sudo
# that is $TRAY_PGID_USER_FILE, the sudo caller's own); run.sh's TERM/INT/HUP and EXIT traps call pg_reap: TERM every
# live recorded group, wait at most 5 s, KILL what is left. Functions only.
# pg_reap may run as root, so it trusts neither file blindly. It reads a file only when it is a regular
# file (no symlink) of root, this shell's user or $SUDO_USER that only its owner may write, and signals a group only when
# run.sh started it: a real start time no older than run.sh's own, the leader (if alive) still with that start time, and
# every member owned by those users and either a descendant of run.sh or, not root's, started no earlier than run.sh.
# Never pgid <= 1 or run.sh's own. Anything else in the files is left alone.
#   pg_track <pid>   record the group led by <pid> (a setsid child: pid = pgid); no-op without TRAY_PGID_FILE
#   pg_reap [exit]   TERM, wait <= 5 s, KILL each live recorded group; "exit" also removes the files (PG_SOFT=1: no KILL)
#   pg_traps         TERM/INT/HUP: pg_reap, then exit 143/130/129 (the EXIT trap then runs as before); PG_DYING=TERM/INT/HUP
#   pg_run <timeout ...>  a device call under GNU timeout: timeout setpgid()s itself out of the job's group,
#                    so the broker's TERM never reaches it. Run it in the background (stdin kept), record its group, wait:
#                    the trap then fires at once and reaps it. In a subshell ($(...), a pipeline) it sets
#                    its own TERM trap for that group, or the parent would wait on the pipe for the whole timeout. Returns its exit code.

# PG_OUT: the output of the shell that sources this file, kept so a TERM inside a redirected call (pg_run ...
# > file) still says in the run's own output that it is preempted. Every child closes it ({PG_OUT}>&-):
# pg_run's call and each setsid launch, so none holds the run's output open after it.
[ -n "${PG_OUT:-}" ] || exec {PG_OUT}>&1

pg_start() { local s; s=$(cat "/proc/$1/stat" 2>/dev/null) || return 1; s=${s##*) }; set -- $s; echo "${20}"; }

pg_track() {
  [ -n "${TRAY_PGID_FILE:-}" ] && [ -n "${1:-}" ] || return 0
  echo "$1 $(pg_start "$1" || echo 0)" >> "$TRAY_PGID_FILE" 2>/dev/null
  return 0
}

pg_trusted() {  # <file> <fd> <" uid ... ">: fd is the path's own inode, a regular file (no symlink) owned by one of those
  # uids that no one else may write
  local m; m=$(stat -c '%d:%i %u %a' -- "$1" 2>/dev/null) || return 1
  set -- "$1" "$2" "$3" $m
  [ ! -L "$1" ] && [ $# = 6 ] && [ "$4" = "$(stat -L -c '%d:%i' "/dev/fd/$2" 2>/dev/null)" ] && [[ $3 == *" $5 "* ]] \
    && (( (8#$6 & 8#022) == 0 ))
}

pg_snap() { local p q; PG_PS=$(ps -A -o pid=,pgid=,ppid=,uid=,ruid= 2>/dev/null); PG_PP=(); while read -r p _ q _; do [ -n "$p" ] && PG_PP[$p]=$q; done <<< "$PG_PS"; }

pg_ours() {  # <pgid>: the group has members (pg_reap's PG_PS), each of root, us or the sudo caller (real and effective
  # uid), and each a descendant of run.sh ($$) or, not root's, started at or after run.sh (PG_T0)
  local p pg pp u r n=0 st q h
  while read -r p pg pp u r; do
    [ "$pg" = "$1" ] || continue
    [[ $PG_OK == *" $u "* && $PG_OK == *" $r "* ]] || return 1
    q=$pp; h=0; while [ -n "$q" ] && [ "$q" -gt 1 ] && [ "$q" != $$ ] && [ $h -lt 64 ]; do q=${PG_PP[$q]:-}; h=$((h + 1)); done
    if [ "$q" != $$ ]; then
      [[ $PG_USR == *" $u "* && $PG_USR == *" $r "* ]] || return 1
      st=$(pg_start "$p") && [ "$st" -ge "$PG_T0" ] || return 1
    fi
    n=$((n + 1))
  done <<< "$PG_PS"
  [ $n -gt 0 ]
}

pg_reap() {
  local PG_OK PG_USR PG_T0 PG_PS f fd g s now u live=() i; local -A PG_PP
  u=""; [ -n "${SUDO_USER:-}" ] && u=$(id -u "$SUDO_USER" 2>/dev/null)
  PG_OK=" 0 $(id -u) ${u:+$u }"; PG_USR=" "; for i in $(id -u) $u; do [ "$i" = 0 ] || PG_USR="$PG_USR$i "; done
  PG_T0=$(pg_start $$) || PG_T0=0
  for f in "${TRAY_PGID_FILE:-}" "${TRAY_PGID_USER_FILE:-}"; do
    [ -n "$f" ] && [ -f "$f" ] || continue
    exec {fd}< "$f" || continue
    if ! pg_trusted "$f" "$fd" "$PG_OK"; then
      exec {fd}<&-; echo "pg_reap: $f is not a regular file of root or the run's user, writable only by its owner: signalling none of its groups" >&2
      continue
    fi
    [ -n "${PG_PS:-}" ] || pg_snap
    while read -r g s; do
      [[ $g =~ ^[0-9]+$ ]] && [ "$g" -gt 1 ] && [ "$g" != "$$" ] || continue
      # no start time read (0), or older than run.sh: not a group this run started
      [[ $s =~ ^[0-9]+$ ]] && [ "$s" -gt 0 ] && [ "$s" -ge "$PG_T0" ] || continue
      kill -0 -- "-$g" 2>/dev/null || continue
      # the leader still alive with another start time: its pid was reused, not our group
      if now=$(pg_start "$g"); then [ "$now" = "$s" ] || continue; fi
      pg_ours "$g" || continue
      live+=("$g")
    done <&$fd
    exec {fd}<&-
    [ "${1:-}" = exit ] && rm -f "$f"
  done
  [ ${#live[@]} = 0 ] && return 0
  for g in "${live[@]}"; do kill -TERM -- "-$g" 2>/dev/null; done
  # PG_DEADLINE_US (microseconds since the epoch; run.sh's rs_dying): wait for the groups only until then, so a
  # TERM path's whole clean-up stays inside one bound (the broker KILLs 10 s after its TERM); unset: 5 s, as before
  local n=10; [ -n "${PG_DEADLINE_US:-}" ] && n=$(( (PG_DEADLINE_US - ${EPOCHREALTIME/[.,]/}) / 500000 )) && [ "$n" -gt 10 ] && n=10
  # PG_SOFT=1 (rs_dying's first call): TERM and wait only, no KILL; the groups that end say so, so a stage's own
  # device close shows before run.sh's RESULT line. The second, plain call TERMs what is left again and KILLs at its deadline
  local t0=${EPOCHREALTIME/[.,]/} all=("${live[@]}")
  for (( i = 0; i < n; i++ )); do
    now=()
    for g in "${live[@]}"; do kill -0 -- "-$g" 2>/dev/null && now+=("$g"); done
    live=("${now[@]}")
    if [ ${#live[@]} = 0 ]; then
      [ -n "${PG_SOFT:-}" ] && echo "pg_reap: process group(s) ${all[*]} ended within $(( (${EPOCHREALTIME/[.,]/} - t0) / 100000 / 10 )).$(( (${EPOCHREALTIME/[.,]/} - t0) / 100000 % 10 )) s of TERM" >&2
      return 0
    fi
    sleep 0.5
  done
  [ -n "${PG_SOFT:-}" ] && return 0
  pg_snap; now=()   # checked again: a group gone meanwhile may be another one now
  for g in "${live[@]}"; do pg_ours "$g" && { kill -KILL -- "-$g" 2>/dev/null; now+=("$g"); }; done
  [ ${#now[@]} = 0 ] || echo "pg_reap: KILLed process group(s) ${now[*]} still running $(( (n > 0 ? n : 0) * 5 / 10 )).$(( (n > 0 ? n : 0) * 5 % 10 )) s after TERM" >&2
  return 0
}

pg_run() {
  "$@" 0<&0 {PG_OUT}>&- &
  local p=$! rc; pg_track "$p"
  [ "${BASHPID:-$$}" = "$$" ] && { wait "$p"; return; }
  # a subshell: a TERM reaps this call's group only (our own kill of a poll subshell must not end the run's other groups)
  trap "kill -TERM -- -$p 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 -- -$p 2>/dev/null || exit 143; sleep 0.5; done; kill -KILL -- -$p 2>/dev/null; exit 143" TERM
  wait "$p"; rc=$?; trap - TERM; return $rc   # not the saved $(trap -p): in a subshell that prints run.sh's reap-all trap
}

pg_traps() {  # PG_DYING (the signal): the EXIT trap's fault capture then opens no chip and skips the slow exit snapshot (lib/fault_capture.sh)
  # PIPE ignored first: with the output pipe gone, a write fails with EPIPE, never a SIGPIPE death before the reap
  trap 'PG_DYING=TERM; trap "" PIPE; pg_reap; exit 143' TERM; trap 'PG_DYING=INT; trap "" PIPE; pg_reap; exit 130' INT
  trap 'PG_DYING=HUP; trap "" PIPE; pg_reap; exit 129' HUP
}
