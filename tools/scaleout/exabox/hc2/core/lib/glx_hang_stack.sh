#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# lib/glx_hang_stack.sh LOG N0 TIMEOUT_S CMD...: runs CMD (the post-power-virus tt-smi -glx_reset, under GNU timeout
# TIMEOUT_S) and passes its exit code through. If CMD still runs near the end of its timeout after its log (LOG past line
# N0) says 'All N chips found' (the hang in POST_RESET or the re-init, tt-umd issue #3912), it first writes into LOG the
# KMD version, each tt-smi thread's kernel wait channel and a native backtrace of the tt-smi pid (eu-stack -p, else
# gdb -batch -p; no sudo), then leaves CMD to the timeout as before: the reset is never shortened or stopped here.
# Under kernel.yama.ptrace_scope 1 only an ancestor may attach. So a helper subshell H starts a waiter P, P starts CMD
# and keeps its exit code, and H later execs the stack tool itself: the tool is then tt-smi's grandparent, and P still
# collects tt-smi's exit code, so a tt-smi that ends on its own after its stack keeps its real exit code. Under sudo the
# stack is of sudo's tt-smi child, and the log says so. The capture starts TIMEOUT_S/4 (at most 45 s) before the
# timeout and its tool gets half that long; a tool still running then is stopped by its pid. On the timeout's TERM it
# passes TERM to CMD and waits for CMD to end (at most TRAY_HANG_STACK_TERM_WAIT_S, default 10 s) before it exits 143,
# so a retry never starts beside a dying tt-smi.
# TRAY_STACK_TOOLS (default "eu-stack gdb") and TRAY_KMD_VERSION_FILE (default /sys/module/tenstorrent/version): tests.
set -u
log=$1 n0=$2 t=$3; shift 3
tool=""; for x in ${TRAY_STACK_TOOLS:-eu-stack gdb}; do command -v "$x" > /dev/null 2>&1 && { tool=$x; break; }; done
d=$(mktemp -d) || exit 125
# H: on USR1 it becomes the stack tool on the pid in $d/target (exec keeps its pid: the tool is an ancestor of tt-smi)
# P: tt-smi's parent; writes tt-smi's pid to $d/pid and its exit code to $d/rc
( trap 'g=$(cat "$d/target"); if [ "$tool" = gdb ]; then exec gdb -batch -nx -p "$g" -ex "thread apply all bt"; else exec "$tool" -p "$g"; fi' USR1
  ( echo "$BASHPID" > "$d/p"; "$@" < /dev/null & echo "$!" > "$d/pid"; wait "$!"; r=$?; echo "$r" > "$d/rc"; exit "$r" ) & wait "$!" ) & h=$!
now_ds() { local e=${EPOCHREALTIME/[.,]/}; echo $(( e / 100000 )); }
# shellcheck disable=SC2329  # on_term runs from the TERM trap
on_term() {  # the timeout's TERM: pass it to tt-smi and wait (bounded) for it to end before exiting
  local c e; c=$(cat "$d/pid" 2> /dev/null); e=$(( $(now_ds) + ${TRAY_HANG_STACK_TERM_WAIT_S:-10} * 10 ))
  [ -n "$c" ] && kill -TERM "$c" 2> /dev/null
  while [ -n "$c" ] && kill -0 "$c" 2> /dev/null && [ "$(now_ds)" -lt "$e" ]; do sleep 0.1; done
  exit 143
}
trap 'rm -rf "${d:?}"' EXIT; trap on_term TERM
t0=$(now_ds); lead=$(( t * 10 / 4 < 450 ? t * 10 / 4 : 450 )); cap=$(( t0 + t * 10 - lead )); took=""
while kill -0 "$h" 2> /dev/null; do
  if [ -z "$took" ] && [ "$(now_ds)" -ge "$cap" ]; then
    took=1; s=$(cat "$d/pid" 2> /dev/null)
    if [ -n "$s" ] && [ -d "/proc/$s" ] && tail -n +$((n0 + 1)) "$log" | grep -aqE 'All [0-9]+ chips found'; then
      stage="POST_RESET"; tail -n +$((n0 + 1)) "$log" | grep -aq 'Re-initializing boards' && stage="the re-init ('Re-initializing boards')"
      g=$s
      echo
      echo "=== tt-smi pid $s still running $(( ($(now_ds) - t0) / 10 )) s into its $t s timeout after all its chips came back, in $stage: its state before the timeout stops it"
      if [ "$(cat "/proc/$s/comm" 2> /dev/null)" = sudo ]; then
        g=$(cat /proc/"$s"/task/*/children 2> /dev/null | awk '{print $1; exit}')
        echo "pid $s is sudo; the state below is of its child pid ${g:-(none found)}, the tt-smi it runs"
      fi
      echo "KMD version: $(cat "${TRAY_KMD_VERSION_FILE:-/sys/module/tenstorrent/version}" 2> /dev/null || echo unreadable)"
      echo "kernel.yama.ptrace_scope: $(cat /proc/sys/kernel/yama/ptrace_scope 2> /dev/null || echo unreadable)"
      u=""; [ -n "$g" ] && u=$(awk '/^Uid:/ {print $2}' "/proc/$g/status" 2> /dev/null)
      for x in /proc/"${g:-none}"/task/*; do
        [ -d "$x" ] && echo "thread ${x##*/} ($(cat "$x/comm" 2> /dev/null)): kernel wait channel $(cat "$x/wchan" 2> /dev/null || echo unreadable)"
      done
      if [ -z "$g" ]; then echo "native stack: not taken: sudo's tt-smi child was not found"
      elif [ ! -d "/proc/$g" ]; then echo "native stack: not taken: tt-smi ended during the capture"  # before its Uid read
      elif [ "$u" != "$(id -u)" ]; then echo "native stack: not taken: pid $g runs as uid ${u:-?} (sudo), and the stack is taken without sudo"
      elif [ -z "$tool" ]; then echo "native stack: not taken: neither eu-stack nor gdb is installed"
      else
        echo "native stack ($tool -p $g, run as tt-smi's ancestor):"
        echo "$g" > "$d/target"; kill -USR1 "$h"; b=$(( lead / 2 )); e=$(( $(now_ds) + b ))
        while kill -0 "$h" 2> /dev/null && [ "$(now_ds)" -lt "$e" ]; do sleep 0.2; done
        kill -0 "$h" 2> /dev/null && { kill -TERM "$h" 2> /dev/null; echo "native stack: $tool did not end in $(( b / 10 )).$(( b % 10 )) s; stopped"; }
        wait "$h" 2> /dev/null
        echo "=== end of native stack"
        # H is gone; P still waits for tt-smi (the timeout stops it) and writes its exit code
        p=$(cat "$d/p" 2> /dev/null)
        while [ ! -s "$d/rc" ] && [ -n "$p" ] && kill -0 "$p" 2> /dev/null; do sleep 0.2; done
        [ -s "$d/rc" ] && exit "$(cat "$d/rc")"
        echo "tt-smi pid $s ended without its exit code being read; counted as failed"; exit 1
      fi
      echo "=== end of tt-smi state"
    fi
  fi
  sleep 0.2
done
wait "$h"; exit
