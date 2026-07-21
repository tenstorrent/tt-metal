#!/bin/bash
# Reap orphaned remote Zebu emulator jobs on the emu host.
#
# A run's emu job is cleanly released by the graceful tt-exalens `exit` a
# finishing pytest sends. A run whose local peer dies non-gracefully (hard kill,
# harness background-wait termination, crash) never sends it, so the remote
# `make ... test_umd_remote` job holds its Zebu slot until EMULATOR_TIMEOUT
# (1200s) — congesting the shared host. This script kills such orphans directly.
# It is the standalone/cron counterpart to run_test.sh's in-run reaper.
#
# Safe by default: reaps only when the single global emulator flock is FREE (no
# live run_test.sh sim), so a running sim is never disturbed. Pass --force to
# reap regardless — e.g. from a batch trap after its child runs are killed.
#
# Usage:
#   reap_stale_emu.sh [--arch quasar] [--emu-host soc-l-12] [--force]
set -u

ARCH="quasar"
EMU_HOST="${EMU_HOST:-${SSH_MACHINE_NAME:-soc-l-12}}"
FORCE="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)     ARCH="$2";     shift 2 ;;
    --emu-host) EMU_HOST="$2"; shift 2 ;;
    --force)    FORCE="true";  shift   ;;
    -h|--help)  grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Single global lock shared by every run_test.sh invocation on the host (all arches).
LOCKFILE="/tmp/tt-llk-test.lock"

# Unless forced, only reap when no local sim holds the emulator lock — a live run
# owns the emulator and its own teardown releases it.
if [[ "$FORCE" != "true" ]]; then
  exec 9>>"$LOCKFILE" 2>/dev/null || { echo "[reap] cannot open $LOCKFILE" >&2; exit 1; }
  if ! flock -n 9; then
    echo "[reap] a local sim holds $LOCKFILE — skipping (its own teardown will release the emulator)"
    exit 0
  fi
  trap 'flock -u 9 2>/dev/null || true' EXIT
fi

# Remote script: kill each emu make's whole process group, then hard-kill any
# straggler zrun/vovsh. A detached VOV farm job that survives still falls back to
# EMULATOR_TIMEOUT. $USER/$found/$p/$g are evaluated on the remote (single-quoted).
remote_cmd='
found=$(pgrep -u "$USER" -f "make -C verification/emu test" 2>/dev/null)
for p in $found; do
  g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d " ")
  [ -n "$g" ] && kill -TERM -"$g" 2>/dev/null
done
sleep 2
# Broad catch: the sh-recipe/zrun/vovsh/tee children detach (setsid to the VOV
# farm) and escape the make process group, so match the shared "test_umd_remote"
# marker in their cmdlines to reap the whole tree.
pkill -9 -u "$USER" -f "test_umd_remote" 2>/dev/null
pkill -9 -u "$USER" -f "make -C verification/emu test" 2>/dev/null
n=$(printf "%s\n" "$found" | grep -c . 2>/dev/null); n=${n:-0}
echo "[reap] $(hostname): killed $n emu make job(s)"
true'

# Feed over stdin to bash -s: the ssh command-argument form is re-parsed by the
# remote login shell and fails here (exit 255); stdin goes straight to bash.
printf '%s' "$remote_cmd" | timeout 30 ssh "$EMU_HOST" -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes 'bash -s' \
  || { echo "[reap] ssh to $EMU_HOST failed/timed out" >&2; exit 1; }
