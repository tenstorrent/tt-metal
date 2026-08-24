#!/bin/bash
# Reap orphaned remote Aether VCS/Zebu jobs on the simulation host.
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
# Scope: by default reaps ONLY this run's own job, identified by --tag (defaults
# to $NNG_SOCKET_NAME). Pass --all for a fleet-wide sweep of everything of ours
# (cron/batch cleanup).
#
# Usage:
#   reap_stale_emu.sh [--arch quasar] [--emu-host soc-l-12]
#                     [--lock /shared/quasar-aether.lock]
#                     [--tag <NNG_SOCKET_NAME>] [--all] [--force]
set -u

ARCH="quasar"
EMU_HOST="${EMU_HOST:-${QSR_AETHER_HOST:-${SSH_MACHINE_NAME:-soc-l-12}}}"
LOCKFILE="${QSR_AETHER_LOCK:-/tmp/tt-llk-test.lock}"
FORCE="false"
TAG="${NNG_SOCKET_NAME:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)     ARCH="$2";     shift 2 ;;
    --emu-host) EMU_HOST="$2"; shift 2 ;;
    --lock)     LOCKFILE="$2"; shift 2 ;;
    --tag)      TAG="$2";      shift 2 ;;
    --all)      TAG="";        shift   ;;
    --force)    FORCE="true";  shift   ;;
    -h|--help)  grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Unless forced, only reap when no sim holds the configured Aether lock — a live
# run owns the resource and its own teardown releases it. In production this is
# a shared-filesystem lock, so the check covers every compute runner.
if [[ "$FORCE" != "true" ]]; then
  mkdir -p "$(dirname "$LOCKFILE")" 2>/dev/null ||
    { echo "[reap] cannot create lock directory for $LOCKFILE" >&2; exit 1; }
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
if [[ -n "$TAG" ]]; then
  # Tag-scoped reap, required once more than one slot runs under this account:
  # the untagged form below is a blanket `pkill -u $USER` and would kill a peer
  # slot's healthy job. The tag is NNG_SOCKET_NAME, which the sim launcher passes
  # as `make ... NAME=<tag>`; detached zrun/tee children carry it in their run
  # directory path instead.
  remote_cmd="
tag=$(printf '%q' "$TAG")
found=\$(pgrep -u \"\$USER\" -f \"make -C verification(/emu)? (sim-test|test) .*NAME=\${tag}( |\\\$)\" 2>/dev/null)
for p in \$found; do
  g=\$(ps -o pgid= -p \"\$p\" 2>/dev/null | tr -d ' ')
  [ -n \"\$g\" ] && kill -TERM -\"\$g\" 2>/dev/null
done
sleep 2
pkill -9 -u \"\$USER\" -f \"make -C verification(/emu)? (sim-test|test) .*NAME=\${tag}( |\\\$)\" 2>/dev/null
pkill -9 -u \"\$USER\" -f \"(test_umd_remote|umd_remote_test)_[^/]*_\${tag}/\" 2>/dev/null
n=\$(printf '%s\\n' \"\$found\" | grep -c . 2>/dev/null); n=\${n:-0}
echo \"[reap] \$(hostname): killed \$n Aether make job(s) for tag \${tag}\"
true"
else
remote_cmd='
found=$(pgrep -u "$USER" -f "make -C verification(/emu)? (sim-test|test)" 2>/dev/null)
for p in $found; do
  g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d " ")
  [ -n "$g" ] && kill -TERM -"$g" 2>/dev/null
done
sleep 2
# Broad catch: the sh-recipe/zrun/vovsh/tee children detach (setsid to the VOV
# farm) and escape the make process group. Emulator uses test_umd_remote while
# VCS uses umd_remote_test, so cover both backends.
pkill -9 -u "$USER" -f "test_umd_remote|umd_remote_test" 2>/dev/null
pkill -9 -u "$USER" -f "make -C verification(/emu)? (sim-test|test)" 2>/dev/null
n=$(printf "%s\n" "$found" | grep -c . 2>/dev/null); n=${n:-0}
echo "[reap] $(hostname): killed $n Aether make job(s)"
true'
fi

# Feed over stdin to bash -s: the ssh command-argument form is re-parsed by the
# remote login shell and fails here (exit 255); stdin goes straight to bash.
printf '%s' "$remote_cmd" | timeout 30 ssh "$EMU_HOST" -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes 'bash -s' \
  || { echo "[reap] ssh to $EMU_HOST failed/timed out" >&2; exit 1; }
