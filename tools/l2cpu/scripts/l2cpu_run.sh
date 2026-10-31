#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
#
# Usage: l2cpu_run.sh "<reason>" <command...>
# One L2CPU epoch = one lock hold: reset the chips, then run the command (which opens the device, releases the
# harts once, does its work and exits). The harts of a tile leave reset once per chip reset, so every run that
# starts the firmware needs its own reset; holding the lock across reset + run keeps other users of the machine
# from interleaving.
#   L2CPU_LOCK        lock file (default /tmp/l2cpu.lock; use the same file as every other device user)
#   L2CPU_RESET_CMD   reset command (default "tt-smi -r"; e.g. "tt-smi -r 0,1,2,3" for multi-chip boards
#                     that must be reset together)
#   L2CPU_RESET_LOG   optional file: one line per reset "<time> | <reason>"
# The reset runs without TT_VISIBLE_DEVICES (it addresses the board, not this process's view of it); the command
# itself runs with the caller's environment.
set -u
reason=${1:?reason}; shift
lock=${L2CPU_LOCK:-/tmp/l2cpu.lock}
reset_cmd=${L2CPU_RESET_CMD:-tt-smi -r}
exec flock "$lock" bash -c '
  if [ -n "${L2CPU_RESET_LOG:-}" ]; then echo "$(date "+%Y-%m-%d %H:%M:%S %Z") | '"$reason"'" >> "$L2CPU_RESET_LOG"; fi
  env -u TT_VISIBLE_DEVICES '"$reset_cmd"' >/dev/null 2>&1 || { echo "l2cpu_run: reset failed ('"$reset_cmd"')" >&2; exit 97; }
  exec "$@"' l2cpu_run "$@"
