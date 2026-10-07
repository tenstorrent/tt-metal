#!/usr/bin/env bash
# usage: mk_worktree.sh <track>   -> worktree /mnt/tt-data/ssinghal/mimo/wt/<track>, branch ssinghal/mimo-<track>, from ssinghal/mimo
# Links the ignored build artifacts of the MiMo clone (symlinks, never copies). Only writes under its own worktree.
set -euo pipefail
T=$1; MAIN=/mnt/tt-data/ssinghal/mimo/tt-metal; WT=/mnt/tt-data/ssinghal/mimo/wt/$T; BR=ssinghal/mimo-$T
[ "$(git -C $MAIN rev-parse --show-toplevel)" = "$MAIN" ] || { echo "bad main"; exit 1; }
git -C $MAIN worktree add "$WT" -b "$BR" ssinghal/mimo
cd "$WT"; [ "$(pwd)" = "$WT" ] || exit 1
for p in runtime build build_Release python_env .cpmcache tt_metal/pre-compiled ttnn/ttnn.egg-info ttnn/ttnn/_ttnn.so tt_metal/llrt/hal/generated tools/triage/inspector_capnp.pyi; do
  [ -e "$MAIN/$p" ] && { mkdir -p "$(dirname "$p")"; [ -e "$p" ] || ln -s "$MAIN/$p" "$p"; }
done
mkdir -p generated
# .gitignore patterns have trailing slashes (dirs only), so the symlinks look untracked and launch.py/log_tool.py refuse a "dirty" tree
EX=$(git rev-parse --git-path info/exclude); mkdir -p "$(dirname "$EX")"
for p in runtime build build_Release python_env .cpmcache tt_metal/pre-compiled ttnn/ttnn.egg-info ttnn/ttnn/_ttnn.so tt_metal/llrt/hal/generated tools/triage/inspector_capnp.pyi generated; do echo "/$p" >> "$EX"; done
echo "worktree ready: $WT ($BR)"; ls -l build python_env ttnn/ttnn/_ttnn.so | head
