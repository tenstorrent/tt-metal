#!/bin/bash
# Build the megakernel release-candidate tt-metal branch LOCALLY (no push). One committed SHA, no overlay.
#   tt-metal RC = anatarajan/qwen38-r16-mk-on (vLLM engine-ON integration, which merges anatarajan/qwen38-r16-mk-int)
#               + 8c2faf2ec95 (QWEN36_MM_VISION_CACHE; contains c8c90fe QWEN36_MM_MAX_IMAGES)
#               + anatarajan/qwen38-r16-mk-int (no-op when -mk-on is current with it)
# Usage: make_rc.sh <tt-metal checkout> [rc branch name]   -> prints the RC SHA. Stops on any merge conflict.
set -euo pipefail
REPO=$1; RC=${2:-anatarajan/qwen38-r16-mk-rc}
INT=anatarajan/qwen38-r16-mk-int; INTEG=${INTEG:-anatarajan/qwen38-r16-mk-on}; VCACHE=8c2faf2ec95
cd "$REPO"
WT=$(mktemp -d "${TMPDIR:-/tmp}/mk-rc.XXXXXX")
git worktree add -q --detach "$WT" "$INTEG"
cd "$WT"
G=git
$G merge --no-edit -q "$VCACHE"
$G merge --no-edit -q "$INT"
for c in c8c90fe4acb 8eb9af536d8; do git merge-base --is-ancestor $c HEAD || { echo "missing $c"; exit 1; }; done
# python-only vs prod r13 outside the qwen36 model dir (no C++ change rides along unnoticed)
git diff --quiet 8eb9af536d8 HEAD -- . ':!models/demos/blackhole/qwen36' || { echo "WARNING: non-qwen36 changes vs r13:"; git diff --stat 8eb9af536d8 HEAD -- . ':!models/demos/blackhole/qwen36' | tail -5; }
git branch -f "$RC" HEAD
SHA=$(git rev-parse HEAD)
cd "$REPO" && git worktree remove --force "$WT"
echo "RC $RC = $SHA"
