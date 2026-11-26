#!/bin/bash
#SBATCH --job-name=ttmetal-build
#SBATCH --nodes=1
#SBATCH --time=01:00:00
# Fetch and build tt-metal on a build node (never on the login node). Submit with build-metal, which sets
# the partition, the excluded nodes and the log path (see env.sh).
# Optional env: BRANCH=<branch to check out first>. Without it, builds the branch already checked out.
set -eo pipefail
. "$XRUN_REPO/scripts/exabox/env.sh"
export XRUN_BYPASS=1
echo "node=$(hostname) cpus=$(nproc) start=$(date -u +%FT%TZ)"
# tt-metal writes compiler-probe temp files to the node-local /tmp (tmpfile() ignores TMPDIR); a full /tmp breaks the firmware step.
free_gb=$(df --output=avail -BG /tmp | tail -1 | tr -dc 0-9)
echo "free /tmp: ${free_gb} GB"
[ "$free_gb" -ge 5 ] || { echo "ABORT: only ${free_gb} GB free in /tmp on $(hostname); add it to XRUN_BUILD_EXCLUDE and resubmit"; exit 3; }
export TMPDIR="$XRUN_DATA/tmp" CCACHE_DIR="$XRUN_DATA/.ccache"
mkdir -p "$TMPDIR"
cd "$XRUN_REPO"
# tt_ops_code_gen is private and pinned (update=none); build nodes have no GitHub login, so never fetch submodules here.
git fetch --prune --recurse-submodules=no origin
if [ -n "${BRANCH:-}" ]; then
  git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" --track "origin/$BRANCH"
fi
git merge --ff-only "@{u}" || echo "WARN: not fast-forwarding (local commits or no upstream)"
git submodule update --init --recursive --jobs 8
echo "building $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
./build_metal.sh -c --toolchain-path "$XRUN_TOOLCHAIN" ${BUILD_ARGS:-}
echo "BUILD_DONE $(date -u +%FT%TZ)"
