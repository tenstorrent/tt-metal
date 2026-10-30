#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Start a cachew session with whichever Node.js this job can run. Composite
# action run steps do not get the runner's bundled Node on PATH, so also look
# where the runner keeps it:
#   * host jobs: <runner root>/externals/node{24,20}/bin/node, where RUNNER_TEMP
#     is <runner root>/_work/_temp (tt CI VMs: /opt/runner);
#   * job containers: the runner mounts the same externals at /__e.
# Without a usable Node, or on any error, report ready=false and leave the job
# untouched so the plain checkout runs exactly as actions/checkout would.
set -u

not_ready() {
  echo "$1; using standard checkout only."
  [ -n "${GITHUB_OUTPUT:-}" ] && echo "ready=false" >>"$GITHUB_OUTPUT"
  exit 0
}

candidates=()
if path_node=$(command -v node 2>/dev/null); then candidates+=("$path_node"); fi
if [ -n "${RUNNER_TEMP:-}" ]; then
  runner_root=$(cd "$RUNNER_TEMP/../.." 2>/dev/null && pwd -P) || runner_root=""
  if [ -n "$runner_root" ]; then
    candidates+=("$runner_root/externals/node24/bin/node" "$runner_root/externals/node20/bin/node")
  fi
fi
candidates+=("/__e/node24/bin/node" "/__e/node20/bin/node")

node_bin=""
for candidate in "${candidates[@]}"; do
  [ -x "$candidate" ] || continue
  # Rejects binaries that cannot run here (e.g. glibc Node in an Alpine container) and old Node.
  if "$candidate" -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 18 ? 0 : 1)' >/dev/null 2>&1; then
    node_bin=$candidate
    break
  fi
done
[ -n "$node_bin" ] || not_ready "Node.js 18+ unavailable"

"$node_bin" "$(dirname "${BASH_SOURCE[0]}")/index.js" prepare || not_ready "Cachew session could not start"
