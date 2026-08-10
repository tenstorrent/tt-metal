#!/usr/bin/env bash
# Restore the refs unnamed for advchal-v3 cell isolation. See parked-refs.txt (sha refname per line).
#
# Unnaming a ref does not delete objects, and this clone is configured gc.pruneExpire=never +
# gc.auto=0 so they cannot be reaped. So this restores every ref exactly, offline.
set -euo pipefail
cd "$(dirname "$0")"
while read -r sha ref; do
  [ -n "$sha" ] || continue
  git -C "${TT_METAL_HOME:-/home/mvasiljevic/tt-metal}" update-ref "$ref" "$sha"
done < parked-refs.txt
echo "restored $(wc -l < parked-refs.txt) refs"
