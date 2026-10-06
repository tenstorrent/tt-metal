#!/bin/bash
# Queue ONE project job on blx03's broker (one-project-device-job rule).
# Usage, on blx03: tmp/blx03/submit.sh [--dry-run] <timeout_s> <command...>
# Env comes from tmp/blx03/env.yaml (a detached caller has no venv; the broker default path is wrong).
# Prints the broker job id and records it in $STATE so a driver cannot resubmit while it is live.
# Exit 75 = another project job is active; 64 = refused (bare 2x4 mesh open); 1 = broker gave no id.
dry=0; [ "$1" = "--dry-run" ] && { dry=1; shift; }
t=$1; shift
cd /home/smarton/fasth3/tt-metal || exit 1
STATE=tmp/blx03/.last_job

# Any smarton job running or queued blocks a new one.
active=$(tt-device-mcp status 1 2>&1 | sed -n "/^RUNNING/,/^RECENT/p" | grep -w smarton)
if [ -n "$active" ]; then echo "busy: $active"; exit 75; fi
# The last recorded job blocks too, in case the overview missed it.
if [ -s "$STATE" ]; then
  last=$(cat "$STATE")
  if tt-device-mcp status -j "$last" 2>&1 | grep -qiE "^Status: *(running|queued|pending)"; then
    echo "busy: job $last from $STATE is still active"; exit 75
  fi
fi

# A bare (2,4) mesh open on a BH galaxy failed fabric init and rebooted blx03 (job 989).
# Scan the command and any files it names, one level deep through shell scripts.
files=()
for a in "$@" $*; do [ -f "$a" ] && files+=("$a"); done
for f in "${files[@]}"; do
  case "$f" in *.sh) for g in $(grep -oE '[^ "'"'"'=]+\.(py|sh)' "$f"); do [ -f "$g" ] && files+=("$g"); done ;; esac
done
bare='mesh_device["'"'"']?[^\n]{0,8}\(\s*2\s*,\s*4\s*\)|open_mesh_device\([^)]*MeshShape\(\s*2\s*,\s*4\s*\)|MESH_DEVICE=?["'"'"']?\(?2,\s*4'
hit=$( { echo "$*" | grep -nE "$bare" | sed 's/^/cmd:/'; [ ${#files[@]} -gt 0 ] && grep -nHE "$bare" "${files[@]}"; } 2>/dev/null | sort -u)
if [ -n "$hit" ]; then
  echo "refused: bare 2x4 mesh open. Open the full mesh and use create_submesh(ttnn.MeshShape(2, 4)) instead."
  echo "$hit"; exit 64
fi

if [ $dry = 1 ]; then echo "dry-run: would run: tt-device-mcp run-bg \"$*\" -w $PWD -e tmp/blx03/env.yaml -t $t"; exit 0; fi
out=$(tt-device-mcp run-bg "$*" -w "$PWD" -e tmp/blx03/env.yaml -t "$t" 2>&1); echo "$out"
id=$(echo "$out" | sed -nE 's/^Job ([0-9]+) queued.*/\1/p' | head -1)
[ -n "$id" ] || { echo "no job id in broker output"; exit 1; }
echo "$id" > "$STATE"; echo "$id"
