#!/usr/bin/env bash
# Off-device kernel prewarm, then submit the real job to the tt-device-mcp broker.
#
# The device is a shared, serialized resource; kernel compilation is not device work. This runs the
# host-side prewarm (tools/kernel_prewarm) BEFORE the broker reservation so the reserved window holds
# the device only for device work. Warm cache => the prewarm is a sub-second dephash no-op; after a
# kernel/header edit => it rebuilds the working set off-device (~20s here) instead of on-device.
#
# One-time floor: the very first run on a fresh cache (no manifest yet) still compiles on-device --
# the manifest is a byproduct of that run and cannot be prewarmed before it exists.
#
#   prewarm_and_submit.sh -e <env.yaml> [-w <workspace>] [-t <timeout_s>] -- <command...>
#
# TT_METAL_CACHE / TT_METAL_HOME are read from the env yaml (or inherited if absent), matching the
# values the broker job will use, so the prewarm warms the exact cache the run consumes.
set -uo pipefail

ENV_FILE=""
WORKSPACE="$PWD"
TIMEOUT=1200

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--env) ENV_FILE="$2"; shift 2;;
    -w|--workspace) WORKSPACE="$2"; shift 2;;
    -t|--timeout) TIMEOUT="$2"; shift 2;;
    --) shift; break;;
    *) echo "prewarm_and_submit.sh: unknown arg $1" >&2; exit 2;;
  esac
done
[[ $# -gt 0 ]] || { echo "usage: prewarm_and_submit.sh -e <env.yaml> [-w dir] [-t sec] -- <command...>" >&2; exit 2; }
CMD="$*"

# A yaml value may be bare or quoted (key: "val" / key: val); strip surrounding quotes and whitespace.
_yaml_get() {
  local key="$1" file="$2"
  [[ -f "$file" ]] || return 0
  sed -nE "s/^${key}:[[:space:]]*[\"']?([^\"']*)[\"']?[[:space:]]*$/\1/p" "$file" | head -n1
}

CACHE="$(_yaml_get TT_METAL_CACHE "$ENV_FILE")"; CACHE="${CACHE:-${TT_METAL_CACHE:-}}"
HOME_DIR="$(_yaml_get TT_METAL_HOME "$ENV_FILE")"; HOME_DIR="${HOME_DIR:-${TT_METAL_HOME:-$WORKSPACE}}"

TOOL="$HOME_DIR/build_Release/tools/kernel_prewarm"
[[ -x "$TOOL" ]] || TOOL="$WORKSPACE/build_Release/tools/kernel_prewarm"

if [[ -x "$TOOL" ]]; then
  echo "== off-device kernel prewarm (cache=${CACHE:-default}) =="
  env TT_METAL_CACHE="$CACHE" TT_METAL_HOME="$HOME_DIR" "$TOOL"
else
  echo "prewarm_and_submit.sh: kernel_prewarm tool not found under build_Release/tools; submitting without prewarm" >&2
fi

echo "== submit to broker (timeout=${TIMEOUT}s) =="
exec tt-device-mcp run-bg "$CMD" -w "$WORKSPACE" -t "$TIMEOUT" ${ENV_FILE:+-e "$ENV_FILE"}
