#!/usr/bin/env bash
# Move kernel JIT compilation out of the broker's device reservation, then submit the real job.
#
# The device is a shared, serialized resource; kernel compilation is host work. This keeps the reserved
# window device-only by compiling off-device first. Three stages, releasing the device between each:
#
#   1. capture  (device, brief) -- only when the build_key is COLD (no manifest, or -c): run the pipeline
#                once with TT_METAL_KERNEL_CAPTURE_ONLY=1 to record the manifest (genfiles + recipe, no
#                gcc, no dispatch; device-init kernels still compile). Skipped when the manifest already
#                covers the run -- then the offline stage alone warms it.
#   2. compile  (host, device FREE): kernel_prewarm tool batch-compiles every manifest recipe off-device.
#   3. run      (device): the real job, now warming a cache that needs no compilation.
#
# Because the compile runs while the device is unreserved, this holds the device less than the in-process
# cold-start (which stays reserved through the compile). Measured on LTX distilled (bh_2x4sp1tp0): cold
# build_key ~166s device-held across the two reservations (vs ~235s in-process, ~531s all-on-device);
# freshly-toggled dprint ~176s. Warm cache => the offline stage is a sub-second dephash no-op, so it is
# safe to prepend to every run.
#
#   prewarm_and_submit.sh -e <env.yaml> [-w <workspace>] [-t <timeout_s>] [-c] -- <command...>
#     -c/--capture  force the capture stage even if a manifest exists (use when the run's build_key is new
#                   but the cache already holds other build_keys, e.g. after toggling dprint/watcher).
#
# TT_METAL_CACHE / TT_METAL_HOME are read from the env yaml (or inherited if absent), matching the values
# the broker job uses, so every stage acts on the exact cache the run consumes.
set -uo pipefail

ENV_FILE=""
WORKSPACE="$PWD"
TIMEOUT=1200
FORCE_CAPTURE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--env) ENV_FILE="$2"; shift 2;;
    -w|--workspace) WORKSPACE="$2"; shift 2;;
    -t|--timeout) TIMEOUT="$2"; shift 2;;
    -c|--capture) FORCE_CAPTURE=1; shift;;
    --) shift; break;;
    *) echo "prewarm_and_submit.sh: unknown arg $1" >&2; exit 2;;
  esac
done
[[ $# -gt 0 ]] || { echo "usage: prewarm_and_submit.sh -e <env.yaml> [-w dir] [-t sec] [-c] -- <command...>" >&2; exit 2; }
CMD="$*"

# A yaml value may be bare or quoted (key: "val" / key: val); strip surrounding quotes and whitespace.
_yaml_get() {
  local key="$1" file="$2"
  [[ -f "$file" ]] || return 0
  sed -nE "s/^${key}:[[:space:]]*[\"']?([^\"']*)[\"']?[[:space:]]*$/\1/p" "$file" | head -n1
}

CACHE="$(_yaml_get TT_METAL_CACHE "$ENV_FILE")"; CACHE="${CACHE:-${TT_METAL_CACHE:-}}"
HOME_DIR="$(_yaml_get TT_METAL_HOME "$ENV_FILE")"; HOME_DIR="${HOME_DIR:-${TT_METAL_HOME:-$WORKSPACE}}"

# The manifest sits at the cache root (a sibling of the per-build_key tt-metal-cache<key> subtree), so a
# missing/empty file means no build_key has ever been captured here -- a genuinely cold cache.
MANIFEST="${CACHE:+$CACHE/kernel_prewarm.manifest}"

TOOL="$HOME_DIR/build_Release/tools/kernel_prewarm"
[[ -x "$TOOL" ]] || TOOL="$WORKSPACE/build_Release/tools/kernel_prewarm"

# Stage 1: capture. Only when cold -- the manifest is a byproduct of a real run, so a fresh cache has
# nothing to compile off-device until one exists. Skipped for a warm cache to avoid a wasted device touch.
if [[ "$FORCE_CAPTURE" == "1" || -z "$MANIFEST" || ! -s "$MANIFEST" ]]; then
  echo "== stage 1/3: capture pass (device held briefly; no gcc, no dispatch) =="
  # The pipeline runs on garbage tensors under capture-only, so its output check may FAIL -- that is
  # expected; the manifest is the artifact. Gate the next stage on the manifest growing, not the exit code.
  MANIFEST_BEFORE=0; [[ -n "$MANIFEST" && -s "$MANIFEST" ]] && MANIFEST_BEFORE=$(stat -c %s "$MANIFEST" 2>/dev/null || echo 0)
  tt-device-mcp run "TT_METAL_KERNEL_CAPTURE_ONLY=1 $CMD" -w "$WORKSPACE" -t "$TIMEOUT" ${ENV_FILE:+-e "$ENV_FILE"} || true
  MANIFEST_AFTER=0; [[ -n "$MANIFEST" && -s "$MANIFEST" ]] && MANIFEST_AFTER=$(stat -c %s "$MANIFEST" 2>/dev/null || echo 0)
  if [[ "$MANIFEST_AFTER" -le "$MANIFEST_BEFORE" ]]; then
    echo "prewarm_and_submit.sh: capture pass produced no manifest growth ($MANIFEST) -- aborting" >&2
    exit 1
  fi
else
  echo "== stage 1/3: capture skipped (manifest present: $MANIFEST) =="
fi

# Stage 2: off-device compile (device free). Builds every manifest recipe, incl. device-init kernels.
if [[ -x "$TOOL" ]]; then
  echo "== stage 2/3: off-device compile (device free; cache=${CACHE:-default}) =="
  env TT_METAL_CACHE="$CACHE" TT_METAL_HOME="$HOME_DIR" "$TOOL"
else
  echo "prewarm_and_submit.sh: kernel_prewarm tool not found under build_Release/tools; submitting without compile" >&2
fi

# Stage 3: the real run, now warm -- zero compilation inside the reservation.
echo "== stage 3/3: submit real run to broker (timeout=${TIMEOUT}s) =="
exec tt-device-mcp run-bg "$CMD" -w "$WORKSPACE" -t "$TIMEOUT" ${ENV_FILE:+-e "$ENV_FILE"}
