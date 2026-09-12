#!/bin/bash
# One-shot perf run for PR #56146 (tensorbin payload 64B alignment).
# Prereqs: tt-metal built with the fix on this host, Wan2.2 T2V weights in the HF cache,
#          and IOMMU ENABLED (see HANDOFF.md). Run from anywhere.
#
# Env overrides:
#   TT_METAL_HOME  (default: autodetected from this script's repo, else $PWD)
#   WORK           scratch dir for caches   (default: /localdev/$USER/wan22_align_exp)
#   BLOCKS         transformer blocks to use (default: 40 == full transformer big linears)
#   DEVICES        mesh width                (default: 4)
#   WAN_TRANSFORMER_DIR  override weight location (default: autodiscovered)
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$HERE/wan_load_bench.py"

# --- environment ---
: "${TT_METAL_HOME:=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
export TT_METAL_HOME
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
export PYTHONPATH="$TT_METAL_HOME:${PYTHONPATH:-}"
export TT_METAL_LOGGER_LEVEL="${TT_METAL_LOGGER_LEVEL:-Info}"
export TT_LOGGER_LEVEL="${TT_LOGGER_LEVEL:-Info}"
WORK="${WORK:-/localdev/$USER/wan22_align_exp}"
BLOCKS="${BLOCKS:-40}"
DEVICES="${DEVICES:-4}"
PY="${PY:-python}"
mkdir -p "$WORK"
LOGDIR="$WORK/logs"; mkdir -p "$LOGDIR"

echo "TT_METAL_HOME=$TT_METAL_HOME  WORK=$WORK  BLOCKS=$BLOCKS  DEVICES=$DEVICES"

# --- 0. IOMMU sanity (OS level) ---
groups=$(ls /sys/kernel/iommu_groups 2>/dev/null | wc -l)
echo "IOMMU groups: $groups ; cmdline: $(grep -o 'intel_iommu=[^ ]*\|amd_iommu=[^ ]*\|iommu=[^ ]*' /proc/cmdline | tr '\n' ' ')"
if [ "$groups" -eq 0 ]; then
  echo "WARNING: /sys/kernel/iommu_groups is empty -- IOMMU looks OFF. Preflight will confirm."
fi

# --- 1. preflight: prove the pinned path is active before the big run ---
echo; echo "### PREFLIGHT ###"
$PY "$BENCH" preflight --work "$WORK/preflight" --devices "$DEVICES"

# --- 2. generate the full aligned cache + unaligned twins ---
echo; echo "### GENERATE ($BLOCKS blocks) ###"
$PY "$BENCH" gen --out "$WORK/aligned" --blocks "$BLOCKS"
$PY "$BENCH" misalign --src "$WORK/aligned" --dst "$WORK/unaligned"

# --- 3. bench both, capturing logs to count "Pinned source memory" rejections ---
run_bench () {  # $1 = cache dir, $2 = label
  local cache="$1" label="$2" log="$LOGDIR/bench_$2.log"
  $PY "$BENCH" bench --cache "$cache" --devices "$DEVICES" >"$log" 2>&1 || { cat "$log"; exit 1; }
  local rej; rej=$(grep -c "Pinned source memory" "$log" || true)
  local res; res=$(grep "RESULT" "$log" | tail -1)
  echo "$label: rejects=$rej | $res"
}
echo; echo "### BENCH ###"
run_bench "$WORK/aligned"   aligned   | tee "$LOGDIR/summary.txt"
run_bench "$WORK/unaligned" unaligned | tee -a "$LOGDIR/summary.txt"

echo; echo "### DONE ### logs in $LOGDIR"
echo "Report the two lines above: aligned should have rejects=0, unaligned should have"
echo "rejects = (#tensors * #devices), and unaligned total_load_time should be higher."
