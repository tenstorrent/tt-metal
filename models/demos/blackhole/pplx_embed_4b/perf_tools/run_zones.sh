#!/bin/bash
# Per-work-unit compute-kernel zones, math_thread_row_sums off vs on.
# Needs the kernel variant trees built first:
#   python sdpa_kernel_variants.py <kvdir> zconly zfconly
#
#   SDPA_AB_KV=<kvdir> run_zones.sh <zconly|zfconly> <chip> <arm> [arm ...]
#
# Scratch goes to $SDPA_AB_WORK, default /tmp/sdpa_ab.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${SDPA_AB_WORK:-/tmp/sdpa_ab}"
KV="${SDPA_AB_KV:-$WORK/kv}"
REPO="$(cd "$HERE/../../../../.." && pwd)"
source "$REPO/python_env/bin/activate"
VAR=$1; CHIP=$2; shift 2
PARSE=--parse; [ "$VAR" = zfconly ] && PARSE=--parse-flat
for name in "$@"; do
  for flag in 0 1; do
    d=$WORK/zprof/${VAR}_${name}_${flag}
    rm -rf "$d"; mkdir -p "$d"
    # Kernel lookup tries the cwd before TT_METAL_KERNEL_PATH, so run from a directory with no ttnn/ tree.
    ( cd "$WORK" && TT_VISIBLE_DEVICES=$CHIP \
      TT_METAL_KERNEL_PATH=$KV/$VAR TT_METAL_CACHE=$KV/cache_$VAR \
      TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$d" \
      timeout 1200 python "$HERE/bench_zones.py" "$HERE/arms_Z.json" "$name" "$flag" > "$d/run.log" 2>&1 )
    echo "=== $VAR $name flag=$flag ==="
    python3 "$HERE/bench_zones.py" $PARSE "$d" 2>&1 | tail -40
  done
done
echo DONE_ZONES
