#!/bin/bash
# Device cycles per SDPA call, math_thread_row_sums off vs on, for the arms of an arm file.
#
#   run_cycles.sh <arms.json> <chip> <out.jsonl> [arm ...]     (default: every arm in the file)
#
# REPLAYS=<n> raises the replay count for short/noisy shapes (default 11).
# Scratch (profiler dumps, per-run logs) goes to $SDPA_AB_WORK, default /tmp/sdpa_ab.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${SDPA_AB_WORK:-/tmp/sdpa_ab}"
REPO="$(cd "$HERE/../../../../.." && pwd)"
source "$REPO/python_env/bin/activate"
ARMS=$1; CHIP=$2; OUT=$3; shift 3
NAMES=${@:-$(python3 -c "import json;print(' '.join(a['name'] for a in json.load(open('$ARMS'))))")}
: > "$OUT"
for name in $NAMES; do
  for flag in 0 1; do
    d=$WORK/prof/${name}_${flag}
    rm -rf "$d"; mkdir -p "$d"
    TT_VISIBLE_DEVICES=$CHIP TT_METAL_CACHE=$WORK/cache_cyc$CHIP \
      TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$d" \
      timeout 900 python "$HERE/bench_cycles.py" "$ARMS" "$name" "$flag" > "$d/run.log" 2>&1
    res=$(python3 "$HERE/bench_cycles.py" --parse "$d" 2>>"$d/run.log" || echo '{}')
    echo "{\"name\":\"$name\",\"flag\":$flag,\"r\":$res}" >> "$OUT"
  done
done
echo DONE_CYCLES
