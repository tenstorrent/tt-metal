#!/bin/bash
# Retro perf measurement on one board (asserts off): per repetition the original once, then each version.
# Named driver_* on purpose: board-idle checks of other jobs treat a running driver_* as busy.
# Usage: driver_perf.sh <board tag> <version name>...   (inside the board's container)
set -uo pipefail
P=/proj_sw/user_dev/ctr-sbratic/tools/perf_cycles; TAG=${1:?board tag}; shift; REPS=${REPS:-3}
export TT_LLK_DISABLE_ASSERTS=1 OUTDIR=$P/out/retro; mkdir -p $OUTDIR
echo "START $(date -u +%FT%TZ) board=$TAG versions=$*" > $OUTDIR/${TAG}_progress.txt
for i in $(seq 1 $REPS); do
  $P/run_perf.sh ${TAG}_orig_r$i perf_eltwise_binary.py $P/ws/$1
  echo "$(date -u +%FT%TZ) ${TAG}_orig_r$i: $(head -1 $OUTDIR/${TAG}_orig_r$i/status.txt)" >> $OUTDIR/${TAG}_progress.txt
  for v in "$@"; do
    $P/run_perf.sh ${v}_r$i perf_eltwise_binary_hal.py $P/ws/$v
    echo "$(date -u +%FT%TZ) ${v}_r$i: $(head -1 $OUTDIR/${v}_r$i/status.txt)" >> $OUTDIR/${TAG}_progress.txt
  done
done
echo "DONE $(date -u +%FT%TZ)" > $OUTDIR/${TAG}_done.txt
