#!/bin/bash
# Re-run the N+11 2x2 with the CLASSIFYING harness.
#
# The original scored cells on exit code, pooling WEDGE (card Unknown|63) with TEARDOWN
# (wait_until_cores_done on a healthy card) -- and unarmed teardown hangs are silent, so a cell
# reading "0/98 clean" never established that no wedge occurred. See FINDINGS N+20.
#
# Cells are INTERLEAVED and randomized, not run block-by-block: card state drifts and the wedge
# straddles run boundaries, so a fixed cell order biases attribution toward whichever cell follows.
#
# Usage: N=100 DELAY=125 ./drisc_2x2_rerun.sh     # N = runs PER CELL (4 cells)

HARNESS=${HARNESS:-./drisc_hang_harness.sh}
N=${N:-100}; DELAY=${DELAY:-125}
BASE=${OUT_DIR:-${TMPDIR:-/tmp}}
SUM=$BASE/2x2_summary.txt; : > $SUM

echo "=== 2x2 RERUN: $N runs/cell at delay $DELAY, interleaved+randomized, classified ===" | tee -a $SUM
echo "cells: drisc/fast drisc/slow tensix/fast tensix/slow" | tee -a $SUM

# Interleave: one run per cell per round, cell order shuffled each round.
for round in $(seq 1 $N); do
  cells=(drisc:fast drisc:slow tensix:fast tensix:slow)
  for ((i=3; i>0; i--)); do j=$((RANDOM % (i+1))); t=${cells[i]}; cells[i]=${cells[j]}; cells[j]=$t; done
  for c in "${cells[@]}"; do
    dr=${c%%:*}; dp=${c##*:}
    # ALL FOUR CELLS RUN UNARMED as of 2026-08-07 (FINDINGS N+24).
    #
    # Slow cells used to be armed because they "failed at teardown ~100% of the time" -- that was the
    # harness passing `--gx 0`, which under slow dispatch means a 12x10 producer grid against an
    # 11-column drainer poll list. Fixed in drisc_hang_harness.sh (GX/GY, default 11x10); slow cells
    # now run 10/10 clean. Arming is no longer needed and would only reintroduce MASKED as a class.
    #
    # Unarmed everywhere also makes the 2x2 a real factorial for the first time: identical 110-core /
    # 550-lane load in every cell, and one baseline instead of two.
    A=0
    OUT_DIR=$BASE TAG=2x2_${dr}_${dp} DELAY=$DELAY N=1 ARMED=$A \
      DISPATCH=$dp DRAINER=$dr APPEND=1 $HARNESS >/dev/null 2>&1
    # harness truncates its own summary each call; keep per-cell CSVs by appending rows here
    tail -n +2 $BASE/harn_2x2_${dr}_${dp}/runs.csv 2>/dev/null >> $BASE/2x2_${dr}_${dp}.csv
  done
  [ $((round % 10)) -eq 0 ] && {
    echo "round $round/$N" | tee -a $SUM
    for c in drisc_fast drisc_slow tensix_fast tensix_slow; do
      f=$BASE/2x2_$c.csv; [ -f $f ] || continue
      w=$(cut -d, -f7 $f | grep -c WEDGE); t=$(cut -d, -f7 $f | grep -c TEARDOWN); n=$(wc -l < $f)
      echo "  $c: n=$n WEDGE=$w TEARDOWN=$t" | tee -a $SUM
    done
  }
done
echo "=== DONE ===" | tee -a $SUM
for c in drisc_fast drisc_slow tensix_fast tensix_slow; do
  f=$BASE/2x2_$c.csv; [ -f $f ] || continue
  w=$(cut -d, -f7 $f | grep -c WEDGE); t=$(cut -d, -f7 $f | grep -c TEARDOWN); n=$(wc -l < $f)
  echo "$c: n=$n WEDGE=$w TEARDOWN=$t" | tee -a $SUM
done
