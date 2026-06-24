#!/bin/bash
# Run on HOST. Sweep mux_core_range_set geometries for the fused-combine smoke. Resets the
# galaxy before EACH run (a hang wedges the combine ring), runs the verdict in the
# container, and records HANG/PASS. Goal: map which mux geometries deadlock and isolate
# whether it's the mux column location, full-height shape, or size.
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
TS=$(date +%Y%m%d_%H%M%S)
SUM="/data/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro/logs/sweep_mux_${TS}.txt"

# box "x0,y0,x1,y1"  : note
declare -a BOXES=(
  "3,0,4,7|original smoke: 2-wide x=3,4 full-height (16 cores)"
  "1,0,2,7|western 2-wide x=1,2 full-height (16 cores)"
  "3,1,4,2|small 2x2 at smoke x=3,4 (4 cores)"
  "1,1,3,3|model default 3x3 x=1..3 y=1..3 (9 cores)"
  "5,0,6,7|eastern 2-wide x=5,6 full-height (16 cores)"
)

echo "=== mux sweep @ $TS ===" | tee "$SUM"
for entry in "${BOXES[@]}"; do
  box="${entry%%|*}"; note="${entry#*|}"
  echo "" | tee -a "$SUM"
  echo "########## mux=($box)  [$note] ##########" | tee -a "$SUM"
  echo "-- reset --" | tee -a "$SUM"
  tt-smi -r >/dev/null 2>&1; echo "   reset rc=$?" | tee -a "$SUM"
  out=$(docker exec --user 4123:4123 "$C" /bin/bash -lc "export SMOKE_MUX='$box'; bash $D/run_verdict.sh mux_${box//,/_}" 2>&1)
  verdict=$(echo "$out" | grep -oE 'VERDICT: [A-Z_]+' | head -1)
  placement=$(echo "$out" | grep -oE 'selected tilize cores [0-9]+, combine cores [0-9]+, matmul cores [0-9]+' | head -1)
  strag=$(echo "$out" | grep -E 'no stragglers|stragglers remain' | head -1)
  echo "   >>> $verdict | $placement | $strag" | tee -a "$SUM"
done
echo "" | tee -a "$SUM"
echo "=== final reset ===" | tee -a "$SUM"
tt-smi -r >/dev/null 2>&1; echo "reset rc=$?" | tee -a "$SUM"
echo "=== SUMMARY ===" | tee -a "$SUM"
grep -E '####|>>>' "$SUM"
echo "(full log: $SUM)"
