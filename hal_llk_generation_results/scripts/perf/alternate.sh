#!/bin/bash
# Alternate original / HAL perf runs on one board: orig_rN, hal8a_rN for N=1..REPS.
P=/proj_sw/user_dev/ctr-sbratic/tools/perf_cycles; REPS=${1:-3}
for i in $(seq 1 $REPS); do
  $P/run_perf.sh orig_r$i perf_eltwise_binary.py
  $P/run_perf.sh hal8a_r$i perf_eltwise_binary_hal.py
done
echo "ALTERNATE DONE $(date -u +%FT%TZ)" > $P/out/alternate_done.txt
