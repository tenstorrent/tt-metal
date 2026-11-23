#!/bin/bash
# Same as alternate.sh but with LLK asserts compiled out (TT_LLK_DISABLE_ASSERTS=1, as the
# framework comment in helpers/test_config.py says perf jobs should run).
P=/proj_sw/user_dev/ctr-sbratic/tools/perf_cycles; REPS=${1:-3}
export TT_LLK_DISABLE_ASSERTS=1
for i in $(seq 1 $REPS); do
  $P/run_perf.sh orig_na_r$i perf_eltwise_binary.py
  $P/run_perf.sh hal8a_na_r$i perf_eltwise_binary_hal.py
done
echo "ALTERNATE_NA DONE $(date -u +%FT%TZ)" > $P/out/alternate_na_done.txt
