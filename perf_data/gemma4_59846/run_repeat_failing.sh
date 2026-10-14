#!/bin/bash
# Repeat every global-accuracy case that failed in global_accuracy.log, with full tracebacks.
D=/data/mmanzoor/sdpa_chunk_sweep
cd /home/mmanzoor/compiler/tt-metal && source activate
mapfile -t FAILED < <(grep -a "^FAILED tests/" $D/global_accuracy.log | sed 's/^FAILED \(tests[^ ]*\).*/\1/' | sort -u)
echo "Repeating ${#FAILED[@]} failed case(s):" | tee $D/repeat_failing.log
printf '  %s\n' "${FAILED[@]}" | tee -a $D/repeat_failing.log
if [ ${#FAILED[@]} -gt 0 ]; then
  pytest -sv --tb=long "${FAILED[@]}" 2>&1 | tee -a $D/repeat_failing.log
  echo "REPEAT EXIT ${PIPESTATUS[0]} $(date)" | tee -a $D/repeat_failing.log
fi
