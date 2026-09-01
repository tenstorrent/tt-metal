#!/bin/bash
cd "$(dirname "$0")"
LOG=/tmp/claude-4015/-data-cglagovich-tt-metal/7bf353e5-b3ae-491a-987f-f299bc56b26a/scratchpad/vsa_pipeline_test.log
nohup ./run_vsa_pipeline_test.sh models/tt_dit/tests/models/minimax_h3/test_vsa_pipeline_minimax_h3.py -q \
  > "$LOG" 2>&1 < /dev/null &
echo "launched pid $!"
