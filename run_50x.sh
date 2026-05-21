#!/bin/bash
for i in {1..50}; do
    echo "=== Run $i/50 ==="
    pytest models/demos/deepseek_v3_d_p/tests/perf/test_prefill_block_perf.py -k "block_2x4" -vvv --tb=short
done
