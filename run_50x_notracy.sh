#!/bin/bash
for i in {1..50}; do
    echo "=== Run $i/50 ==="
    pytest models/demos/deepseek_v3_d_p/tests/test_prefill_block_loop.py -k "mesh-2x4-2link and layer3 and gate_device and no_ref and isl_6k4"
done
