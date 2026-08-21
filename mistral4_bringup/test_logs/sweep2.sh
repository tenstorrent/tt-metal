#!/usr/bin/env bash
# Re-run with 32-chip-safe params (// 4 expert scaledown => 1 expert/chip on 8x4).
R=/data/ssalice/temp/tt-metal/mistral4_bringup/test_logs/run.sh
T=models/demos/deepseek_v3_d_p/tests
$R vp_prefill_dispatch_perf $T/op_unit_tests/test_prefill_dispatch.py -k "mistral4-perf_no_pcc and mesh-8x4" -vvv --tb=short
$R vp_prefill_combine_8x4   $T/op_unit_tests/test_prefill_combine.py -k "mistral4 and mesh-8x4" -vvv --tb=short
echo "SWEEP2 DONE"
