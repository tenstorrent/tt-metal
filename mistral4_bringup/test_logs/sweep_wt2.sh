#!/usr/bin/env bash
R=/data/ssalice/temp/tt-metal/mistral4_bringup/test_logs/run_wt.sh
T=models/demos/deepseek_v3_d_p/tests
$R op_dispatch_combine $T/op_unit_tests/test_ttnn_dispatch_combine.py -k "mistral4 and 8x4" -vvv --tb=short
$R op_dispatch         $T/op_unit_tests/test_prefill_dispatch.py -k "mistral4-perf_no_pcc and 8x4" -vvv --tb=short
echo "SWEEP_WT2 DONE"
