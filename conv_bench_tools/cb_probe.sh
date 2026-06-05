#!/bin/bash
cd /localdev/wransom/tt-metal
env TT_CONV_BENCH_MODE="${MODE:-helper_sbm}" timeout 320 bash scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/conv/test_conv_bench.py 2>&1 | \
  grep -iE "tuner would pick|beyond max L1 size|Out of Memory|^FAILED|^PASSED|SAFE_PYTEST_RESULT|TT_FATAL|weight_num_subblocks=|completion reader" | head -6
