#!/bin/bash
# Smoke probe: run one mode (MODE env, default helper_sbm) under the hang-guarded pytest wrapper and
# surface the key dispatch / eligibility / status lines. Run vu_288 in all 3 modes before the sweep.
WT=/localdev/wransom/tt-metal/.claude/worktrees/agent-a48fa14207415d0cb
cd "$WT"
export TT_METAL_HOME="$WT"
export PYTHONPATH="$WT/ttnn:$WT"
env TT_CONV_BENCH_MODE="${MODE:-helper_sbm}" TT_CONV_BENCH_FORCE_TRM="$( [ "${MODE:-helper_sbm}" = helper_trm ] && echo 1 )" \
  timeout 320 bash scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/conv/test_conv_bench.py 2>&1 | \
  grep -iE "tuner picks|ELIGIBLE|FALLBACK|CONV_BENCH\[|beyond max L1 size|Out of Memory|^FAILED|^PASSED|SAFE_PYTEST_RESULT|TT_FATAL|completion reader" | head -12
