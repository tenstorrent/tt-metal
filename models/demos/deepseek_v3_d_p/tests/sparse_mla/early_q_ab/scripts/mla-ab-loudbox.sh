#!/bin/bash
cd /data/malimpic/tt-metal-mla
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD MLA_PERF_RING4=1 MLA_A2AMM_IMPL=${MLA_A2AMM_IMPL:-generic} TT_VISIBLE_DEVICES=0,1,2,3 TMPDIR=/data/malimpic/tmp
echo "=== MLA A/B LoudBox ring4 start $(date)"
scripts/run_safe_pytest.sh "models/demos/deepseek_v3_d_p/tests/sparse_mla/test_early_q_fused_ab.py::${AB_TEST:-test_early_q_block_ab}" -s 2>&1 | tee /data/malimpic/mla-ab/loudbox-${MLA_A2AMM_IMPL:-generic}.$(date +%H%M).log | grep -E "EARLY_Q_HOST|EARLY_Q_(AB|BLOCK)_RESULT \{|passed|failed|SAFE_PYTEST_RESULT|Error"
echo "=== MLA A/B LoudBox ring4 end $(date)"
