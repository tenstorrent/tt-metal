#!/bin/bash
# Run the remaining Wan2.1 correctness configs sequentially, resetting the fabric
# between each. Each run saves images under outputs/correctness/<tag>/ and logs to
# /tmp/corr_<tag>.log. Continues on failure.
set -u
source /localdev/cglagovich/setup_env.sh >/dev/null 2>&1
source python_env/bin/activate
unset TT_METAL_OPERATION_TIMEOUT_SECONDS
SHP="1x768x1024,1x1024x1024,1x1280x720,1x1536x2048,1x2048x1152"
T=models/tt_dit/tests/models/wan2_1/test_performance_wan2_1.py

run() {  # tag test_fn traced kfilter
  local tag=$1 fn=$2 tr=$3 kf=$4
  echo "===== RUN $tag (traced=$tr, $fn -k $kf) ====="
  tt-smi -glx_reset >/dev/null 2>&1
  WAN_RUN_TAG="$tag" WAN_TRACED="$tr" WAN_SWEEP_SHAPES="$SHP" \
    python -m pytest "$T::$fn" -k "$kf" -s -p no:cacheprovider --timeout=100000000 \
    > "/tmp/corr_$tag.log" 2>&1
  echo "===== DONE $tag (exit $?) ====="
}

run 4x4_sp0tp1_traced       test_resolution_sweep 1 wh_4x4_sp0tp1
run 4x4_sp0tp1_cfg_untraced test_cfg_parallel     0 cfg2_4x4_sp0tp1
run 4x4_sp0tp1_cfg_traced   test_cfg_parallel     1 cfg2_4x4_sp0tp1
run 2x4_sp0tp1_untraced     test_resolution_sweep 0 wh_2x4_sp0tp1
run 2x4_sp0tp1_traced       test_resolution_sweep 1 wh_2x4_sp0tp1
run 2x4_sp0tp1_cfg_untraced test_cfg_parallel     0 cfg2_2x4_sp0tp1
run 2x4_sp0tp1_cfg_traced   test_cfg_parallel     1 cfg2_2x4_sp0tp1
echo "===== ALL CORRECTNESS RUNS DONE ====="
