#!/bin/bash
# Run a pytest (through scripts/run_safe_pytest.sh) on one 4x4 half of a Blackhole galaxy, or the full 8x4.
# usage: run_stage_test.sh <top|bottom|full> <x|y|none> <logfile> <run_safe_pytest args...>
#   x / y pick the 4x4 subtorus MGD (TORUS_X / TORUS_Y orientation); none uses the default 8x4 MGD.
# env: KIMI_K3_CKPT, KIMI_K3_HF_MODEL, TT_KIMI_K3_PREFILL_TTNN_CACHE (4x4 weight cache), SAFE_PYTEST_TIMEOUT
# Example (profile one 12-layer stage at 640 tokens/chip):
#   KIMI_K3_TEST_SEQ_LEN=2560 run_stage_test.sh top x /tmp/p.log --profile \
#     "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_transformer_depth.py::test_depth_ladder_matches_golden[blackhole-L12-fabric2d-4x4]" -rs
HALF=$1; ORI=$2; LOG=$3; shift 3
REPO=$(cd "$(dirname "$0")/../../../../.." && pwd)
cd "$REPO"
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export KIMI_K3_CKPT=${KIMI_K3_CKPT:-/mnt/weka/model-weights/llm/moonshotai/Kimi-K3-dequantized}
export KIMI_K3_HF_MODEL=${KIMI_K3_HF_MODEL:-/mnt/weka/model-weights/llm/moonshotai/Kimi-K3-mxfp4-2496450e}
export HF_HUB_OFFLINE=1
# Without a cache the test rebuilds every expert weight from the checkpoint (~40 min), and under
# --profile the resulting host trace is large enough for Tracy to drop op records.
export TT_KIMI_K3_PREFILL_TTNN_CACHE=${TT_KIMI_K3_PREFILL_TTNN_CACHE:-$HOME/k3_cache_4x4}
# tt_metal/jit_build/build.cpp passes ccache settings inline, which needs ccache >= 4.8.
unset TT_METAL_CCACHE_KERNEL_SUPPORT
case $HALF in
  top)    export TT_VISIBLE_DEVICES=0,4,28,24,1,5,29,25,2,6,30,26,3,7,31,27 ;;
  bottom) export TT_VISIBLE_DEVICES=11,15,23,19,10,14,22,18,9,13,21,17,8,12,20,16 ;;
  full)   unset TT_VISIBLE_DEVICES ;;
  *) echo "half must be top, bottom or full"; exit 2 ;;
esac
MGD_DIR=$PWD/models/demos/deepseek_v3_d_p/experimental_descriptors
case $ORI in
  x) export TT_MESH_GRAPH_DESC_PATH=$MGD_DIR/single_bh_galaxy_subtorus_x4_graph_descriptor.textproto ;;
  y) export TT_MESH_GRAPH_DESC_PATH=$MGD_DIR/single_bh_galaxy_subtorus_y4_graph_descriptor.textproto ;;
  none) unset TT_MESH_GRAPH_DESC_PATH ;;
  *) echo "orientation must be x, y or none"; exit 2 ;;
esac
export SAFE_PYTEST_TIMEOUT=${SAFE_PYTEST_TIMEOUT:-60}
echo "run_stage_test half=$HALF ori=$ORI TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-all} MGD=${TT_MESH_GRAPH_DESC_PATH:-default}" > "$LOG"
scripts/run_safe_pytest.sh "$@" >> "$LOG" 2>&1
rc=$?
# run_safe_pytest.sh can report PASS on a collection error and FAIL when only the profiler report
# step failed, so print pytest's own summary next to its verdict.
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "^=+ .*(passed|failed|error|skipped)|SAFE_PYTEST_RESULT|PROFILER CSV" | tail -3
exit $rc
