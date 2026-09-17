#!/usr/bin/env bash
# Local B1 controlled dispatch/combine replay, adapted from run-local.sh for this checkout.
set -uo pipefail
source /data/akhan/tt-metal/models/demos/deepseek_v3_d_p/tests/perf/pp4/env.sh
export TT_METAL_HOME=/data/akhan/tt-metal
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export PYTHONPATH="$TT_METAL_HOME:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build_Release/lib"
export TT_VISIBLE_DEVICES=0,1,2,3,11,10,9,8
export MESH_DEVICE=TG ARCH_NAME=blackhole
export TT_METAL_OPERATION_TIMEOUT_SECONDS=300
export TT_MISTRAL4_PREFILL_TTNN_CACHE="${M4_CACHE_8x1}"
unset TT_MESH_GRAPH_DESC_PATH TT_MESH_ID TTNN_OP_PROFILER TT_METAL_PROFILER_TRACE_TRACKING
unset TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT TT_METAL_PROFILE_PERF_COUNTERS
unset TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_PROFILER TT_MOE_ROUTING_CAPTURE
cd "$TT_METAL_HOME"

CASE_NAME=${1:?usage: run_b1.sh <case-name>}
RUN_DIR=$(mktemp -d "${B1_OUT:?B1_OUT must be set}/run-${CASE_NAME}-XXXXXX")
export B1_ROUTING_CASE="${B1_CASES:?B1_CASES must be set}/${CASE_NAME}.json"
export B1_ITERATIONS=${B1_ITERATIONS:-10}
export TT_METAL_PROFILER_DIR="$RUN_DIR/profiler"
printf '%s\n' "$RUN_DIR"
cp "$B1_ROUTING_CASE" "$RUN_DIR/case.json"
./python_env/bin/python -m tracy -p -r --check-exit-code -a device_kernel_duration \
  -o "$RUN_DIR/profiler" -t 5000 \
  -m "pytest models/demos/deepseek_v3_d_p/tests/perf/test_b1_controlled_dispatch_combine.py -v -s" \
  > "$RUN_DIR/runner.log" 2>&1
printf '%s\n' "$?" > "$RUN_DIR/exitcode"
