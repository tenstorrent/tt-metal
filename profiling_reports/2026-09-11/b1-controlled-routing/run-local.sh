#!/usr/bin/env bash
set -euo pipefail
set +x
source /data/ssalice/.bashrc_out >/dev/null 2>&1
source /data/ssalice/temp/tt-metal/python_env/bin/activate
export TT_METAL_HOME=/data/ssalice/temp/tt-metal/.worktrees/mistral4-prefill-followups
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export PYTHONPATH="$TT_METAL_HOME:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib"
export TT_VISIBLE_DEVICES=0,1,2,3,11,10,9,8
export MESH_DEVICE=TG
export ARCH_NAME=blackhole
export TT_METAL_OPERATION_TIMEOUT_SECONDS=300
export MISTRAL4_HF_MODEL=/mnt/models/blaze/mistralai/Mistral-Small-4-119B-2603
export TT_MISTRAL4_PREFILL_TTNN_CACHE=/mnt/models/blaze/mistralai/Mistral-Small-4-Cache/CI
export MISTRAL4_MLA_REF_CACHE=/mnt/models/blaze/mistralai/Mistral-Small-4-Cache/mla_ref
export TT_MISTRAL4_PREFILL_HOST_REF_CACHE=/mnt/models/blaze/mistralai/Mistral-Small-4-Cache/host_ref
export MPLCONFIGDIR="$TT_METAL_HOME/profiling_reports/2026-09-11/visibility-probe/matplotlib"
unset TT_MESH_GRAPH_DESC_PATH TT_MESH_ID TTNN_OP_PROFILER TT_METAL_PROFILER_TRACE_TRACKING TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT
unset TT_METAL_PROFILE_PERF_COUNTERS TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_PROFILER
cd "$TT_METAL_HOME"
B1_ROOT=/data/ssalice/temp/tt-metal/.worktrees/mistral4-b1-investigation
B1_CASE_NAME=${1:-layer18-captured}
B1_RUN_DIR=$(mktemp -d "$B1_ROOT/profiling_reports/2026-09-11/b1-controlled-routing/run-${B1_CASE_NAME}-XXXXXX")
export B1_ROUTING_CASE="$B1_ROOT/profiling_reports/2026-09-11/b1-controlled-routing/cases/${B1_CASE_NAME}.json"
export B1_ITERATIONS=10
export TT_METAL_PROFILER_DIR="$B1_RUN_DIR/profiler"
printf '%s\n' "$B1_RUN_DIR"
cp "$B1_ROUTING_CASE" "$B1_RUN_DIR/case.json"
cp "$0" "$B1_RUN_DIR/run.sh"
cp "$B1_ROOT/models/demos/deepseek_v3_d_p/tests/perf/test_b1_controlled_dispatch_combine.py" "$B1_RUN_DIR/worker.py"
set +e
python -m tracy -p -r --check-exit-code -a device_kernel_duration -o "$B1_RUN_DIR/profiler" -t 5000 -m "pytest $B1_ROOT/models/demos/deepseek_v3_d_p/tests/perf/test_b1_controlled_dispatch_combine.py -v -s" > "$B1_RUN_DIR/runner.log" 2>&1
b1_status=$?
printf '%s\n' "$b1_status" > "$B1_RUN_DIR/exitcode"
exit "$b1_status"
