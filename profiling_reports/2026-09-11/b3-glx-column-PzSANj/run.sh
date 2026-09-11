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
B3_RUN_DIR=$(mktemp -d "$TT_METAL_HOME/profiling_reports/2026-09-11/b3-glx-column-XXXXXX")
export TT_METAL_PROFILER_DIR="$B3_RUN_DIR/profiler"
printf '%s\n' "$B3_RUN_DIR"
git rev-parse HEAD > "$B3_RUN_DIR/revision.txt"
git diff > "$B3_RUN_DIR/working-tree.patch"
cp "$0" "$B3_RUN_DIR/run.sh"
set +e
python -m pytest models/demos/deepseek_v3_d_p/tests/perf/test_mla_perf.py::test_mistral4_mla_chunked_perf_loudbox -v -s > "$B3_RUN_DIR/runner.log" 2>&1
b3_status=$?
set -e
printf '%s\n' "$b3_status" > "$B3_RUN_DIR/exitcode"
exit "$b3_status"
