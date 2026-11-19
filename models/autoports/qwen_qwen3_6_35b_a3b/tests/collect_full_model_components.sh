#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Run from configured checkout, assigned idle mesh, fresh destination; no retries/resets.
set -euo pipefail
artifact=${1:?fresh artifact directory}
[[ ! -e "$artifact" ]] || { echo 'Preserve existing capture; choose a fresh directory' >&2; exit 2; }
command -v tt-perf-report >/dev/null
devices=(/dev/tenstorrent/*)
[[ -e "${devices[0]}" ]] || { echo 'No Tenstorrent devices present' >&2; exit 2; }
status=0
timeout 30 fuser "${devices[@]}" >/dev/null 2>&1 || status=$?
[[ "$status" == 1 ]] || { echo 'Mesh owned or ownership check failed' >&2; exit 2; }
mkdir -p "$artifact"
unset TT_METAL_WATCHER TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
unset TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES TT_METAL_PROFILER_NO_CACHE_OP_INFO
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=5000
export TT_METAL_PROFILER_MID_RUN_DUMP=1
export TTNN_OP_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1
module=models.autoports.qwen_qwen3_6_35b_a3b.tests
# xtrace is retained as the exact command record, separate from capture output.
exec 9>"$artifact/commands.log"
export BASH_XTRACEFD=9
set -x
capture_status=0
timeout 3600 python -m tracy -p -v --no-web-server -o "$artifact/raw" \
    -m "$module.run_full_model_performance" --reduced --profile --context 1024 \
    --output "$artifact/performance.json" >"$artifact/capture.log" 2>&1 || capture_status=$?
# tracy-capture can save an incomplete trace and return zero after a source-location overflow.
if rg -q 'Instrumentation failure:|markers were dropped|report will be partial' "$artifact/capture.log"; then
    echo 'Profiler capture is incomplete; preserved logs must be inspected before retrying' >&2
    exit 1
fi
[[ "$capture_status" == 0 ]] || exit "$capture_status"
timeout 3600 python -m "$module.profile_full_model_components" --mode convert \
    --artifact "$artifact" >"$artifact/postprocess.log" 2>&1
for phase in token_out model_only sampler_only; do
    signpost="PERF_${phase^^}"
    timeout 300 tt-perf-report "$artifact/ops.csv" --active-experts 8 \
        --start-signpost "$signpost" --end-signpost "${signpost}_END" \
        --csv "$artifact/${phase}_perf_report.csv" >"$artifact/${phase}_perf_report.console.log" 2>&1
    timeout 300 tt-perf-report "$artifact/ops.csv" --active-experts 8 \
        --start-signpost "$signpost" --end-signpost "${signpost}_END" \
        --no-summary --no-color >"$artifact/${phase}_perf_report.txt" 2>&1
done
timeout 300 python -m "$module.profile_full_model_components" --mode summarize \
    --artifact "$artifact" >"$artifact/summary.log" 2>&1
