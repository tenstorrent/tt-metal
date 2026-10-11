#!/usr/bin/env bash
# Reduced non-serving profile, serialized with every other device command.
set -euo pipefail
source models/autoports/meta_models_muse_glimmer_30b/tests/full_model_env.sh
unset TT_METAL_WATCHER TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
export TRACY_NO_WEB_SERVER=1
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000
full_model_root=models/autoports/meta_models_muse_glimmer_30b
profile_destination="${PROFILE_DESTINATION:-$full_model_root/doc/full_model/tracy_reduced}"
mkdir -p "$profile_destination"
python_env/bin/python -m tracy -r -p -v -o "$profile_destination/raw" "$full_model_root/tests/profile_full_model.py" --output "$profile_destination/perf_run.json" > "$profile_destination/collection.log" 2>&1
profile_ops=$(rg --files --hidden --no-ignore "$profile_destination/raw" | rg '/ops_perf_results[^/]*[.]csv$' | sort | tail -1)
test -n "$profile_ops"
for phase in prefill decode sample argmax token_out token_out_no_readback reset; do
    profile_signpost="PERF_${phase^^}"
    tt-perf-report "$profile_ops" --start-signpost "$profile_signpost" --end-signpost "${profile_signpost}_END" --tracing-mode --csv "$profile_destination/${phase}_perf_report.csv" > "$profile_destination/${phase}_perf_report.console.log"
    tt-perf-report "$profile_ops" --start-signpost "$profile_signpost" --end-signpost "${profile_signpost}_END" --tracing-mode --no-summary --no-color > "$profile_destination/${phase}_perf_report.txt"
done
gzip -n -c "$profile_ops" > "$profile_destination/ops_perf_results.csv.gz"
