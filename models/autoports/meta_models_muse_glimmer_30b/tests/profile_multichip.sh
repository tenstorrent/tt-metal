#!/usr/bin/env bash
# Serial, non-serving profiling. Source the lane environment before invoking.
set -euo pipefail
model_root=models/autoports/meta_models_muse_glimmer_30b
reference_python=/home/vkovacevic/muse-glimmer/hf-reference-env/bin/python
unset TT_METAL_WATCHER TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
export TRACY_NO_WEB_SERVER=1
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000
policy="${POLICY:-}"
if [[ -z "$policy" ]]; then
 policy='{}'
fi
extra_args=()
length="${PROFILE_LENGTH:-1024}"
reference_suffix=""
if [[ -n "${PROFILE_PAGES:-}" ]]; then
 extra_args+=(--pages "$PROFILE_PAGES")
 reference_suffix="_pages${PROFILE_PAGES}"
fi
if [[ "${MICROBENCH:-0}" == 1 ]]; then
 extra_args+=(--microbench)
fi
for layer in ${LAYERS:-0 3}; do
 destination="${STAGE_DIR:-$model_root/doc/multichip_decoder}/${PROFILE_SUBDIR:-tracy}/layer_$layer"
 mkdir -p "$destination"
 printf '%s\n' "$policy" > "$destination/policy.json"
 # Reuse the serial baseline tensor without rerunning device work under Tracy.
 reference="baseline_l${layer}_b1_s${length}_c0${reference_suffix}.pt"
 ln -sf "../../$reference" "$destination/$reference"
 "$reference_python" -m tracy -r -p -v -o "$destination/raw" "$model_root/tests/run_multichip.py" --mode "${MODE:-replicated}" --layer "$layer" --length "$length" --policy-file "$destination/policy.json" --perf --profile "${extra_args[@]}" --output "$destination/perf_run.json" > "$destination/collection.log" 2>&1
 ops_csv=$(rg --files --hidden --no-ignore "$destination/raw" | rg '/ops_perf_results[^/]*[.]csv$' | sort | tail -1)
 test -n "$ops_csv"
 for phase in prefill decode; do
   signpost="PERF_${phase^^}"
   tt-perf-report "$ops_csv" --start-signpost "$signpost" --end-signpost "${signpost}_END" --tracing-mode --csv "$destination/${phase}_perf_report.csv" > "$destination/${phase}_perf_report.console.log"
   tt-perf-report "$ops_csv" --start-signpost "$signpost" --end-signpost "${signpost}_END" --tracing-mode --no-summary --no-color > "$destination/${phase}_perf_report.txt"
 done
 gzip -n -c "$ops_csv" > "$destination/ops_perf_results.csv.gz"
 gzip -n -c "$destination/collection.log" > "$destination/collection.log.gz"
done
