#!/usr/bin/env bash
# Run from the checkout after activating python_env and lane-env.sh.
set -euo pipefail
model_root=models/autoports/meta_models_muse_glimmer_30b
reference_python=${HF_REFERENCE_PYTHON:-/home/vkovacevic/muse-glimmer/hf-reference-env/bin/python}
unset TT_METAL_WATCHER TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
export TRACY_NO_WEB_SERVER=1
# Keep all 20 measured trace replays plus setup/correctness markers.
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000
for layer in 0 3; do
    destination="$model_root/doc/functional_decoder/tracy/layer_$layer"
    mkdir -p "$destination"
    "$reference_python" -m tracy -r -p -v -o "$destination/raw" "$model_root/tests/run_functional.py" --real --layer "$layer" --lengths 1024 --perf --output "$destination/perf_run.json" > "$destination/collection.log" 2>&1
    ops_csv=$(rg --files --hidden --no-ignore "$destination/raw" | rg '/ops_perf_results[^/]*[.]csv$' | sort | tail -1)
    test -n "$ops_csv"
    for phase in prefill decode; do
        cp "$ops_csv" "$destination/${phase}_ops.csv"
        signpost="PERF_${phase^^}"
        tt-perf-report "$destination/${phase}_ops.csv" --start-signpost "$signpost" --end-signpost "${signpost}_END" --csv "$destination/${phase}_perf_report.csv" --no-advice > "$destination/${phase}_perf_report.console.log"
        tt-perf-report "$destination/${phase}_ops.csv" --start-signpost "$signpost" --end-signpost "${signpost}_END" --no-summary --no-advice --no-color > "$destination/${phase}_perf_report.txt"
        gzip -n -c "$destination/${phase}_ops.csv" > "$destination/${phase}_ops.csv.gz"
    done
    gzip -n -c "$destination/collection.log" > "$destination/collection.log.gz"
done
