#!/usr/bin/env bash
# Run from the checkout after activating python_env and lane-env.sh.
set -euo pipefail
model_root=models/autoports/meta_models_muse_glimmer_30b
reference_python=${HF_REFERENCE_PYTHON:-/home/vkovacevic/muse-glimmer/hf-reference-env/bin/python}
unset TT_METAL_WATCHER TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
export TRACY_NO_WEB_SERVER=1
# Keep all 20 measured trace replays plus setup/correctness markers.
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000
for layer in ${LAYERS:-0 3}; do
    destination="$model_root/doc/optimized_decoder/${PROFILE_SUBDIR:-tracy}/layer_$layer"
    mkdir -p "$destination"
    policy_args=()
    if [[ -n "${POLICY:-}" ]]; then policy_args=(--policy "$POLICY"); fi
    "$reference_python" -m tracy -r -p -v -o "$destination/raw" "$model_root/tests/run_optimized.py" --real "${policy_args[@]}" --layer "$layer" --lengths 1024 --perf --output "$destination/perf_run.json" > "$destination/collection.log" 2>&1
    ops_csv=$(rg --files --hidden --no-ignore "$destination/raw" | rg '/ops_perf_results[^/]*[.]csv$' | sort | tail -1)
    test -n "$ops_csv"
    for phase in prefill decode; do
        cp "$ops_csv" "$destination/${phase}_ops.csv"
        signpost="PERF_${phase^^}"
        tt-perf-report "$destination/${phase}_ops.csv" --start-signpost "$signpost" --end-signpost "${signpost}_END" --tracing-mode --csv "$destination/${phase}_perf_report.csv" > "$destination/${phase}_perf_report.console.log"
        # Normalize generated table line endings; preserve original raw CSV in gzip.
        "$reference_python" - "$destination/${phase}_perf_report.csv" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
PY
        tt-perf-report "$destination/${phase}_ops.csv" --start-signpost "$signpost" --end-signpost "${signpost}_END" --tracing-mode --no-summary --no-color > "$destination/${phase}_perf_report.txt"
        gzip -n -c "$destination/${phase}_ops.csv" > "$destination/${phase}_ops.csv.gz"
    done
    gzip -n -c "$destination/collection.log" > "$destination/collection.log.gz"
done
