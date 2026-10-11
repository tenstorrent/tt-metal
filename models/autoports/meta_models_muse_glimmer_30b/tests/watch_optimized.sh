#!/usr/bin/env bash
# Separate from Tracy/profiler collection; no watcher assertions are disabled.
set -euo pipefail
root=models/autoports/meta_models_muse_glimmer_30b
ref=${HF_REFERENCE_PYTHON:-/home/vkovacevic/muse-glimmer/hf-reference-env/bin/python}
export TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset TT_METAL_DEVICE_PROFILER
for layer in 0 3; do
    for batch in ${BATCHES:-1 9 32}; do
        length=33
        if [[ "$batch" == 1 ]]; then length=1024; fi
        destination="$root/doc/optimized_decoder/watcher_${layer}_b${batch}"
        mkdir -p "$destination"
        TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH="$PWD/$destination" "$ref" "$root/tests/run_optimized.py" --real --layer "$layer" --batch "$batch" --lengths "$length" --output "$destination/results.json" > "$destination/console.log" 2>&1
        cp "$destination/generated/watcher/watcher.log" "$destination/watcher.log"
    done
done
