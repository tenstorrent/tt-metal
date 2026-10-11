#!/usr/bin/env bash
# Activate tt-metal python_env and lane-env.sh before invoking. Serial hardware only.
set -euo pipefail
root=models/autoports/meta_models_muse_glimmer_30b
ref=${HF_REFERENCE_PYTHON:-/home/vkovacevic/muse-glimmer/hf-reference-env/bin/python}
export HF_REFERENCE_PYTHON="$ref"
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
"$ref" -m pytest -q "$root/tests/test_optimized_decoder.py" --basetemp "$root/doc/optimized_decoder/pytest_cases" --junitxml "$root/doc/optimized_decoder/pytest.xml"
for layer in 0 3; do
    for batch in 17 25 31 32; do
        "$ref" -m models.autoports.meta_models_muse_glimmer_30b.tests.run_optimized --real --layer "$layer" --batch "$batch" --lengths 33 --output "$root/doc/optimized_decoder/batch${batch}_${layer}.json"
    done
    "$ref" -m models.autoports.meta_models_muse_glimmer_30b.tests.run_optimized --real --layer "$layer" --batch 2 --lengths 33 --continuation 1057 --output "$root/doc/optimized_decoder/continuation_${layer}.json"
    "$ref" -m models.autoports.meta_models_muse_glimmer_30b.tests.run_optimized --real --layer "$layer" --lengths 4097 --control --output "$root/doc/optimized_decoder/chunk_control_${layer}.json"
    "$ref" -m models.autoports.meta_models_muse_glimmer_30b.tests.run_optimized_trace_reuse --layer "$layer"
done
