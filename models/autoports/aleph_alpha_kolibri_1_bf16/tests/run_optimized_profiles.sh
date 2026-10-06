#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Serial profiler runs; watcher must remain disabled.
set -euo pipefail
optimized_doc=models/autoports/aleph_alpha_kolibri_1_bf16/doc/optimized_decoder
optimized_module=models.autoports.aleph_alpha_kolibri_1_bf16.tests
unset OPT_POLICY OPT_RUNTIME OPT_LAYOUT OPT_PROJECTIONS OPT_PREFILL OPT_SPLIT OPT_WORKER_L1_SIZE
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
export OPT_REAL_INPUT=1 OPT_CACHE=bfloat8_b TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
for optimized_layer in 0 4; do
    for optimized_variant in final baseline; do
        optimized_extra=()
        optimized_profile_dir="$optimized_doc/tracy/layer_${optimized_layer}"
        if [[ "$optimized_variant" == baseline ]]; then
            optimized_extra=(--baseline)
            optimized_profile_dir="$optimized_doc/baseline/tracy/layer_${optimized_layer}"
        fi
        mkdir -p "$optimized_profile_dir"
        OPT_TAG="tracy_${optimized_variant}_${optimized_layer}" python -m tracy -r -p -v --no-web-server \
            -o "$optimized_profile_dir/raw" -m "${optimized_module}.optimized_profile" \
            --layer "$optimized_layer" --repetitions 3 "${optimized_extra[@]}" > "$optimized_profile_dir/collection.log" 2>&1
        python -m "${optimized_module}.optimized_render_profile" "$optimized_profile_dir" \
            > "$optimized_profile_dir/render.log" 2>&1
        echo "PASS profile ${optimized_variant} layer ${optimized_layer}"
    done
    optimized_profile_dir="$optimized_doc/tracy/long_layer_${optimized_layer}"
    mkdir -p "$optimized_profile_dir"
    OPT_PROFILE_DRAIN=1 OPT_TAG="tracy_long_${optimized_layer}" python -m tracy -r -p -v --no-web-server \
        -o "$optimized_profile_dir/raw" -m "${optimized_module}.optimized_profile" \
        --layer "$optimized_layer" --tokens 8193 --public --repetitions 1 \
        > "$optimized_profile_dir/collection.log" 2>&1
    python -m "${optimized_module}.optimized_render_profile" "$optimized_profile_dir" --repetitions 1 \
        > "$optimized_profile_dir/render.log" 2>&1
    echo "PASS long profile layer ${optimized_layer}"
done
mkdir -p "$optimized_doc/reader_profile"
python -m tracy -r -p -v --no-web-server -o "$optimized_doc/reader_profile/raw" \
    -m "${optimized_module}.optimized_dense_geometry" --layer 0 --readers-only --repetitions 3 \
    > "$optimized_doc/reader_profile/collection.log" 2>&1
python -m "${optimized_module}.optimized_reader_report" "$optimized_doc/reader_profile" \
    "$optimized_doc/dense_geometry_0_readers.json" > "$optimized_doc/reader_profile/render.log" 2>&1
echo "PASS reader profile"
