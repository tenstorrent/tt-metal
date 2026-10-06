#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run from checkout root after sourcing python_env and the hardware lane environment.
# One hardware process at a time; stop immediately if any required check fails.
set -euo pipefail
optimized_doc=models/autoports/aleph_alpha_kolibri_1_bf16/doc/optimized_decoder
optimized_module=models.autoports.aleph_alpha_kolibri_1_bf16.tests
unset OPT_POLICY OPT_RUNTIME OPT_LAYOUT OPT_PROJECTIONS OPT_PREFILL OPT_SPLIT OPT_WORKER_L1_SIZE
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
export OPT_REAL_INPUT=1 OPT_CACHE=bfloat8_b TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
for optimized_layer in ${OPT_ACCEPT_LAYERS:-0 4}; do
    echo "START acceptance layer ${optimized_layer}"
    TT_METAL_TRACE_ALLOC_TRACKING=0 OPT_TAG=final python -m "${optimized_module}.optimized_profile" \
        --layer "$optimized_layer" --repetitions 100 > "$optimized_doc/final_${optimized_layer}.log" 2>&1
    TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_WATCHER=10 python -m "${optimized_module}.optimized_coverage" \
        --layer "$optimized_layer" --output "watcher_${optimized_layer}.json" > "$optimized_doc/watcher_${optimized_layer}.log" 2>&1
    for optimized_batch in 3 17 32; do
        TT_METAL_TRACE_ALLOC_TRACKING=1 python -m "${optimized_module}.optimized_decoder_checks" \
            --layer "$optimized_layer" --length 33 --batch "$optimized_batch" --public \
            --output "batch_${optimized_batch}_${optimized_layer}.json" > "$optimized_doc/batch_${optimized_batch}_${optimized_layer}.log" 2>&1
    done
    TT_METAL_TRACE_ALLOC_TRACKING=1 OPT_CACHE=bfloat16 python -m "${optimized_module}.optimized_decoder_checks" \
        --layer "$optimized_layer" --length 129 --public --output "bf16_cache_${optimized_layer}.json" \
        > "$optimized_doc/bf16_cache_${optimized_layer}.log" 2>&1
    TT_METAL_TRACE_ALLOC_TRACKING=1 python -m "${optimized_module}.optimized_context_edges" \
        --layer "$optimized_layer" > "$optimized_doc/context_edges_${optimized_layer}.log" 2>&1
    TT_METAL_TRACE_ALLOC_TRACKING=1 OPT_CACHE=bfloat16 python -m "${optimized_module}.optimized_context_edges" \
        --layer "$optimized_layer" --output "bf16_context_edges_${optimized_layer}.json" \
        > "$optimized_doc/bf16_context_edges_${optimized_layer}.log" 2>&1
    echo "PASS acceptance layer ${optimized_layer}"
done
