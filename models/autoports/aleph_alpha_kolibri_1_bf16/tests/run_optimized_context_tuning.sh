#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Serial context attention tuning; uses the complete optimized QKV cache prefix.
set -euo pipefail
optimized_doc=models/autoports/aleph_alpha_kolibri_1_bf16/doc/optimized_decoder
optimized_module=models.autoports.aleph_alpha_kolibri_1_bf16.tests
unset OPT_RUNTIME OPT_LAYOUT OPT_PROJECTIONS OPT_PREFILL OPT_SPLIT OPT_WORKER_L1_SIZE
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
export OPT_REAL_INPUT=1 OPT_CACHE=bfloat8_b TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
for optimized_k in 128 256 512; do
    for optimized_layer in 0 4; do
        OPT_POLICY="{\"decode_k_chunk\":${optimized_k}}" python -m "${optimized_module}.optimized_context_edges" \
            --layer "$optimized_layer" --decode-only --output "context_tuning_k${optimized_k}_${optimized_layer}.json" \
            > "$optimized_doc/context_tuning_k${optimized_k}_${optimized_layer}.log" 2>&1
        echo "PASS context K${optimized_k} layer ${optimized_layer}"
    done
done
for optimized_pair in '512 32' '1024 16' '1024 32'; do
    read -r optimized_k optimized_cores <<< "$optimized_pair"
    OPT_POLICY="{\"decode_k_chunk\":${optimized_k},\"decode_cores_per_head\":${optimized_cores}}" \
        python -m "${optimized_module}.optimized_context_edges" --layer 4 --decode-only \
        --output "context_tuning_k${optimized_k}_c${optimized_cores}_4.json" \
        > "$optimized_doc/context_tuning_k${optimized_k}_c${optimized_cores}_4.log" 2>&1
    echo "PASS context K${optimized_k} cores${optimized_cores} layer4"
done
