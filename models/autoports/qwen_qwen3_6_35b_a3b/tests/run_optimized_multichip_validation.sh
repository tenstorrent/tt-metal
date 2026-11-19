#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
model=models/autoports/qwen_qwen3_6_35b_a3b
artifact=$model/doc/optimized_multichip_decoder
prior=$model/doc/multichip_decoder
module=models.autoports.qwen_qwen3_6_35b_a3b.tests
run_case() {
    local name=$1 bound=$2
    shift 2
    [[ ! -e "$artifact/STOP" ]] || exit 3
    if fuser /dev/tenstorrent/*; then echo 'Target mesh is owned' >&2; exit 2; fi
    timeout "$bound" "$@" --output "$artifact/$name.json" > "$artifact/logs/$name.log" 2>&1
    echo "PASS $name"
}
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
case ${1:?clean, boundary, stress, watcher, watcher_paths, natural, mixed, functional, or context} in
    clean)
        for layer in 0 3; do
            run_case "final_l${layer}_b1" 1800 python -m "$module.run_multichip_decoder" \
                --layer "$layer" --length 32 --decode-steps 8 --repeats 200 --prefill-repeats 3 --stress-repeats 3
            run_case "final_l${layer}_b32" 1800 python -m "$module.run_multichip_decoder" \
                --layer "$layer" --length 32 --batch 32 --decode-steps 8 --repeats 60 --prefill-repeats 3 \
                --stress-repeats 3 --activation-file "$prior/natural_b32.pt" --save-prefill-states
        done
        ;;
    boundary)
        for layer in 0 3; do
            for length in 1 31 65 257; do
                run_case "boundary_l${layer}_s${length}" 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 \
                    TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m "$module.run_multichip_decoder" \
                    --layer "$layer" --length "$length" --decode-steps 4 --repeats 5 --prefill-repeats 1 --stress-repeats 2
            done
        done
        ;;
    stress)
        for layer in 0 3; do
            run_case "stress_l${layer}_s129_d128" 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 \
                TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m "$module.run_multichip_decoder" \
                --layer "$layer" --length 129 --decode-steps 128 --repeats 20 --prefill-repeats 1 --stress-repeats 4
        done
        ;;
    natural)
        for layer in 0 3; do
            run_case "natural_l${layer}_b32" 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 \
                TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m "$module.run_multichip_decoder" \
                --layer "$layer" --length 32 --batch 32 --decode-steps 8 --repeats 30 --prefill-repeats 1 \
                --stress-repeats 3 --activation-file "$prior/natural_b32.pt" \
                --baseline-load "$prior/baseline_natural_l${layer}_b32.pt"
        done
        ;;
    watcher)
        for layer in 0 3; do
            for batch in 1 32; do
                extra=()
                if [[ $batch == 32 ]]; then
                    extra=(--activation-file "$prior/natural_b32.pt" --baseline-load "$prior/baseline_natural_l${layer}_b32.pt")
                fi
                run_case "watcher_l${layer}_b${batch}" 1800 env TT_METAL_WATCHER=10 \
                    TT_METAL_LOGS_PATH="$artifact/watcher/l${layer}_b${batch}" \
                    TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
                    python -m "$module.run_multichip_decoder" --layer "$layer" --length 32 --batch "$batch" \
                    --decode-steps 8 --repeats 5 --prefill-repeats 1 --stress-repeats 2 "${extra[@]}"
            done
        done
        ;;
    watcher_paths)
        run_case watcher_mixed_non_aligned 1800 env TT_METAL_WATCHER=10 \
            TT_METAL_LOGS_PATH="$artifact/watcher/mixed_non_aligned" \
            TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
            python -m "$module.run_multichip_mixed_stack" --prefix-lengths 33 35 \
            --continuation-lengths 32 34 --decode-steps 4
        for layer in 0 3; do
            run_case "watcher_l${layer}_s1" 1800 env TT_METAL_WATCHER=10 \
                TT_METAL_LOGS_PATH="$artifact/watcher/l${layer}_s1" \
                TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
                python -m "$module.run_multichip_decoder" --layer "$layer" --length 1 \
                --decode-steps 4 --repeats 5 --prefill-repeats 1 --stress-repeats 2
        done
        ;;
    mixed)
        run_case mixed_stack_selected 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
            python -m "$module.run_multichip_mixed_stack"
        mkdir -p "$artifact/trace_alloc_tracking"
        timeout 300 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
            python -m "$module.run_multichip_trace_control" --output "$artifact/trace_alloc_tracking/positive_control.json" \
            > "$artifact/trace_alloc_tracking/positive_control.log" 2>&1
        echo 'PASS trace_allocation_positive_control'
        ;;
    functional)
        for layer in 0 3; do
            run_case "functional_l${layer}" 1800 python -m "$module.run_multichip_decoder" \
                --layer "$layer" --length 128 --decode-steps 128 --repeats 30 --prefill-repeats 3 \
                --stress-repeats 3 --functional-sequence --functional-repeats 3
        done
        ;;
    context)
        for layer in 3 0; do
            run_case "context_l${layer}_32k" 7200 python -m "$module.run_multichip_context_smoke" \
                --layer "$layer" --length 32769 --context 65536 --baseline-load "$prior/baseline_context_l${layer}_32k.pt"
        done
        ;;
    *) exit 2 ;;
esac
