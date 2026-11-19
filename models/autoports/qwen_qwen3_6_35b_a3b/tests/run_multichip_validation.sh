#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
artifact=models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder
module=models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_decoder
run_case() {
    local name=$1
    shift
    [[ ! -e "$artifact/STOP" ]] || { echo 'Stopped between device jobs'; exit 3; }
    if fuser /dev/tenstorrent/*; then echo 'Target mesh is owned' >&2; exit 2; fi
    timeout 1800 "$@" --output "$artifact/$name.json" > "$artifact/logs/$name.log" 2>&1
    echo "PASS $name"
}
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
case ${1:?clean, boundary, stress, or watcher} in
    clean)
        for layer in 0 3; do
            run_case "final_l${layer}_b1" python -m "$module" --layer "$layer" --length 32 --decode-steps 2 \
                --repeats 200 --prefill-repeats 3 --stress-repeats 3
            run_case "final_l${layer}_b32" python -m "$module" --layer "$layer" --length 32 --batch 32 --decode-steps 8 \
                --repeats 60 --prefill-repeats 3 --stress-repeats 3 --activation-file "$artifact/natural_b32.pt"
        done
        ;;
    boundary)
        for layer in 0 3; do
            for length in 1 31 65 257; do
                run_case "boundary_l${layer}_s${length}" env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
                    python -m "$module" --layer "$layer" --length "$length" --decode-steps 4 \
                    --repeats 3 --prefill-repeats 1 --stress-repeats 2
            done
        done
        ;;
    stress)
        for layer in 0 3; do
            run_case "stress_l${layer}_s129_d128" env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
                python -m "$module" --layer "$layer" --length 129 --decode-steps 128 \
                --repeats 20 --prefill-repeats 1 --stress-repeats 4
        done
        ;;
    watcher)
        for layer in 0 3; do
            for batch in 1 32; do
                extra=()
                steps=2
                if [[ $batch == 32 ]]; then
                    steps=8
                    extra=(--activation-file "$artifact/natural_b32.pt" --baseline-load "$artifact/baseline_natural_l${layer}_b32.pt")
                fi
                run_case "watcher_l${layer}_b${batch}" env TT_METAL_WATCHER=10 \
                    TT_METAL_LOGS_PATH="$artifact/watcher/l${layer}_b${batch}" \
                    TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
                    python -m "$module" --layer "$layer" --length 32 --batch "$batch" --decode-steps "$steps" \
                    --repeats 5 --prefill-repeats 1 --stress-repeats 2 "${extra[@]}"
            done
        done
        ;;
    *) echo 'Unknown validation group' >&2; exit 2 ;;
esac
