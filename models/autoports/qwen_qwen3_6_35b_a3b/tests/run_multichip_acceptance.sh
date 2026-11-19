#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
artifact=models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder
module=models.autoports.qwen_qwen3_6_35b_a3b.tests
run_case() {
    local name=$1 bound=$2
    shift 2
    [[ ! -e "$artifact/STOP" ]] || { echo 'Stopped between device jobs'; exit 3; }
    if fuser /dev/tenstorrent/*; then echo 'Target mesh is owned' >&2; exit 2; fi
    timeout "$bound" "$@" > "$artifact/logs/$name.log" 2>&1
    echo "PASS $name"
}
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
if [[ ${1:-all} == all ]]; then
    run_case mixed_stack_selected 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
        python -m "$module.run_multichip_mixed_stack" --output "$artifact/mixed_stack_selected.json"
fi
if [[ ${1:-all} != context ]]; then
for layer in 0 3; do
    baseline=(--baseline-save "$artifact/baseline_natural_l${layer}_b32.pt")
    if [[ -e "$artifact/baseline_natural_l${layer}_b32.pt" ]]; then
        baseline=(--baseline-load "$artifact/baseline_natural_l${layer}_b32.pt")
    fi
    run_case "natural_l${layer}_b32" 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
        python -m "$module.run_multichip_decoder" --layer "$layer" --length 32 --batch 32 --decode-steps 8 \
        --repeats 30 --prefill-repeats 1 --stress-repeats 3 --activation-file "$artifact/natural_b32.pt" \
        "${baseline[@]}" --output "$artifact/natural_l${layer}_b32.json"
done
fi
for layer in 3 0; do
    baseline=(--baseline-save "$artifact/baseline_context_l${layer}_32k.pt")
    if [[ -e "$artifact/baseline_context_l${layer}_32k.pt" ]]; then
        baseline=(--baseline-load "$artifact/baseline_context_l${layer}_32k.pt")
    fi
    run_case "context_l${layer}_32k" 7200 env -u TT_METAL_TRACE_ALLOC_TRACKING \
        python -m "$module.run_multichip_context_smoke" --layer "$layer" --length 32769 --context 65536 \
        "${baseline[@]}" --output "$artifact/context_l${layer}_32k.json"
done
baseline=(--baseline-save "$artifact/baseline_context_fullmax.pt")
if [[ -e "$artifact/baseline_context_fullmax.pt" ]]; then
    baseline=(--baseline-load "$artifact/baseline_context_fullmax.pt")
fi
run_case context_fullmax 14400 env -u TT_METAL_TRACE_ALLOC_TRACKING \
    python -m "$module.run_multichip_context_smoke" --layer 3 --length 262142 --context 262144 \
    "${baseline[@]}" --output "$artifact/context_fullmax.json"
