#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
artifact=models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder
module=models.autoports.qwen_qwen3_6_35b_a3b.tests
waiter="${TT_MODEL_BRINGUP_ROOT:?}/scripts/wait_for"
unset TT_VISIBLE_DEVICES TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
run_job() {
    [[ ! -f "$artifact/STOP_QUEUE" ]] || exit 3
    local label="$1"
    shift
    if fuser /dev/tenstorrent/0; then exit 3; fi
    local command
    printf -v command '%q ' "$@"
    "$waiter" --cmd "$command" --file "$artifact/logs/$label.log" --timeout 1830 --tail 5
}
export QWEN_DECODER=optimized
for layer in 0 3 34; do
    run_job "natural_final_$layer" timeout 1800 python -m "$module.run_decoder" --layer "$layer" --length 256 --context 512 --activation-file "$artifact/natural257.pt" --prefill-repeats 3 --decode-repeats 20 --output "$artifact/natural_final_$layer.json"
done
for layer in 0 3; do
    run_job "functional_$layer" timeout 1800 python -m "$module.stress_optimized_decoder" --layer "$layer" --seed 913 --repeats 3 --clean-performance --output "$artifact/functional_$layer.json"
done
export QWEN_DECODER=fused
for layer in 0 3; do
    for length in 127 128 129 257 4095 4096 4097; do
        run_job "baseline_l${layer}_s${length}" timeout 1800 python -m "$module.run_decoder" --layer "$layer" --length "$length" --context 8192 --prefill-repeats 3 --decode-repeats 20 --output "$artifact/baseline_l${layer}_s${length}.json"
    done
done
export QWEN_DECODER=optimized
for layer in 0 3; do
    run_job "profile_$layer" timeout 1800 bash models/autoports/qwen_qwen3_6_35b_a3b/tests/collect_optimized_perf.sh "$layer" "$artifact/tracy/layer_$layer"
done
run_job profile_bfp4_candidate timeout 1800 bash models/autoports/qwen_qwen3_6_35b_a3b/tests/collect_optimized_perf.sh 3 "$artifact/tracy/bfp4_candidate" '{"expert_dtype":"bfp4","expert_fidelity":"LoFi","sparse_block_w":[32,16]}'
