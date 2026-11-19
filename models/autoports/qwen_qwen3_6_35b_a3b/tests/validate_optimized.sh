#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Proposed serial acceptance queue. Run once only after selecting final defaults.
set -euo pipefail
export QWEN_DECODER=optimized
artifact=models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_decoder
module=models.autoports.qwen_qwen3_6_35b_a3b.tests
waiter="${TT_MODEL_BRINGUP_ROOT:?Source the bringup environment}/scripts/wait_for"
mkdir -p "$artifact/logs"
started=0
[[ -n "${VALIDATION_START:-}" ]] || started=1
run_job() {
    [[ ! -f "$artifact/STOP_QUEUE" ]] || exit 3
    local label="$1"
    shift
    if [[ "$started" == 0 ]]; then
        [[ "$label" == "$VALIDATION_START" ]] || return 0
        started=1
    fi
    if fuser /dev/tenstorrent/0; then
        echo "Device 0 is already owned; refusing to launch $label" >&2
        exit 3
    fi
    local command
    printf -v command '%q ' "$@"
    "$waiter" --cmd "$command" --file "$artifact/logs/$label.log" --timeout 1830 --tail 8
}
export TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
for layer in 0 3 34; do
    seed=913
    [[ "$layer" != 34 ]] || seed=154
    run_job "stress_$layer" timeout 1800 python -m "$module.stress_optimized_decoder" --layer "$layer" --seed "$seed" --repeats 3 --output "$artifact/stress_$layer.json"
    run_job "stress_extended_$layer" timeout 1800 python -m "$module.stress_optimized_decoder" --layer "$layer" --seed "$seed" --repeats 3 --decode-length 1024 --output "$artifact/stress_extended_$layer.json"
done
for layer in 0 3; do
    run_job "battery_$layer" timeout 1800 python -m "$module.battery" --layer "$layer" --mode all --batch 32 --output "$artifact/battery_$layer.json"
    run_job "trace_reuse_watcher_$layer" timeout 1800 env TT_METAL_WATCHER=10 "TT_METAL_LOGS_PATH=$PWD/$artifact/watcher/layer_$layer" python -m "$module.trace_reuse" --layer "$layer" --output "$artifact/trace_reuse_$layer.json"
    run_job "context_32768_$layer" timeout 1800 python -m "$module.long_context" --layer "$layer" --length 32768 --output "$artifact/context_32768_$layer.json"
done
run_job page_extent_regression timeout 1800 python -m "$module.page_extent_regression"
run_job padded_trace_regression timeout 1800 python -m "$module.padded_trace_regression"
python -m "$module.audit_watcher"
run_job real_weight_pytest timeout 1800 pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_optimized_decoder.py -q "--basetemp=$PWD/$artifact/pytest" "--junitxml=$artifact/real_weight_pytest.xml"
# Clean latency measurements are separate from Watcher and allocation diagnostics.
unset TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
for layer in 0 3; do
    run_job "final_$layer" timeout 1800 python -m "$module.run_decoder" --layer "$layer" --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 --output "$artifact/final_$layer.json"
    for length in 127 128 129 257 4095 4096 4097; do
        run_job "final_l${layer}_s${length}" timeout 1800 python -m "$module.run_decoder" --layer "$layer" --length "$length" --context 8192 --prefill-repeats 3 --decode-repeats 20 --output "$artifact/final_l${layer}_s${length}.json"
    done
done
