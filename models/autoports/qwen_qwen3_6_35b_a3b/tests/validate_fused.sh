#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Serial chip-0 validation. STOP_QUEUE is honored only between completed jobs.
set -euo pipefail
export QWEN_DECODER=fused
artifact=models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder
module=models.autoports.qwen_qwen3_6_35b_a3b.tests
waiter="$TT_MODEL_BRINGUP_ROOT/scripts/wait_for"
run_job() {
    [[ ! -f "$artifact/STOP_QUEUE" ]] || exit 3
    local label="$1"
    shift
    if fuser /dev/tenstorrent/0; then
        echo "Device 0 is already owned; refusing to launch $label" >&2
        exit 3
    fi
    local deadline=1830
    [[ "$label" != linear_long ]] || deadline=7230
    "$waiter" --cmd "$*" --file "$artifact/logs/$label.log" --timeout "$deadline" --tail 8
}
export TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
for layer in 0 3; do
    run_job "stress_$layer" "timeout 1800 python -m $module.stress_fused_decoder --layer $layer --seed 913 --repeats 3 --output $artifact/stress_$layer.json"
    run_job "stress_extended_$layer" "timeout 1800 python -m $module.stress_fused_decoder --layer $layer --seed 913 --repeats 3 --decode-length 1024 --output $artifact/stress_extended_$layer.json"
    run_job "stress_endurance_$layer" "timeout 1800 python -m $module.stress_fused_decoder --layer $layer --seed 913 --repeats 3 --prompt-length 128 --decode-length 8192 --continuation-length 33 --output $artifact/stress_endurance_$layer.json"
    run_job "battery_$layer" "timeout 1800 python -m $module.battery --layer $layer --mode all --batch 32 --output $artifact/battery_$layer.json"
    run_job "trace_reuse_watcher_$layer" "timeout 1800 env TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$PWD/$artifact/watcher/layer_$layer python -m $module.trace_reuse --layer $layer --output $artifact/trace_reuse_$layer.json"
done
# Known linear-attention regressions: fresh HF references, never cached A/B oracles.
# These run under the same allocation and program-cache tracking as seed913 above.
for seed in 42 154; do
    run_job "stress_endurance_seed${seed}_0" "timeout 1800 python -m $module.stress_fused_decoder --layer 0 --seed $seed --repeats 3 --prompt-length 128 --decode-length 8192 --continuation-length 33 --output $artifact/stress_endurance_seed${seed}_0.json"
done
# Required late-layer regression; fresh HF and the same tracking/cache environment.
run_job "stress_representative_34_seed154" "timeout 1800 python -m $module.stress_fused_decoder --layer 34 --seed 154 --repeats 3 --prompt-length 128 --decode-length 8192 --continuation-length 33 --output $artifact/stress_representative_34_seed154.json"
run_job linear_long "timeout 7200 python -m $module.long_context --layer 0 --length 262144 --oracle-cache $artifact/oracle_cache/linear_262144.pt --output $artifact/linear_long.json"
run_job full_long "timeout 1800 python -m $module.long_tail_regression"
run_job page_extent_regression "timeout 1800 python -m $module.page_extent_regression"
run_job padded_trace_regression "timeout 1800 python -m $module.padded_trace_regression"
run_job synthetic_pytest "timeout 1800 pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_fused_decoder.py -q --basetemp=$PWD/$artifact/pytest --junitxml=$artifact/synthetic_pytest.xml"
unset TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
for layer in 0 3; do
    run_job "final_$layer" "timeout 1800 python -m $module.run_decoder --layer $layer --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 --compare-functional --output $artifact/final_$layer.json"
    for length in 127 128 129 257 4097; do
        run_job "final_l${layer}_s${length}" "timeout 1800 python -m $module.run_decoder --layer $layer --length $length --context 8192 --compare-functional --output $artifact/final_l${layer}_s${length}.json"
    done
done
python -m "$module.audit_fused_source"
python -m "$module.audit_watcher"
