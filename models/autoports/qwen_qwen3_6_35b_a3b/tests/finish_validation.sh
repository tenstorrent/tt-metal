#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Serial device queue. Touch doc/functional_decoder/STOP_QUEUE to stop between jobs.
set -euo pipefail
artifact=models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder
waiter="$TT_MODEL_BRINGUP_ROOT/scripts/wait_for"
if [[ $# -gt 0 ]]; then
    "$waiter" --pid "$1" --file "$artifact/logs/full_long_fixed.log" --timeout 14430 --tail 3
fi
[[ -f "$artifact/full_long.json" ]]
run_job() {
    [[ ! -f "$artifact/STOP_QUEUE" ]] || exit 3
    local label="$1"
    shift
    "$waiter" --cmd "$*" --file "$artifact/logs/$label.log" --timeout 1830 --tail 12
}
layers=(0 3)
[[ "${FUNCTIONAL_FULL_ONLY:-0}" == 1 ]] && layers=(3)
for layer in "${layers[@]}"; do
    run_job "trace_reuse_watcher_$layer" "timeout 1800 env -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$PWD/$artifact/watcher/layer_$layer python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.trace_reuse --layer $layer --output $artifact/trace_reuse_$layer.json"
done
run_job full_resume_fixed "timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.battery --layer 3 --mode resume --output $artifact/full_resume_fixed.json"
if [[ "${FUNCTIONAL_FULL_ONLY:-0}" != 1 ]]; then
    run_job linear_mixed_batch "timeout 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.battery --layer 0 --mode batch --batch 32 --output $artifact/linear_mixed_batch.json"
fi
run_job full_shared_pool "timeout 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.battery --layer 3 --mode batch --batch 32 --output $artifact/full_shared_pool.json"
run_job synthetic_pytest "timeout 1800 pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py -q --basetemp=$PWD/$artifact/pytest --junitxml=$artifact/synthetic_pytest.xml"
for layer in "${layers[@]}"; do
    run_job "partition_$layer" "timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.battery --layer $layer --mode boundaries --output $artifact/partition_$layer.json"
done
run_job padded_trace_regression "timeout 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.padded_trace_regression"
