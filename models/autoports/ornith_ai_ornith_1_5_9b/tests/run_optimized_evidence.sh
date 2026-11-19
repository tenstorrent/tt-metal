#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=models/autoports/ornith_ai_ornith_1_5_9b
art="$root/doc/optimized_decoder"
module=models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized
for layer in 0 3; do
    kind=linear
    if [[ "$layer" == 3 ]]; then kind=full; fi
    echo "START final $kind"
    timeout 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m "$module" --layer "$layer" --lengths 1,2,3,4,31,32,33,1023,1024,1025,2051,17 --resume --batch 32 --output "$art/${kind}_final.json" > "$art/${kind}_final.log" 2>&1
    echo "PASS final $kind"
done
mkdir -p "$art/trace_alloc_tracking"
echo 'START positive control'
timeout 600 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m "$module" --layer 0 --lengths 32 --positive-control --output "$art/trace_alloc_tracking/positive_control.json" > "$art/trace_alloc_tracking/positive_control.log" 2>&1
echo 'PASS positive control'
timeout 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m pytest "$root/tests/test_optimized_decoder.py" -q -s --timeout 1500 > "$art/real_weight_pytest.log" 2>&1
echo 'PASS real-weight pytest'
for layer in 0 3; do
    kind=linear
    if [[ "$layer" == 3 ]]; then kind=full; fi
    echo "START watcher $kind"
    timeout 1800 env TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGS_PATH="$PWD/$art/watcher/$kind" python -m "$module" --layer "$layer" --lengths 1,2,3,4,31,32,33,1023,1024,1025,2051,17 --resume --batch 32 --output "$art/${kind}_watcher.json" > "$art/${kind}_watcher.log" 2>&1
    echo "PASS watcher $kind"
done
bash "$root/tests/run_optimized_profiles.sh"
echo EVIDENCE_QUEUE_PASS
