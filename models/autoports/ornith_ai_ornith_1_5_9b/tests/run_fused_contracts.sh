#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=models/autoports/ornith_ai_ornith_1_5_9b
art="$root/doc/fused_decoder"
module=models.autoports.ornith_ai_ornith_1_5_9b.tests
for layer in 0 3; do
    kind=linear
    if [[ "$layer" == 3 ]]; then kind=full; fi
    echo "START paired $kind"
    timeout 1800 python -m "$module.compare_fused" --layer "$layer" --output "$art/${kind}_paired.json" > "$art/${kind}_paired.log" 2>&1
    timeout 1800 python -m "$module.compare_fused" --layer "$layer" --lengths 128 --steps 128 --output "$art/${kind}_trajectory128.json" > "$art/${kind}_trajectory128.log" 2>&1
    echo "PASS paired and trajectory $kind"
    for mode in baseline default; do
        extra=()
        if [[ "$mode" == baseline ]]; then extra=(--baseline); fi
        for length in 128 1023 1024 1025 8192; do
            name=${mode}_${kind}_s${length}
            timeout 1800 python -m "$module.validate_fused" "${extra[@]}" --layer "$layer" --lengths "$length" --benchmark --output "$art/$name.json" > "$art/$name.log" 2>&1
            echo "PASS $name"
        done
    done
    echo "START long context $kind"
    timeout 5400 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m "$module.long_context_fused" --layer "$layer" --output "$art/${kind}_long.json" > "$art/${kind}_long.log" 2>&1
    echo "PASS long context $kind"
done
echo FUSED_CONTRACTS_PASS
