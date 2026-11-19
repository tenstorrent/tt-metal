#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=models/autoports/ornith_ai_ornith_1_5_9b
art="$root/doc/optimized_decoder"
if [[ "${1:-}" != --skip-profiles ]]; then bash "$root/tests/run_optimized_profiles.sh"; fi
for layer in 0 3; do
    for precision in bfp8 bfp4; do
        test ! -e "$art/STOP" || exit 0
        extra=()
        if [[ "$precision" == bfp4 ]]; then extra=(--bfp4); fi
        echo "START role sweep $precision layer $layer"
        timeout 1800 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.optimized_matmul_sweep \
            --layer "$layer" "${extra[@]}" --output "$art/roles_${precision}_l${layer}.json" \
            > "$art/roles_${precision}_l${layer}.log" 2>&1
        echo "PASS role sweep $precision layer $layer"
    done
done
