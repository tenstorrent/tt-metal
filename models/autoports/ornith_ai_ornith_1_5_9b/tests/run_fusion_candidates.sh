#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
art=models/autoports/ornith_ai_ornith_1_5_9b/doc/fused_decoder
for candidate in separate_activation separate_binary interleaved_decode packed_prefill; do
    timeout 1200 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.fusion_candidate --candidate "$candidate" --layer 0 --lengths 1024 --benchmark --output "$art/${candidate}_linear.json" > "$art/${candidate}_linear.log" 2>&1
    echo "PASS $candidate"
done
