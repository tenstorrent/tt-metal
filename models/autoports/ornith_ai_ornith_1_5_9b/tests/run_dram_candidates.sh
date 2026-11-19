#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail
art=models/autoports/ornith_ai_ornith_1_5_9b/doc/fused_decoder
for readers in 1 2 4; do
    for cores in 32 64; do
        for block in 1 0; do
            name=dram_r${readers}_c${cores}_b${block}_linear
            timeout 1200 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.dram_fusion_candidate --readers "$readers" --cores "$cores" --block "$block" --layer 0 --lengths 1024 --benchmark --output "$art/$name.json" > "$art/$name.log" 2>&1
            status=$?
            echo "$name $status"
            if [[ "$status" == 124 || "$status" == 137 ]]; then exit "$status"; fi
        done
    done
done
