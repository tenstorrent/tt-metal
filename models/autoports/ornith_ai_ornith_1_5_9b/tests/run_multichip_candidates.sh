#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
D=models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder
G=models/autoports/ornith_ai_ornith_1_5_9b/generated/multichip
for layer in 0 3; do
  for mode in baseline sharded replicated gather scatter both direct hybrid; do
    if [[ -s "$D/perf_${mode}_$layer.json" ]]; then continue; fi
    opts=()
    if [[ "$mode" == baseline ]]; then opts+=(--baseline); fi
    if [[ "$mode" == replicated || "$mode" == direct || "$mode" == hybrid ]]; then opts+=(--layout replicated); fi
    if [[ "$mode" == sharded || "$mode" == replicated ]]; then opts+=(--candidate separate); fi
    if [[ "$mode" == gather || "$mode" == scatter || "$mode" == both || "$mode" == direct || "$mode" == hybrid ]]; then opts+=(--candidate "$mode"); fi
    timeout 600 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe \
      --layer "$layer" --lengths 1024 --readers 4 --cores 32 --trace --benchmark --steps 8 \
      --reference "$G/perf_$layer.pt" --output "$D/perf_${mode}_$layer.json" "${opts[@]}" \
      > "$D/perf_${mode}_$layer.log" 2>&1
  done
done
