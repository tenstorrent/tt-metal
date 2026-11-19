#!/usr/bin/env bash
set -euo pipefail
D=models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder
G=models/autoports/ornith_ai_ornith_1_5_9b/generated/multichip
for layer in 0 3; do
  for mode in baseline sharded; do
    opts=()
    if [[ "$mode" == baseline ]]; then opts+=(--baseline); fi
    timeout 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
      python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe \
      --layer "$layer" --lengths 1,2,3,4,31,32,33,1023,1024,1025,2051 --trace --steps 8 \
      --reference "$G/contracts_$layer.pt" --output "$D/contracts_${mode}_$layer.json" "${opts[@]}" \
      > "$D/contracts_${mode}_$layer.log" 2>&1
  done
done
