#!/usr/bin/env bash
set -euo pipefail
root=models/autoports/qwen_qwen3_8_27b
out=$root/doc/datatype_sweep
for dtype in bfloat4_b bfloat16; do
    for seq in 33 4097; do
        bash "$root/tests/optimized_multichip_env.sh" python -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder \
            --layer 3 --seq "$seq" --steps 4 --multichip-only --policy "{\"kv_dtype\":\"$dtype\"}" \
            --output "$out/kv_${dtype}_s${seq}.json" > "$out/kv_${dtype}_s${seq}.log" 2>&1
        echo "KV_RUNTIME_SMOKE_PASS dtype=$dtype seq=$seq"
    done
done
