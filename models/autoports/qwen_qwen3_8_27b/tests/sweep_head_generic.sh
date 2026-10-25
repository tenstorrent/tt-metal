#!/usr/bin/env bash
set -euo pipefail
base=models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model
for spec in '4 11 10 3' '5 11 10 3' '8 11 10 3' '10 11 10 3' '8 10 10 4' '8 10 8 1'; do
    read -r block gx gy sub <<< "$spec"
    name=head_generic_k${block}_g${gx}x${gy}_sub${sub}
    python -m models.autoports.qwen_qwen3_8_27b.tests.probe_head_geometry --generic --block "$block" --grid-x "$gx" --grid-y "$gy" --subblock "$sub" --output "$base/$name.json" > "$base/$name.log" 2>&1
    echo "HEAD_CANDIDATE_DONE $name"
done
