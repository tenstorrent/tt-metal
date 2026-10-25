#!/usr/bin/env bash
set -euo pipefail
base=models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model
for spec in '8192 1 4 8' '6144 1 4 8' '6144 2 10 8' '6144 3 10 16' '6144 3 16 10' '3072 3 20 8' '3072 3 32 5' '12288 3 5 8'; do
    read -r chunk readers block cores <<< "$spec"
    name=head_c${chunk}_r${readers}_k${block}_s${cores}
    python -m models.autoports.qwen_qwen3_8_27b.tests.probe_head_geometry --chunk "$chunk" --readers "$readers" --block "$block" --cores "$cores" --output "$base/$name.json" > "$base/$name.log" 2>&1
    echo "HEAD_CANDIDATE_DONE $name"
done
