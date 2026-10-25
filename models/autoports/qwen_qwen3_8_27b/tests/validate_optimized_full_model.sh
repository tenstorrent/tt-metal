#!/usr/bin/env bash
# Serialized post-benchmark diagnostics; never mix Watcher and profiler.
set -euo pipefail
root=models/autoports/qwen_qwen3_8_27b
out=$root/doc/optimized_full_model
for batch in 1 32; do
    bash "$root/tests/optimized_multichip_env.sh" env TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGS_PATH="/workspace/$out/watcher_b${batch}" \
        python "$root/tests/probe_buffered_output.py" --batch "$batch" \
        --output "$out/watcher_buffered_b${batch}.json" > "$out/watcher_buffered_b${batch}.log" 2>&1
    echo "WATCHER_PASS batch=$batch"
done
bash "$root/tests/optimized_multichip_env.sh" python "$root/tests/probe_external_prepare.py" \
    --batch 32 --output "$out/external_prepare_reduced_b32.json" > "$out/external_prepare_reduced_b32.log" 2>&1
bash "$root/tests/optimized_multichip_env.sh" python "$root/tests/probe_mixed_slots.py" \
    --output "$out/mixed_slots.json" > "$out/mixed_slots.log" 2>&1
for batch in 1 32; do
    bash "$root/tests/optimized_multichip_env.sh" bash "$root/tests/profile_optimized_full_model.sh" "$batch" \
        > "$out/profile_b${batch}.log" 2>&1
    echo "PROFILE_COMPLETE batch=$batch"
done
