#!/usr/bin/env bash
set -euo pipefail
root=models/autoports/qwen_qwen3_8_27b
out=$root/doc/datatype_sweep
baseline='{"kv_dtype":"bfloat16","attention_dtype":"bfloat4_b","mlp_dtype":"bfloat4_b","down_dtype":"bfloat4_b","fidelity":"LoFi","sdpa_chunk":256,"sdpa_cores":16}'
for length in 1 33 255 4095 4096 4097; do
    extra=()
    if [[ $length == 255 ]]; then extra=(--remap); fi
    bash "$root/tests/optimized_multichip_env.sh" python -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder \
        --layer 3 --seq "$length" --steps 4 --capacity-alignment 262144 \
        --policy '{"kv_dtype":"bfloat16"}' --baseline-policy "$baseline" "${extra[@]}" \
        --output "$out/kv_bf16_native_s${length}.json" > "$out/kv_bf16_native_s${length}.log" 2>&1
    echo "BF16_NATIVE_BOUNDARY_COMPLETE $length"
done
for batch in 4 8 16 32; do
    bash "$root/tests/optimized_multichip_env.sh" python -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder \
        --layer 3 --seq 33 --steps 4 --batch "$batch" --ragged --capacity-alignment 32768 \
        --policy '{"kv_dtype":"bfloat16"}' --baseline-policy "$baseline" \
        --output "$out/kv_bf16_b${batch}.json" > "$out/kv_bf16_b${batch}.log" 2>&1
    echo "BF16_BATCH_COMPLETE $batch"
done
bash "$root/tests/optimized_multichip_env.sh" python -m models.autoports.qwen_qwen3_8_27b.tests.probe_precision_stack \
    --config "$out/configs/kv_bf16.json" --context 262144 --native-end \
    --output "$out/kv_bf16_native_replay.json" > "$out/kv_bf16_native_replay.log" 2>&1
echo 'BF16_NATIVE_REPLAY_COMPLETE'
