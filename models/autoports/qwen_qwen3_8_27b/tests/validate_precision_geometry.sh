#!/usr/bin/env bash
set -euo pipefail
root=models/autoports/qwen_qwen3_8_27b
out=$root/doc/datatype_sweep
for layer in ${LAYERS:-0 3}; do
    if [[ ${SKIP_MICRO:-0} != 1 ]]; then
    bash "$root/tests/optimized_multichip_env.sh" python -m models.autoports.qwen_qwen3_8_27b.tests.probe_bfp8_geometry \
        --layer "$layer" --output "$out/bfp8_geometry_layer${layer}.json" > "$out/bfp8_geometry_layer${layer}.log" 2>&1
    fi
    for seq in ${SEQUENCES:-1 33 4095 4096 4097}; do
        bash "$root/tests/optimized_multichip_env.sh" python -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder \
            --layer "$layer" --seq "$seq" --steps 4 --policy '{"attention_dtype":"bfloat8_b","mlp_dtype":"bfloat8_b","down_dtype":"bfloat8_b"}' \
            --baseline-policy '{"attention_dtype":"bfloat8_b","mlp_dtype":"bfloat8_b","down_dtype":"bfloat8_b"}' \
            --output "$out/geometry_l${layer}_s${seq}.json" > "$out/geometry_l${layer}_s${seq}.log" 2>&1
        echo "GEOMETRY_PASS layer=$layer seq=$seq"
    done
    for batch in 8 16 32; do
        bash "$root/tests/optimized_multichip_env.sh" python -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder \
            --layer "$layer" --seq 33 --steps 4 --batch "$batch" --policy '{"attention_dtype":"bfloat8_b","mlp_dtype":"bfloat8_b","down_dtype":"bfloat8_b"}' \
            --baseline-policy '{"attention_dtype":"bfloat8_b","mlp_dtype":"bfloat8_b","down_dtype":"bfloat8_b"}' \
            --output "$out/geometry_l${layer}_b${batch}.json" > "$out/geometry_l${layer}_b${batch}.log" 2>&1
        echo "GEOMETRY_PASS layer=$layer batch=$batch"
    done
    bash "$root/tests/optimized_multichip_env.sh" python -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder \
        --layer "$layer" --seq 33 --steps 8 --policy '{"ccl_dtype":"bfloat16"}' \
        --output "$out/ccl_bf16_l${layer}.json" > "$out/ccl_bf16_l${layer}.log" 2>&1
    echo "CCL_BF16_PASS layer=$layer"
done
