#!/bin/bash

TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache

set -eo pipefail

run_python_model_tests_grayskull() {
    # Run all model tests - calls all individual group functions
    run_python_model_tests_bert_matmul_grayskull
    run_python_model_tests_bert_bmm_grayskull
    run_python_model_tests_bert_mixed_precision_grayskull
    run_python_model_tests_bert_tms_grayskull
    run_python_model_tests_bert_program_cache_grayskull
    run_python_model_tests_bert_fused_ops_grayskull
    run_python_model_tests_falcon_grayskull
}

run_python_model_tests_bert_matmul_grayskull() {
    # BERT matmul tests
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large*matmul* -k in0_L1-in1_L1-bias_L1-out_L1
}

run_python_model_tests_bert_bmm_grayskull() {
    # BERT BMM tests
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large*bmm* -k in0_L1-in1_L1-out_L1
}

run_python_model_tests_bert_mixed_precision_grayskull() {
    # Tests for mixed precision (sweeps combos of bfp8_b/bfloat16 dtypes for fused_qkv_bias and ff1_bias_gelu matmul and pre_softmax_bmm)
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_matmul -k "fused_qkv_bias and batch_9 and L1"
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_matmul -k "ff1_bias_gelu and batch_9 and DRAM"
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_bmm -k "pre_softmax_bmm and batch_9"
}

run_python_model_tests_bert_tms_grayskull() {
    # BERT TMs (transformer modules)
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_split_query_key_value_and_split_heads.py -k "in0_L1-out_L1 and batch_9"
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_concatenate_heads.py -k "in0_L1-out_L1 and batch_9"
}

run_python_model_tests_bert_program_cache_grayskull() {
    # Test program cache
    pytest models/experimental/bert_large_performant/unit_tests/ -k program_cache
}

run_python_model_tests_bert_fused_ops_grayskull() {
    # Fused ops unit tests
    pytest models/experimental/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_ln.py -k "in0_L1-out_L1 and batch_9"
    pytest models/experimental/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_softmax.py -k "in0_L1 and batch_9"
}

run_python_model_tests_falcon_grayskull() {
    # Falcon tests
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_128 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_512 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_attn_matmul.py
}

# Stubs for grayskull (tests not available for this architecture)
run_python_model_tests_deepseek_grayskull() {
    echo "DeepSeek tests not available for Grayskull"
}

run_python_model_tests_resnet_unet_grayskull() {
    echo "ResNet/Unet tests not available for Grayskull"
}

run_python_model_tests_mobilenet_grayskull() {
    echo "MobileNet tests not available for Grayskull"
}

run_python_model_tests_vit_grayskull() {
    echo "ViT tests not available for Grayskull"
}

run_python_model_tests_llama_grayskull() {
    echo "Llama tests not available for Grayskull"
}

run_python_model_tests_slow_runtime_grayskull() {
    echo "Slow runtime tests not available for Grayskull"
}

run_python_model_tests_wormhole_b0() {
    # Run all model tests - calls all individual group functions
    run_python_model_tests_deepseek_wormhole_b0
    run_python_model_tests_falcon_wormhole_b0
    run_python_model_tests_resnet_unet_wormhole_b0
    run_python_model_tests_mobilenet_wormhole_b0
    run_python_model_tests_vit_wormhole_b0
    run_python_model_tests_llama_wormhole_b0
    run_python_model_tests_slow_runtime_wormhole_b0
}

run_python_model_tests_deepseek_wormhole_b0() {
    # DeepSeekV3
    uv pip install -r models/demos/deepseek_v3/reference/deepseek/requirements.txt
    MESH_DEVICE=AUTO pytest models/demos/deepseek_v3/tests/unit --timeout 60 --durations=0
}

run_python_model_tests_falcon_wormhole_b0() {
    # Falcon tests
    # attn_matmul_from_cache is currently not used in falcon7b
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_attn_matmul.py -k "not attn_matmul_from_cache"
    # higher sequence lengths and different formats trigger memory issues
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_128 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
}

run_python_model_tests_resnet_unet_wormhole_b0() {
    # ResNet50 and Unet tests
    pytest models/demos/vision/classification/resnet50/wormhole/tests/test_resnet50_functional.py -k "pretrained_weight_false"
    pytest -svv models/experimental/functional_unet/tests/test_unet_model.py
}

run_python_model_tests_mobilenet_wormhole_b0() {
    # Mobilenetv2
    pytest -svv models/demos/vision/classification/mobilenetv2/tests/pcc/test_mobilenetv2.py
}

run_python_model_tests_vit_wormhole_b0() {
    # ViT-base
    pytest -svv models/demos/vision/classification/vit/wormhole/tests/test_ttnn_optimized_sharded_vit_wh.py
}

run_python_model_tests_llama_wormhole_b0() {
    # Llama3.1-8B
    llama8b=meta-llama/Llama-3.1-8B-Instruct

    # Run all Llama3 tests for 8B - dummy weights with tight PCC check
    tt_cache=$TT_CACHE_HOME/$llama8b
    HF_MODEL=$llama8b TT_CACHE_PATH=$tt_cache pytest models/tt_transformers/tests/test_model.py -k "quick" ; fail+=$?
    echo "LOG_METAL: Llama3 tests for $llama8b completed"
}

run_python_model_tests_slow_runtime_wormhole_b0() {
    # Unet Shallow in slow runtime mode
    export TTNN_CONFIG_OVERRIDES='{
        "enable_fast_runtime_mode": false,
        "enable_comparison_mode": true,
        "comparison_mode_should_raise_exception": true,
        "comparison_mode_pcc": 0.998
    }'
    pytest -svv models/experimental/functional_unet/tests/test_unet_model.py
}

# Alias for backward compatibility
run_python_model_tests_slow_runtime_mode_wormhole_b0() {
    run_python_model_tests_slow_runtime_wormhole_b0
}

run_python_model_tests_blackhole() {
    # Run all model tests - calls all individual group functions
    run_python_model_tests_stable_diffusion_blackhole
    run_python_model_tests_llama_blackhole
    run_python_model_tests_resnet_unet_blackhole
    run_python_model_tests_slow_runtime_blackhole
}

run_python_model_tests_stable_diffusion_blackhole() {
    # Stable Diffusion tests
    SD_HF_DOWNLOAD_OVERRIDE=1 pytest models/demos/vision/generative/stable_diffusion/blackhole/tests --timeout 420 --ignore=models/demos/vision/generative/stable_diffusion/blackhole/tests/test_perf.py
}

run_python_model_tests_llama_blackhole() {
    # Llama3.1-8B
    llama8b=meta-llama/Llama-3.1-8B-Instruct
    # Run all Llama3 tests for 8B - dummy weights with tight PCC check
    for hf_model in "$llama8b"; do
        tt_cache=$TT_CACHE_HOME/$hf_model
        HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest models/tt_transformers/tests/test_model.py -k "quick" --timeout 360 ; fail+=$?
        echo "LOG_METAL: Llama3 tests for $hf_model completed"
    done
}

run_python_model_tests_resnet_unet_blackhole() {
    # ResNet50 and Unet tests
    pytest models/demos/vision/classification/resnet50/wormhole/tests/test_resnet50_functional.py --timeout 300
    pytest models/experimental/functional_unet/tests/test_unet_model.py --timeout 90
}

run_python_model_tests_slow_runtime_blackhole() {
    # Unet Shallow in slow runtime mode
    export TTNN_CONFIG_OVERRIDES='{
        "enable_fast_runtime_mode": false,
        "enable_comparison_mode": true,
        "comparison_mode_should_raise_exception": true,
        "comparison_mode_pcc": 0.998
    }'
    pytest -svv models/experimental/functional_unet/tests/test_unet_model.py --timeout 90
}

# Alias for backward compatibility
run_python_model_tests_slow_runtime_mode_blackhole() {
    run_python_model_tests_slow_runtime_blackhole
}

# Stubs for blackhole (not all test groups apply)
run_python_model_tests_deepseek_blackhole() {
    echo "DeepSeek tests not available for Blackhole"
}

run_python_model_tests_falcon_blackhole() {
    echo "Falcon tests not available for Blackhole as separate group"
}

run_python_model_tests_mobilenet_blackhole() {
    echo "MobileNet tests not available for Blackhole"
}

run_python_model_tests_vit_blackhole() {
    echo "ViT tests not available for Blackhole"
}
