#!/bin/bash

TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache

set -eo pipefail

run_python_model_tests_grayskull() {
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large*matmul* -k in0_L1-in1_L1-bias_L1-out_L1
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large*bmm* -k in0_L1-in1_L1-out_L1
    # Tests for mixed precision (sweeps combos of bfp8_b/bfloat16 dtypes for fused_qkv_bias and ff1_bias_gelu matmul and pre_softmax_bmm)
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_matmul -k "fused_qkv_bias and batch_9 and L1"
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_matmul -k "ff1_bias_gelu and batch_9 and DRAM"
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_bmm -k "pre_softmax_bmm and batch_9"

    # BERT TMs
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_split_query_key_value_and_split_heads.py -k "in0_L1-out_L1 and batch_9"
    pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_concatenate_heads.py -k "in0_L1-out_L1 and batch_9"

    # Test program cache
    pytest models/experimental/bert_large_performant/unit_tests/ -k program_cache

    # Fused ops unit tests
    pytest models/experimental/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_ln.py -k "in0_L1-out_L1 and batch_9"
    pytest models/experimental/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_softmax.py -k "in0_L1 and batch_9"

    # Falcon tests
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_128 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_512 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_attn_matmul.py
}

run_python_model_tests_wormhole_b0() {
    # Falcon tests
    # attn_matmul_from_cache is currently not used in falcon7b
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_attn_matmul.py -k "not attn_matmul_from_cache"
    # higher sequence lengths and different formats trigger memory issues
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_128 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
    pytest models/demos/wormhole/resnet50/tests/test_resnet50_functional.py -k "pretrained_weight_false"

    # Unet Shallow
    pytest -svv models/experimental/functional_unet/tests/test_unet_model.py

    # Mobilenetv2git
    pytest -svv models/demos/mobilenetv2/tests/pcc/test_mobilenetv2.py

    # ViT-base
    pytest -svv models/demos/wormhole/vit/tests/test_ttnn_optimized_sharded_vit_wh.py


    # Llama3.1-8B
    llama8b=meta-llama/Llama-3.1-8B-Instruct

    # Run all Llama3 tests for 8B - dummy weights with tight PCC check
    tt_cache=$TT_CACHE_HOME/$llama8b
    HF_MODEL=$llama8b TT_CACHE_PATH=$tt_cache pytest models/tt_transformers/tests/test_model.py -k "quick" ; fail+=$?
    echo "LOG_METAL: Llama3 tests for $llama8b completed"

}

run_python_model_tests_slow_runtime_mode_wormhole_b0() {
    # Unet Shallow
    export TTNN_CONFIG_OVERRIDES='{
        "enable_fast_runtime_mode": false,
        "enable_comparison_mode": true,
        "comparison_mode_should_raise_exception": true,
        "comparison_mode_pcc": 0.998
    }'
    pytest -svv models/experimental/functional_unet/tests/test_unet_model.py
}

run_python_model_tests_blackhole() {
    SD_HF_DOWNLOAD_OVERRIDE=1 pytest models/demos/blackhole/stable_diffusion/tests --ignore=models/demos/blackhole/stable_diffusion/tests/test_perf.py

    # Llama3.1-8B
    llama8b=meta-llama/Llama-3.1-8B-Instruct
    # Run all Llama3 tests for 8B - dummy weights with tight PCC check
    for hf_model in "$llama8b"; do
        tt_cache=$TT_CACHE_HOME/$hf_model
        HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest models/tt_transformers/tests/test_model.py -k "quick" ; fail+=$?
        echo "LOG_METAL: Llama3 tests for $hf_model completed"
    done

    pytest models/demos/wormhole/resnet50/tests/test_resnet50_functional.py
    pytest models/experimental/functional_unet/tests/test_unet_model.py
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/
}

run_python_model_tests_slow_runtime_mode_blackhole() {
    # Unet Shallow
    export TTNN_CONFIG_OVERRIDES='{
        "enable_fast_runtime_mode": false,
        "enable_comparison_mode": true,
        "comparison_mode_should_raise_exception": true,
        "comparison_mode_pcc": 0.998
    }'
    pytest -svv models/experimental/functional_unet/tests/test_unet_model.py
}
