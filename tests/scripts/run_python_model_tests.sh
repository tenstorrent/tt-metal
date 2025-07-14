#!/bin/bash

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

    pytest tests/ttnn/integration_tests/resnet/test_ttnn_functional_resnet50.py -k "pretrained_weight_false"
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
    pytest tests/ttnn/integration_tests/resnet/test_ttnn_functional_resnet50.py -k "pretrained_weight_false"

    # Yolov4
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py -k "pretrained_weight_false"

    # Yolov8
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/ttnn/integration_tests/yolov8x/test_yolov8x.py::test_yolov8x_640

    # Yolov8s
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/ttnn/integration_tests/yolov8s/test_yolov8s.py::test_yolov8s_640

    # Unet Shallow
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -svv models/experimental/functional_unet/tests/test_unet_model.py

    # Mamba
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -svv models/demos/wormhole/mamba/tests/test_residual_block.py -k "pretrained_weight_false"

    # Mobilenetv2git
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -svv tests/ttnn/integration_tests/mobilenetv2/test_mobilenetv2.py

    #Yolov10
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -svv tests/ttnn/integration_tests/yolov10x/test_ttnn_yolov10x.py::test_yolov10x

    #Yolov7
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -svv tests/ttnn/integration_tests/yolov7/test_ttnn_yolov7.py

    # ViT-base
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -svv tests/ttnn/integration_tests/vit/test_ttnn_optimized_sharded_vit_wh.py

    # Llama3.1-8B
    llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/
    # Llama3.2-1B
    llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
    # Llama3.2-3B
    llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
    # Llama3.2-11B
    llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

    # Run all Llama3 tests for 8B, 1B, and 3B weights - dummy weights with tight PCC check
    for llama_dir in  "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
        LLAMA_DIR=$llama_dir WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/tt_transformers/tests/test_model.py -k "quick" ; fail+=$?
        echo "LOG_METAL: Llama3 tests for $llama_dir completed"
    done

    # Mistral-7B-v0.3
    mistral_weights=mistralai/Mistral-7B-Instruct-v0.3
    HF_MODEL=$mistral_weights WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/tt_transformers/tests/test_model.py -k "quick" ; fail+=$?
}

run_python_model_tests_slow_runtime_mode_wormhole_b0() {
    # Unet Shallow
    export TTNN_CONFIG_OVERRIDES='{
        "enable_fast_runtime_mode": false,
        "enable_comparison_mode": true,
        "comparison_mode_should_raise_exception": true,
        "comparison_mode_pcc": 0.998
    }'
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -svv models/experimental/functional_unet/tests/test_unet_model.py
}

run_python_model_tests_blackhole() {
    pytest models/demos/blackhole/stable_diffusion/tests

    # Llama3.1-8B
    llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/
    # Run all Llama3 tests for 8B - dummy weights with tight PCC check
    for llama_dir in "$llama8b"; do
        LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/tests/test_model.py -k "quick" ; fail+=$?
        echo "LOG_METAL: Llama3 tests for $llama_dir completed"
    done

    pytest tests/ttnn/integration_tests/resnet/test_ttnn_functional_resnet50.py
    pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4_bh.py
    pytest models/experimental/functional_unet/tests/test_unet_model.py
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
