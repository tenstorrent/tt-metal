#/bin/bash

set -eo pipefail

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

# This must run in slow dispatch mode
# pytest -svv $TT_METAL_HOME/tests/python_api_testing/sweep_tests/pytests/test_sweep_conv_with_address_map.py

# For now, adding tests with fast dispatch and non-32B divisible page sizes here. Python/models people,
# you can move to where you'd like.

if [ "$ARCH_NAME" != "wormhole_b0" ]; then
    # Tests for tensors in L1
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

    pytest tests/ttnn/integration_tests/resnet/test_ttnn_functional_resnet50_new.py -k "pretrained_weight_false"
    # Falcon tests
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_128 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_512 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_attn_matmul.py
  else
    # wormhole_b0

    # Falcon tests
    # attn_matmul_from_cache is currently not used in falcon7b
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_attn_matmul.py -k "not attn_matmul_from_cache"
    # higher sequence lengths and different formats trigger memory issues
    pytest models/demos/falcon7b_common/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py -k "seq_len_128 and in0_BFLOAT16-in1_BFLOAT8_B-out_BFLOAT16-weights_DRAM"
    pytest tests/ttnn/integration_tests/resnet/test_ttnn_functional_resnet50_new.py -k "pretrained_weight_false"

    # Unet Shallow
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -svv models/experimental/functional_unet/tests/test_unet_model.py

    # Llama 3.1 8B
    pytest models/demos/wormhole/llama31_8b/tests/test_llama_decoder.py
fi
