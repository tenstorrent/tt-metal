# Bert large matmuls
# no bias
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_fused_qkv_matmul_test[BFLOAT8_B-in0_L1-in1_DRAM-bias_None-out_L1]
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_selfout_matmul_test[BFLOAT8_B-in0_L1-in1_DRAM-bias_None-out_L1]
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_ff1_matmul_test[BFLOAT8_B-in0_L1-in1_DRAM-bias_None-out_L1-gelu_activation]
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_ff2_matmul_test[BFLOAT8_B-in0_L1-in1_DRAM-bias_None-out_L1]
# bias dram
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_fused_qkv_matmul_test[BFLOAT8_B-in0_L1-in1_DRAM-bias_DRAM-out_L1]
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_selfout_matmul_test[BFLOAT8_B-in0_L1-in1_DRAM-bias_DRAM-out_L1]
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_ff1_matmul_test[BFLOAT8_B-in0_L1-in1_DRAM-bias_DRAM-out_L1-gelu_activation]
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_ff2_matmul_test[BFLOAT8_B-in0_L1-in1_DRAM-bias_DRAM-out_L1]
# bias L1
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_fused_qkv_matmul_test[BFLOAT8_B-in0_L1-in1_DRAM-bias_L1-out_L1]
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_selfout_matmul_test[BFLOAT8_B-in0_L1-in1_DRAM-bias_L1-out_L1]
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_ff1_matmul_test[BFLOAT8_B-in0_L1-in1_DRAM-bias_L1-out_L1-gelu_activation]
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_ff2_matmul_test[BFLOAT8_B-in0_L1-in1_DRAM-bias_L1-out_L1]

# Bert large bmms
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_pre_softmax_bmm_test[BFLOAT8_B-in0_L1-in1_L1-out_L1]
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_post_softmax_bmm_test[BFLOAT8_B-in0_L1-in1_L1-out_L1]

# TMs
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_split_query_key_value_and_split_heads
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_concatenate_heads_test[BFLOAT8_B-in0_L1-out_L1]

# Fused ops
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_layernorm_test[add_LN_GB-BFLOAT16-in0_L1-out_L1]
pytest models/experimental/bert_large_performant/unit_tests -k test_bert_large_softmax_test[scale_mask_softmax-BFLOAT16-in0_L1]
