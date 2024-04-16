env pytest tests/ttnn/integration_tests

env pytest models/experimental/whisper -k WhipserEncoderLayer_inference
env pytest models/experimental/whisper -k WhipserEncoder_inference
env pytest models/experimental/whisper -k WhipserDecoder_inference
env pytest models/experimental/whisper -k whisper_model
env pytest models/experimental/whisper -k whisper_for_audio_classification
env pytest models/experimental/whisper -k whisper_for_conditional_generation

env pytest models/experimental/t5/tests -k t5_attention
env pytest models/experimental/t5/tests -k t5_layer_self_attention
env pytest models/experimental/t5/tests -k t5_layer_cross_attention
env pytest models/experimental/t5/tests -k t5_block
env pytest models/experimental/t5/tests -k t5_stack
env pytest models/experimental/t5/tests -k t5_model

env pytest models/experimental/roberta/tests -k roberta_layer
env pytest models/experimental/roberta/tests -k roberta_encoder
env pytest models/experimental/roberta/tests -k roberta_model
env pytest models/experimental/roberta/tests -k roberta_for_masked_lm
env pytest models/experimental/roberta/tests -k roberta_for_qa
env pytest models/experimental/roberta/tests -k roberta_for_sequence_classification
# THIS DIES HERE
env pytest models/experimental/roberta/tests -k roberta_for_token_classification
env pytest models/experimental/roberta/tests -k roberta_for_multiple_choice

env pytest models/experimental/bloom/tests -k bloom_block
env pytest models/experimental/bloom/tests -k bloom_model
env pytest models/experimental/bloom/tests -k bloom_causal_lm

env pytest models/experimental/stable_diffusion/tests/test_cross_attn_down_block.py -k test_run_cross_attn_down_block_real_input_inference
env pytest models/experimental/stable_diffusion/tests/test_cross_attn_up_block.py -k test_run_cross_attn_up_block_real_input_inference
env pytest models/experimental/stable_diffusion/tests/test_downblock_2d.py -k test_run_downblock_real_input_inference
env pytest models/experimental/stable_diffusion/tests/test_unet_mid_block.py -k test_run_unet_mid_block_real_input_inference
env pytest models/experimental/stable_diffusion/tests/test_upblock_2d.py -k test_run_upblock_real_input_inference
env pytest models/experimental/stable_diffusion/tests -k test_unbatched_stable_diffusion #

env pytest models/demos/metal_BERT_large_11/tests/test_demo.py::test_demo -k batch_12
env pytest models/demos/metal_BERT_large_11/tests/test_demo.py::test_demo_squadv2 -k batch_12

env pytest models/experimental/mnist/tests -k mnist_inference

env pytest models/experimental/nanogpt/tests -k nanogpt_attention
env pytest models/experimental/nanogpt/tests -k nanogpt_block
env pytest models/experimental/nanogpt/tests -k nanogpt_mlp
env pytest models/experimental/nanogpt/tests -k nanogpt_model

env pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference[HiFi4-activations_BFLOAT16-weights_BFLOAT16-batch_1]
env pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference[HiFi4-activations_BFLOAT16-weights_BFLOAT16-batch_2]
