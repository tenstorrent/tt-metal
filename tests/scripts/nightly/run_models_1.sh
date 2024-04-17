env pytest tests/ttnn/integration_tests

env pytest models/experimental/whisper/tests/test_whisper_encoder_layer.py
env pytest models/experimental/whisper/tests/test_whisper_encoder.py
env pytest models/experimental/whisper/tests/test_whisper_decoder.py
env pytest models/experimental/whisper/tests/test_whisper_model.py
env pytest models/experimental/whisper/tests/test_whisper_for_audio_classification.py
env pytest models/experimental/whisper/tests/test_whisper_for_conditional_generation.py

env pytest models/experimental/t5/tests/test_t5_attention.py
env pytest models/experimental/t5/tests/test_t5_layer_self_attention.py
env pytest models/experimental/t5/tests/test_t5_layer_cross_attention.py
env pytest models/experimental/t5/tests/test_t5_block.py
env pytest models/experimental/t5/tests/test_t5_stack.py
env pytest models/experimental/t5/tests/test_t5_model.py

env pytest models/experimental/roberta/tests/test_roberta_layer.py
env pytest models/experimental/roberta/tests/test_roberta_encoder.py
env pytest models/experimental/roberta/tests/test_roberta_model.py
env pytest models/experimental/roberta/tests/test_roberta_for_masked_lm.py
env pytest models/experimental/roberta/tests/test_roberta_for_qa.py
env pytest models/experimental/roberta/tests/test_roberta_for_sequence_classification.py
env pytest models/experimental/roberta/tests/test_roberta_for_token_classification.py # -> issue #7533
env pytest models/experimental/roberta/tests/test_roberta_for_multiple_choice.py

env pytest models/experimental/bloom/tests/test_bloom_block.py
env pytest models/experimental/bloom/tests/test_bloom_model.py
env pytest models/experimental/bloom/tests/test_bloom_causal_lm.py

env pytest models/experimental/stable_diffusion/tests/test_cross_attn_down_block.py # -> test failing , #7536
env pytest models/experimental/stable_diffusion/tests/test_cross_attn_up_block.py    # -> test failing, #7536
env pytest models/experimental/stable_diffusion/tests/test_downblock_2d.py
env pytest models/experimental/stable_diffusion/tests/test_unet_mid_block.py  # test failing, #7536
env pytest models/experimental/stable_diffusion/tests/test_upblock_2d.py
env pytest models/experimental/stable_diffusion/tests/test_unbatched_stable_diffusion.py      # test failing, #7536

env pytest models/demos/metal_BERT_large_11/tests/test_demo.py

env pytest models/experimental/mnist/tests/test_mnist.py

# can symlink the entire folder
env pytest models/experimental/nanogpt/tests/test_nanogpt_attention.py     # -> hangs gs, #7534
env pytest models/experimental/nanogpt/tests/test_nanogpt_block.py         # -> hangs gs. #7534
env pytest models/experimental/nanogpt/tests/test_nanogpt_mlp.py
env pytest models/experimental/nanogpt/tests/test_nanogpt_model.py        # failing, #7534

# why is this not in test_perf_device_resnet.py, also these parameters are specifically skipped inside the test
# env pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference[HiFi4-activations_BFLOAT16-weights_BFLOAT16-batch_1]
# env pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference[HiFi4-activations_BFLOAT16-weights_BFLOAT16-batch_2]
