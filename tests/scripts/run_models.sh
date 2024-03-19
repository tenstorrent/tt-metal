#/bin/bash
set -eo pipefail
if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi
cd $TT_METAL_HOME
export PYTHONPATH=$TT_METAL_HOME

if [[ $ARCH_NAME == "grayskull" ]]; then
  # TODO(arakhmati): Run ttnn tests only on graskull until the issue with ttnn.reshape on wormhole is resolved
  env pytest tests/ttnn/integration_tests
fi

if [[ $ARCH_NAME == "wormhole" ]]; then
  env pytest tests/ttnn/integration_tests/stable_diffusion
  env pytest tests/ttnn/integration_tests/unet
fi

env pytest models/experimental/whisper -k whisper_attention
env pytest models/experimental/whisper -k WhipserDecoderLayer_inference

env pytest models/experimental/deit/tests/test_deit_for_image_classification_with_teacher.py -k test_deit_for_image_classification_with_teacher_inference

env pytest models/experimental/vit/tests/test_vit_image_classification.py -k test_vit_image_classification

env pytest models/experimental/t5/tests -k t5_dense_act_dense
env pytest models/experimental/t5/tests -k t5_layer_norm
env pytest models/experimental/t5/tests -k t5_layer_ff

env pytest models/experimental/roberta/tests -k roberta_self_attention
env pytest models/experimental/roberta/tests -k roberta_self_output
env pytest models/experimental/roberta/tests -k roberta_attention
env pytest models/experimental/roberta/tests -k roberta_intermediate
env pytest models/experimental/roberta/tests -k roberta_output
env pytest models/experimental/roberta/tests -k roberta_pooler
env pytest models/experimental/roberta/tests -k roberta_lm_head
env pytest models/experimental/roberta/tests -k roberta_classification_head

env pytest models/experimental/bloom/tests -k baddbmm
env pytest models/experimental/bloom/tests -k bloom_attention
env pytest models/experimental/bloom/tests -k bloom_gelu_forward
env pytest models/experimental/bloom/tests -k bloom_merge_heads
env pytest models/experimental/bloom/tests -k bloom_mlp

# Currently hangs due to #4968
# env pytest models/demos/metal_BERT_large_11/tests/test_bert_batch_dram.py -k batch_7-BFLOAT8_B-SHARDED
# env pytest models/demos/metal_BERT_large_11/tests/test_demo.py::test_demo -k batch_7
# env pytest models/demos/metal_BERT_large_11/tests/test_demo.py::test_demo_squadv2 -k batch_7

env pytest models/experimental/synthetic_gradients/tests -k test_batchnorm1d
env pytest models/experimental/synthetic_gradients/tests -k test_linear
env pytest models/experimental/synthetic_gradients/tests -k test_block
env pytest models/experimental/synthetic_gradients/tests -k test_full_inference

env pytest models/experimental/lenet/tests -k test_lenet_inference

env pytest models/experimental/convnet_mnist/tests -k mnist_inference

env pytest models/experimental/yolov5/tests -k Yolov5_detection_model
env pytest models/experimental/yolov3 -k conv2d_module
env pytest models/experimental/yolov3 -k conv_module
env pytest models/experimental/yolov3 -k concat_module
env pytest models/experimental/yolov3 -k bottleneck_module
env pytest models/experimental/yolov3 -k detect_module
env pytest models/experimental/yolov3 -k detection_model
env pytest models/experimental/yolov3 -k upsample_module

env pytest models/experimental/efficientnet/tests -k efficientnet_b0_model_real
env pytest models/experimental/efficientnet/tests -k efficientnet_v2_s_model_real
env pytest models/experimental/efficientnet/tests -k efficientnet_lite0_model_real

env pytest models/demos/falcon7b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT16-L1-falcon_7b-layers_32-prefill_seq128]

env pytest models/experimental/stable_diffusion/tests/test_embedding.py

env pytest models/experimental/mistral/tests/test_mistral_feed_forward.py
env pytest models/experimental/mistral/tests/test_mistral_rms_norm.py
env pytest models/experimental/mistral/tests/test_mistral_transformer_block.py

# GRAYSKULL ONLY

if [[ $ARCH_NAME == "grayskull" ]]; then

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

env pytest models/demos/metal_BERT_large_11/tests/test_bert_batch_dram.py
env pytest models/demos/metal_BERT_large_11/tests/test_demo.py::test_demo -k batch_12
env pytest models/demos/metal_BERT_large_11/tests/test_demo.py::test_demo_squadv2 -k batch_12

env pytest models/demos/falcon7b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT16-L1-falcon_7b-layers_32-decode_batch32]

env pytest models/experimental/mnist/tests -k mnist_inference

env pytest models/experimental/nanogpt/tests -k nanogpt_attention
env pytest models/experimental/nanogpt/tests -k nanogpt_block
env pytest models/experimental/nanogpt/tests -k nanogpt_mlp
env pytest models/experimental/nanogpt/tests -k nanogpt_model

env pytest models/experimental/mistral/tests/test_mistral_attention.py

env pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference[HiFi4-activations_BFLOAT16-weights_BFLOAT16-batch_1]
env pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference[HiFi4-activations_BFLOAT16-weights_BFLOAT16-batch_2]

#3524 SD gets lower PCC than FD for Resnet
if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
  env pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference[HiFi4-activations_BFLOAT16-weights_BFLOAT16-batch_8]
  env pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference -k "activations_BFLOAT8_B-weights_BFLOAT8_B and (batch_8 or batch_16)"
  env pytest models/demos/resnet/tests/test_demo.py::test_demo_sample[8-models/demos/resnet/demo/images/]
  env pytest models/demos/resnet/tests/test_demo.py::test_demo_imagenet[8-400]
fi
fi
