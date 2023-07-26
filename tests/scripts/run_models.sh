#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

export PYTHONPATH=$TT_METAL_HOME

# Turning on fast dispatch for the remaining models
export TT_METAL_DEVICE_DISPATCH_MODE=1

env pytest tests/python_api_testing/models/stable_diffusion/test_embedding.py
env pytest tests/python_api_testing/models/stable_diffusion/test_cross_attn_down_block.py -k test_run_cross_attn_down_block_real_input_inference
env pytest tests/python_api_testing/models/stable_diffusion/test_cross_attn_up_block.py -k test_run_cross_attn_up_block_real_input_inference
env pytest tests/python_api_testing/models/stable_diffusion/test_downblock_2d.py -k test_run_downblock_real_input_inference
env pytest tests/python_api_testing/models/stable_diffusion/test_unet_mid_block.py -k test_run_unet_mid_block_real_input_inference
env pytest tests/python_api_testing/models/stable_diffusion/test_upblock_2d.py -k test_run_upblock_real_input_inference
env pytest tests/python_api_testing/models/stable_diffusion -k test_batched_stable_diffusion
env pytest tests/python_api_testing/models/stable_diffusion -k test_unbatched_stable_diffusion

env pytest tests/python_api_testing/models/deit/tests/test_deit_for_image_classification_with_teacher.py -k test_deit_for_image_classification_with_teacher_inference

env pytest tests/python_api_testing/models/vit/tests/test_vit_image_classification.py -k test_vit_image_classification

env pytest tests/python_api_testing/models/bert/bert_encoder.py -k bert_encoder
env pytest tests/python_api_testing/models/bert -k bert_question_and_answering

env pytest tests/python_api_testing/models/bert_large_performant/unit_tests -k "bert_large and not mixed_precision and not batch_8 and not batch_7"
env pytest tests/python_api_testing/models/bert_large_performant/unit_tests -k "mixed_precision and batch_9"
env pytest tests/python_api_testing/models/metal_BERT_large_15/mha.py
env pytest tests/python_api_testing/models/metal_BERT_large_15/ffn.py
env pytest tests/python_api_testing/models/metal_BERT_large_15/bert_encoder.py
env pytest tests/python_api_testing/models/metal_BERT_large_15/test_bert_batch_dram.py::test_bert_batch_dram
env pytest tests/python_api_testing/models/metal_BERT_large_15/test_bert_batch_dram.py::test_bert_batch_dram_with_program_cache

env pytest tests/python_api_testing/models/t5 -k t5_dense_act_dense
env pytest tests/python_api_testing/models/t5 -k t5_layer_norm
env pytest tests/python_api_testing/models/t5 -k t5_attention
env pytest tests/python_api_testing/models/t5 -k t5_layer_ff
env pytest tests/python_api_testing/models/t5 -k t5_layer_self_attention
env pytest tests/python_api_testing/models/t5 -k t5_layer_cross_attention
env pytest tests/python_api_testing/models/t5 -k t5_block
env pytest tests/python_api_testing/models/t5 -k t5_stack
env pytest tests/python_api_testing/models/t5 -k t5_model

env pytest tests/python_api_testing/models/synthetic_gradients -k batchnorm1d_test
env pytest tests/python_api_testing/models/synthetic_gradients -k linear_test
env pytest tests/python_api_testing/models/synthetic_gradients -k block_test
env pytest tests/python_api_testing/models/synthetic_gradients -k full_inference

env pytest tests/python_api_testing/models/llama -k llama_layer_norm
env pytest tests/python_api_testing/models/llama -k llama_mlp
env pytest tests/python_api_testing/models/llama -k llama_attention
env pytest tests/python_api_testing/models/llama -k llama_decoder

env pytest tests/python_api_testing/models/whisper -k whisper_attention
env pytest tests/python_api_testing/models/whisper -k WhipserEncoderLayer_inference
env pytest tests/python_api_testing/models/whisper -k WhipserEncoder_inference
env pytest tests/python_api_testing/models/whisper -k WhipserDecoderLayer_inference
env pytest tests/python_api_testing/models/whisper -k WhipserDecoder_inference
env pytest tests/python_api_testing/models/whisper -k whisper_model
env pytest tests/python_api_testing/models/whisper -k whisper_for_audio_classification
env pytest tests/python_api_testing/models/whisper -k whisper_for_conditional_generation

env pytest tests/python_api_testing/models/lenet -k test_lenet_inference
env pytest tests/python_api_testing/models/ConvNet_MNIST -k  mnist

env pytest tests/python_api_testing/models/roberta -k roberta_self_attention
env pytest tests/python_api_testing/models/roberta -k roberta_self_output
env pytest tests/python_api_testing/models/roberta -k roberta_attention
env pytest tests/python_api_testing/models/roberta -k roberta_intermediate
env pytest tests/python_api_testing/models/roberta -k roberta_output
env pytest tests/python_api_testing/models/roberta -k roberta_layer
env pytest tests/python_api_testing/models/roberta -k roberta_encoder
env pytest tests/python_api_testing/models/roberta -k roberta_pooler
env pytest tests/python_api_testing/models/roberta -k roberta_model
env pytest tests/python_api_testing/models/roberta -k roberta_lm_head
env pytest tests/python_api_testing/models/roberta -k roberta_for_masked_lm
env pytest tests/python_api_testing/models/roberta -k roberta_for_qa
env pytest tests/python_api_testing/models/roberta -k roberta_for_sequence_classification
env pytest tests/python_api_testing/models/roberta -k roberta_classification_head
env pytest tests/python_api_testing/models/roberta -k roberta_for_token_classification
env pytest tests/python_api_testing/models/roberta -k roberta_for_multiple_choice

env pytest tests/python_api_testing/models/yolov5/tests -k Yolov5_detection_model

env pytest tests/python_api_testing/models/bloom -k baddbmm
env pytest tests/python_api_testing/models/bloom -k bloom_attention
env pytest tests/python_api_testing/models/bloom -k bloom_block
env pytest tests/python_api_testing/models/bloom -k bloom_gelu_forward
env pytest tests/python_api_testing/models/bloom -k bloom_merge_heads
env pytest tests/python_api_testing/models/bloom -k bloom_mlp
env pytest tests/python_api_testing/models/bloom -k bloom_model
env pytest tests/python_api_testing/models/bloom -k bloom_causal_lm

env pytest tests/python_api_testing/models/yolov3 -k conv2d_module
env pytest tests/python_api_testing/models/yolov3 -k conv_module
env pytest tests/python_api_testing/models/yolov3 -k concat_module
env pytest tests/python_api_testing/models/yolov3 -k bottleneck_module
env pytest tests/python_api_testing/models/yolov3 -k detect_module
env pytest tests/python_api_testing/models/yolov3 -k detection_model
env pytest tests/python_api_testing/models/yolov3 -k upsample_module

env pytest tests/python_api_testing/models/EfficientNet/tests -k efficientnet_b0_model_real
env pytest tests/python_api_testing/models/EfficientNet/tests -k efficientnet_v2_s_model_real
env pytest tests/python_api_testing/models/EfficientNet/tests -k efficientnet_lite0_model_real

env pytest tests/python_api_testing/models/nanogpt -k nanogpt_model_real
