#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

export PYTHONPATH=$TT_METAL_HOME

env pytest tests/python_api_testing/models/stable_diffusion/test_feedforward.py
env pytest tests/python_api_testing/models/stable_diffusion/test_embedding.py
env pytest tests/python_api_testing/models/stable_diffusion/test_cross_attention.py
env pytest tests/python_api_testing/models/stable_diffusion/test_cross_attn_down_block.py
env pytest tests/python_api_testing/models/stable_diffusion/test_downblock_2d.py
env pytest tests/python_api_testing/models/stable_diffusion/test_downsample_2d.py
env pytest tests/python_api_testing/models/stable_diffusion/test_residual_block.py
env pytest tests/python_api_testing/models/stable_diffusion/test_transformers.py
env pytest tests/python_api_testing/models/stable_diffusion/test_unet_mid_block.py
env pytest tests/python_api_testing/models/stable_diffusion/test_upblock_2d.py
env pytest tests/python_api_testing/models/stable_diffusion/test_upsample_2d.py
env pytest tests/python_api_testing/models/stable_diffusion/test_upsample_nearest2d.py

env pytest tests/python_api_testing/models/bert/bert_encoder.py -k bert_encoder
env pytest tests/python_api_testing/models/bert -k bert_question_and_answering
env pytest tests/python_api_testing/models/bert_large_performant/unit_tests -k bert_large

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

env pytest tests/python_api_testing/models/LeNet -k LeNet
env pytest tests/python_api_testing/models/ConvNet_MNIST -k  mnist

env pytest tests/python_api_testing/models/bloom -k bloom_mlp
env pytest tests/python_api_testing/models/bloom -k bloom_attention
env pytest tests/python_api_testing/models/bloom -k bloom_block
env pytest tests/python_api_testing/models/bloom -k bloom_gelu_forward
