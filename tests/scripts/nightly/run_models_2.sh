env pytest models/experimental/whisper -k whisper_attention
env pytest models/experimental/whisper -k WhipserDecoderLayer_inference

env pytest models/experimental/deit/tests/test_deit_for_image_classification_with_teacher.py

env pytest models/experimental/vit/tests/test_vit_image_classification.py

env pytest models/experimental/t5/tests/test_t5_dense_act_dense.py
env pytest models/experimental/t5/tests/test_t5_layer_norm.py
env pytest models/experimental/t5/tests/test_t5_layer_ff.py

env pytest models/experimental/roberta/tests/test_roberta_self_attention.py
env pytest models/experimental/roberta/tests/test_roberta_self_output.py
env pytest models/experimental/roberta/tests/test_roberta_attention.py
env pytest models/experimental/roberta/tests/test_roberta_intermediate.py
env pytest models/experimental/roberta/tests/test_roberta_output.py
env pytest models/experimental/roberta/tests/test_roberta_pooler.py
env pytest models/experimental/roberta/tests/test_roberta_lm_head.py
env pytest models/experimental/roberta/tests/test_roberta_classification_head.py

env pytest models/experimental/bloom/tests -k baddbmm
env pytest models/experimental/bloom/tests -k bloom_attention
env pytest models/experimental/bloom/tests -k bloom_gelu_forward
env pytest models/experimental/bloom/tests -k bloom_merge_heads
env pytest models/experimental/bloom/tests -k bloom_mlp

env pytest models/demos/metal_BERT_large_11/tests/test_bert_batch_dram.py -k batch_7-BFLOAT8_B-SHARDED
env pytest models/demos/metal_BERT_large_11/tests/test_demo.py::test_demo -k batch_7
env pytest models/demos/metal_BERT_large_11/tests/test_demo.py::test_demo_squadv2 -k batch_7

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

env pytest models/demos/ttnn_falcon7b/tests -k falcon_mlp
env pytest models/demos/ttnn_falcon7b/tests -k falcon_rotary_embeddings
env pytest models/demos/ttnn_falcon7b/tests -k falcon_attention
env pytest models/demos/ttnn_falcon7b/tests -k falcon_decoder
