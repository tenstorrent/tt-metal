env pytest models/experimental/whisper/tests/test_whisper_attention.py
env pytest models/experimental/whisper/tests/test_whisper_decoder_layer.py

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

env pytest models/experimental/bloom/tests/test_baddbmm.py
env pytest models/experimental/bloom/tests/test_bloom_attention.py
env pytest models/experimental/bloom/tests/test_bloom_gelu_forward.py
env pytest models/experimental/bloom/tests/test_bloom_merge_heads.py
env pytest models/experimental/bloom/tests/test_bloom_mlp.py

env pytest models/demos/metal_BERT_large_11/tests/test_bert_batch_dram.py -k batch_7-BFLOAT8_B-SHARDED
env pytest models/demos/metal_BERT_large_11/tests/test_demo.py::test_demo -k batch_7
env pytest models/demos/metal_BERT_large_11/tests/test_demo.py::test_demo_squadv2 -k batch_7

env pytest models/experimental/synthetic_gradients/tests/

env pytest models/experimental/lenet/tests/test_lenet.py

env pytest models/experimental/convnet_mnist/tests/test_convnet_mnist.py

env pytest models/experimental/yolov5/tests/test_yolov5_detection_model.py

env pytest models/experimental/yolov3/tests/test_yolov3_upsample.py
env pytest models/experimental/yolov3/tests/test_yolov3_concat.py
env pytest models/experimental/yolov3/tests/test_yolov3_conv2d.py
env pytest models/experimental/yolov3/tests/test_yolov3_bottleneck.py
env pytest models/experimental/yolov3/tests/test_yolov3_detection_model.py
env pytest models/experimental/yolov3/tests/test_yolov3_detect.py
env pytest models/experimental/yolov3/tests/test_yolov3_conv.py

env pytest models/experimental/efficientnet/tests/test_efficientnet_model.py

env pytest models/demos/falcon7b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT16-L1-falcon_7b-layers_32-prefill_seq128]

env pytest models/experimental/stable_diffusion/tests/test_embedding.py

env pytest models/demos/ttnn_falcon7b/tests/test_falcon_mlp.py
env pytest models/demos/ttnn_falcon7b/tests/test_falcon_rotary_embedding.py
env pytest models/demos/ttnn_falcon7b/tests/test_falcon_attention.py
env pytest models/demos/ttnn_falcon7b/tests/test_falcon_decoder.py
