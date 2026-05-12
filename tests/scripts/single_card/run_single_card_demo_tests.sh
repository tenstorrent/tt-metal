#!/bin/bash
set -e

TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache

# ============================================================================
# Pytest Command Configuration
# ============================================================================
# Use PYTEST_CMD environment variable to customize how pytest is invoked.
# This allows wrapping pytest with Tracy profiler for ops recording.
#
# Default: "pytest" (standard pytest)
# For ops recording: "python -m tracy -r -p --dump-device-data-mid-run -m pytest"
#
# Example usage:
#   export PYTEST_CMD="python -m tracy -r -p --dump-device-data-mid-run -m pytest"
#   source run_single_card_demo_tests.sh
#   run_resnet_func
# ============================================================================
PYTEST_CMD="${PYTEST_CMD:-pytest}"

echo "[CONFIG] PYTEST_CMD=${PYTEST_CMD}"

run_ds_r1_qwen_func() {
  ds_r1_qwen_14b=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
  HF_MODEL=$ds_r1_qwen_14b MESH_DEVICE=N300 $PYTEST_CMD --timeout 1200 models/tt_transformers/demo/simple_text_demo.py -k performance-ci-1

  ds_r1_qwen_1_5b=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  HF_MODEL=$ds_r1_qwen_1_5b MESH_DEVICE=N300 $PYTEST_CMD --timeout 1200 models/experimental/tt_transformers_v2/ds_r1_qwen.py
}

run_phi4_func() {
  HF_MODEL=microsoft/phi-4 $PYTEST_CMD models/tt_transformers/demo/simple_text_demo.py -k "accuracy and ci-token-matching"
}

run_segformer_func() {
  #Segformer Segmentation Demo
  $PYTEST_CMD models/demos/vision/segmentation/segformer/demo/demo_for_semantic_segmentation.py

  #Segformer Classification Demo
  $PYTEST_CMD models/demos/vision/segmentation/segformer/demo/demo_for_image_classification.py

}

run_sentencebert_func() {

  #SentenceBERT Demo
  $PYTEST_CMD models/demos/wormhole/sentence_bert/demo/demo.py

}

run_bge_m3_demo() {

  # BGE-M3 Demo
  HF_MODEL=BAAI/bge-m3 $PYTEST_CMD models/demos/wormhole/bge_m3/demo/demo.py --timeout 1200
  echo "LOG_METAL: BGE-M3 Demo completed"

}

run_yolov11_func() {

 #Yolov11 Demo
 $PYTEST_CMD models/demos/yolov11/demo/demo.py

}

run_yolov11m_func() {

 #Yolov11m Demo
 $PYTEST_CMD --disable-warnings models/demos/yolov11m/demo/demo.py --timeout 600; fail+=$?

}

run_ufld_v2_func() {
  #ufld_v2 demo
  $PYTEST_CMD models/demos/vision/segmentation/ufld_v2/wormhole/demo/demo.py
}

run_vgg_func() {

  #VGG11/VGG16
  $PYTEST_CMD models/demos/vision/classification/vgg/demo/demo.py

}

run_bert_tiny_func() {

  pytest models/demos/wormhole/bert_tiny/demo/demo.py

}

run_bert_func() {
  fail=0

  $PYTEST_CMD models/demos/metal_BERT_large_11/demo/demo.py -k batch_7 || fail=1
  $PYTEST_CMD models/demos/metal_BERT_large_11/demo/demo.py -k batch_8 || fail=1

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi

}

run_resnet_stability() {

  $PYTEST_CMD models/demos/vision/classification/resnet50/wormhole/tests/test_resnet50_stability.py -k "short"

}

run_resnet_func() {

  $PYTEST_CMD models/demos/vision/classification/resnet50/wormhole/demo/demo.py

}

run_sdxl_func() {
  TT_MM_THROTTLE_PERF=5 $PYTEST_CMD models/demos/stable_diffusion_xl_base/demo/demo.py -k "device_vae and device_encoders and with_trace and no_cfg_parallel"
  TT_MM_THROTTLE_PERF=5 $PYTEST_CMD models/demos/stable_diffusion_xl_base/demo/demo_base_and_refiner.py -k "device_vae and device_encoders and with_trace and no_cfg_parallel"
  TT_MM_THROTTLE_PERF=5 $PYTEST_CMD models/demos/stable_diffusion_xl_base/demo/demo_img2img.py -k "device_vae and device_encoders and with_trace and no_cfg_parallel"
  TT_MM_THROTTLE_PERF=5 $PYTEST_CMD models/demos/stable_diffusion_xl_base/demo/demo_inpainting.py -k "device_vae and device_encoders and with_trace and no_cfg_parallel"
  TT_MM_THROTTLE_PERF=5 $PYTEST_CMD models/demos/stable_diffusion_xl_base/demo/demo_lora.py -k "device_vae and device_encoders and with_trace and no_cfg_parallel"
}

run_distilbert_func() {

  pytest models/demos/wormhole/distilbert/demo/demo.py

}


run_mnist_func() {

  $PYTEST_CMD models/demos/vision/classification/mnist/demo/demo.py

}

run_squeezebert_func() {

  $PYTEST_CMD models/demos/squeezebert/demo/demo.py

}

run_efficientnet_b0_func(){

  $PYTEST_CMD models/experimental/efficientnetb0/demo/demo.py

}

run_stable_diffusion_func() {

  $PYTEST_CMD --timeout 600 --input-path="models/demos/vision/generative/stable_diffusion/wormhole/demo/input_data.json" models/demos/vision/generative/stable_diffusion/wormhole/demo/demo.py::test_demo

}


run_yolov9c_perf() {
  # yolov9c demo
  $PYTEST_CMD models/demos/yolov9c/demo/demo.py

}
run_yolov8s_perf() {

  # yolov8s demo
  $PYTEST_CMD models/demos/yolov8s/demo/demo.py

}


run_mobilenetv2_perf(){

#  mobilenetv2 demo
 $PYTEST_CMD models/demos/vision/classification/mobilenetv2/demo/demo.py

}

run_yolov8s_world_perf() {

  # yolov8s_world demo
  $PYTEST_CMD models/demos/yolov8s_world/demo/demo.py


}

run_vanilla_unet_demo() {
 # vanilla_unet demo
 $PYTEST_CMD models/demos/vision/segmentation/vanilla_unet/demo/demo.py
}

run_swin_s_demo() {

  $PYTEST_CMD models/experimental/swin_s/demo/demo.py

}

run_swin_v2_demo() {

  $PYTEST_CMD models/experimental/swin_v2/demo/demo.py

}

run_yolov8x_perf() {

  # yolov8x demo
  $PYTEST_CMD models/demos/yolov8x/demo/demo.py


}
run_yolov4_perf() {
  #yolov4 demo
  $PYTEST_CMD models/demos/yolov4/demo.py

}

run_yolov10x_demo() {
  # yolov10x demo
  $PYTEST_CMD models/demos/yolov10x/demo/demo.py


}

run_yolov7_demo() {
  # yolov7 demo
  $PYTEST_CMD models/demos/yolov7/demo/demo.py


}

run_yolov6l_demo() {
  # yolov6 demo
  $PYTEST_CMD models/demos/yolov6l/demo/demo.py

}

run_vgg_unet_demo() {
 # vgg_unet demo
  $PYTEST_CMD models/demos/vision/segmentation/vgg_unet/wormhole/demo/demo.py
}


run_yolov12x_demo() {

  $PYTEST_CMD models/demos/yolov12x/demo/demo.py

}


run_vovnet_demo(){

 $PYTEST_CMD models/experimental/vovnet/demo/demo.py

}

run_vit_demo(){

 $PYTEST_CMD models/demos/vision/classification/vit/wormhole/tests/test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py

}

main() {
  # For CI pipeline - source func commands but don't execute tests if not invoked directly
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Script is being sourced, not executing main function"
    return 0
  fi

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME

  # Insert tests for running locally (currently none since script is only sourced for CI)
}

main "$@"
