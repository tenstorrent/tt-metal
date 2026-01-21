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

run_falcon7b_func() {

  $PYTEST_CMD -n auto --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo -k "default_mode_1024_stochastic"

}

run_mistral7b_func() {

  mistral7b=mistralai/Mistral-7B-Instruct-v0.3
  mistral_cache=$TT_CACHE_HOME/$mistral7b
  HF_MODEL=$mistral7b TT_CACHE_PATH=$mistral_cache $PYTEST_CMD -n auto models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-token-matching" --timeout 1200; fail+=$?

}

run_qwen7b_func() {

  qwen7b=Qwen/Qwen2-7B-Instruct
  qwen_cache=$TT_CACHE_HOME/$qwen7b
  HF_MODEL=$qwen7b TT_CACHE_PATH=$qwen_cache MESH_DEVICE=N300 $PYTEST_CMD -n auto models/tt_transformers/demo/simple_text_demo.py -k performance-ci-1 --timeout 1800

}

run_qwen25_vl_perfunc() {
  fail=0

  # install qwen25_vl requirements
  uv pip install -r models/demos/qwen25_vl/requirements.txt

  # export PYTEST_ADDOPTS for concise $PYTEST_CMD output
  export PYTEST_ADDOPTS="--tb=short"

  # Qwen2.5-VL-3B-Instruct
  qwen25_vl_3b=Qwen/Qwen2.5-VL-3B-Instruct
  # Qwen2.5-VL-7B-Instruct
  qwen25_vl_7b=Qwen/Qwen2.5-VL-7B-Instruct

  # simple generation-accuracy tests for qwen25_vl_3b
  HF_MODEL=$qwen25_vl_3b TT_CACHE_PATH=$TT_CACHE_HOME/$qwen25_vl_3b $PYTEST_CMD -n auto models/demos/qwen25_vl/demo/combined.py -k tt_vision --timeout 1200 || fail=1
  echo "LOG_METAL: demo/combined.py tests for $qwen25_vl_3b on N300 completed"

  # complete demo tests
  for qwen_model in "$qwen25_vl_3b" "$qwen25_vl_7b"; do
    cache_path=$TT_CACHE_HOME/$qwen_model
    HF_MODEL=$qwen_model TT_CACHE_PATH=$cache_path $PYTEST_CMD -n auto models/demos/qwen25_vl/demo/demo.py --timeout 900 || fail=1
    echo "LOG_METAL: Tests for $qwen_model on N300 completed"
  done

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_ds_r1_qwen_func() {
  ds_r1_qwen_14b=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
  HF_MODEL=$ds_r1_qwen_14b MESH_DEVICE=N300 $PYTEST_CMD models/tt_transformers/demo/simple_text_demo.py -k performance-ci-1

  ds_r1_qwen_1_5b=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  HF_MODEL=$ds_r1_qwen_1_5b MESH_DEVICE=N300 $PYTEST_CMD models/experimental/tt_transformers_v2/ds_r1_qwen.py
}

run_gemma3_func() {
  HF_MODEL=/mnt/MLPerf/tt_dnn-models/google/gemma-3-4b-it $PYTEST_CMD models/demos/gemma3/demo/text_demo.py -k "ci-token-matching"
  echo "LOG_METAL: Gemma3 4B accuracy tests completed (text only)"
}

run_gemma3_perf() {
  HF_MODEL=/mnt/MLPerf/tt_dnn-models/google/gemma-3-4b-it $PYTEST_CMD models/demos/gemma3/demo/text_demo.py -k "performance and ci-1"
  echo "LOG_METAL: Gemma3 4B perf tests completed (text only)"
  HF_MODEL=/mnt/MLPerf/tt_dnn-models/google/gemma-3-4b-it $PYTEST_CMD models/demos/gemma3/demo/vision_demo.py -k "performance and batch1-multi-image-trace"
  echo "LOG_METAL: Gemma3 4B perf tests completed (text and vision)"
}

run_phi4_func() {
  HF_MODEL=microsoft/phi-4 $PYTEST_CMD models/tt_transformers/demo/simple_text_demo.py -k "accuracy and ci-token-matching"
}

run_segformer_func() {
  #Segformer Segmentation Demo
  $PYTEST_CMD models/demos/segformer/demo/demo_for_semantic_segmentation.py

  #Segformer Classification Demo
  $PYTEST_CMD models/demos/segformer/demo/demo_for_image_classification.py

}

run_sentencebert_func() {

  #SentenceBERT Demo
  $PYTEST_CMD models/demos/wormhole/sentence_bert/demo/demo.py

}

run_yolov11_func() {

 #Yolov11 Demo
 $PYTEST_CMD models/demos/yolov11/demo/demo.py

}

run_yolov11m_func() {

 #Yolov11m Demo
 $PYTEST_CMD --disable-warnings models/demos/yolov11m/demo/demo.py --timeout 600; fail+=$?

}

run_llama3_func() {
  fail=0

  # Llama3 Accuracy tests
  # Llama3.2-1B
  llama1b=meta-llama/Llama-3.2-1B-Instruct
  # Llama3.2-3B
  llama3b=meta-llama/Llama-3.2-3B-Instruct
  # Llama3.1-8B (11B weights are the same)
  llama8b=meta-llama/Llama-3.1-8B-Instruct
  # Llama3.2-11B
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct

  # Run Llama3 accuracy tests for 1B, 3B, 8B, 11b weights
  for hf_model in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    cache_path=$TT_CACHE_HOME/$hf_model
    HF_MODEL=$hf_model TT_CACHE_PATH=$cache_path $PYTEST_CMD -n auto models/tt_transformers/demo/simple_text_demo.py -k ci-token-matching  --timeout 420 || fail=1
    echo "LOG_METAL: Llama3 accuracy tests for $hf_model completed"
  done

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi

}

run_ufld_v2_func() {
  #ufld_v2 demo
  $PYTEST_CMD models/demos/wormhole/ufld_v2/demo/demo.py
}

run_vgg_func() {

  #VGG11/VGG16
  $PYTEST_CMD models/demos/vgg/demo/demo.py

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

  $PYTEST_CMD models/demos/wormhole/resnet50/tests/test_resnet50_stability.py -k "short"

}

run_resnet_func() {

  $PYTEST_CMD models/demos/wormhole/resnet50/demo/demo.py

}

run_sdxl_func() {
  TT_MM_THROTTLE_PERF=5 $PYTEST_CMD models/experimental/stable_diffusion_xl_base/tests/test_sdxl_accuracy.py --start-from=0 --num-prompts=2 -k "device_encoders and device_vae and no_cfg_parallel"
  TT_MM_THROTTLE_PERF=5 $PYTEST_CMD  models/experimental/stable_diffusion_xl_base/demo/demo_img2img.py -k "device_vae and device_encoders and with_trace and no_cfg_parallel"
  TT_MM_THROTTLE_PERF=5 $PYTEST_CMD  models/experimental/stable_diffusion_xl_base/demo/demo_inpainting.py -k "device_vae and device_encoders and with_trace and no_cfg_parallel"
}

run_distilbert_func() {

  pytest models/demos/wormhole/distilbert/demo/demo.py

}


run_mnist_func() {

  $PYTEST_CMD models/demos/mnist/demo/demo.py

}

run_squeezebert_func() {

  $PYTEST_CMD models/demos/squeezebert/demo/demo.py

}

run_efficientnet_b0_func(){

  $PYTEST_CMD models/experimental/efficientnetb0/demo/demo.py

}

run_stable_diffusion_func() {

  $PYTEST_CMD --input-path="models/demos/wormhole/stable_diffusion/demo/input_data.json" models/demos/wormhole/stable_diffusion/demo/demo.py::test_demo

}

run_mistral7b_perf() {

  # To ensure a proper perf measurement and dashboard upload of Mistral-7B N150, we have to run them on the N300 perf pipeline for now
  mistral7b=mistralai/Mistral-7B-Instruct-v0.3
  mistral_cache=$TT_CACHE_HOME/$mistral7b
  # Run Mistral-7B-v0.3 for N150
  MESH_DEVICE=N150 HF_MODEL=$mistral7b TT_CACHE_PATH=$mistral_cache $PYTEST_CMD -n auto models/tt_transformers/demo/simple_text_demo.py --timeout 600 -k "not performance-ci-stress-1"; fail+=$?
  # Run Mistral-7B-v0.3 for N300
  MESH_DEVICE=N300 HF_MODEL=$mistral7b TT_CACHE_PATH=$mistral_cache $PYTEST_CMD -n auto models/tt_transformers/demo/simple_text_demo.py --timeout 600 -k "not performance-ci-stress-1"; fail+=$?

}

run_llama3_perf() {
  fail=0

  # Llama3.2-1B
  llama1b=meta-llama/Llama-3.2-1B-Instruct
  # Llama3.2-3B
  llama3b=meta-llama/Llama-3.2-3B-Instruct
  # Llama3.1-8B
  llama8b=meta-llama/Llama-3.1-8B-Instruct
  # Llama3.2-11B (same tet weights as 8B)
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct

  # Run all Llama3 tests for 1B, 3B, 8B weights for N150
  # To ensure a proper perf measurement and dashboard upload of the Llama3 models on a N150, we have to run them on the N300 perf pipeline for now
  for hf_model in "$llama1b" "$llama3b" "$llama8b"; do
    cache_path=$TT_CACHE_HOME/$hf_model
    MESH_DEVICE=N150 HF_MODEL=$hf_model TT_CACHE_PATH=$cache_path $PYTEST_CMD -n auto models/tt_transformers/demo/simple_text_demo.py --timeout 600 -k "not performance-ci-stress-1" || fail=1
    echo "LOG_METAL: Llama3 tests for $hf_model completed on N150"
  done
  # Run all Llama3 tests for 1B, 3B, 8B and 11B weights
  for hf_model in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    cache_path=$TT_CACHE_HOME/$hf_model
    HF_MODEL=$hf_model TT_CACHE_PATH=$cache_path $PYTEST_CMD -n auto models/tt_transformers/demo/simple_text_demo.py --timeout 600 -k "not performance-ci-stress-1" || fail=1
    echo "LOG_METAL: Llama3 tests for $hf_model completed"
  done

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi

}

run_falcon7b_perf() {

  # Falcon7b (perf verification for 128/1024/2048 seq lens and output token verification)
  $PYTEST_CMD -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b_common/demo/input_data.json' models/demos/wormhole/falcon7b/demo_wormhole.py

}

run_mamba_perf() {

  $PYTEST_CMD -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/wormhole/mamba/demo/prompts.json' models/demos/wormhole/mamba/demo/demo.py --timeout 420

}

run_whisper_perf() {

  # Whisper conditional generation
  $PYTEST_CMD models/demos/whisper/demo/demo.py --input-path="models/demos/whisper/demo/dataset/conditional_generation" -k "conditional_generation"

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
 $PYTEST_CMD models/demos/mobilenetv2/demo/demo.py

}

run_yolov8s_world_perf() {

  # yolov8s_world demo
  $PYTEST_CMD models/demos/yolov8s_world/demo/demo.py


}

run_vanilla_unet_demo() {
 # vanilla_unet demo
 $PYTEST_CMD models/demos/vanilla_unet/demo/demo.py
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
  $PYTEST_CMD models/demos/wormhole/vgg_unet/demo/demo.py
}


run_yolov12x_demo() {

  $PYTEST_CMD models/demos/yolov12x/demo/demo.py

}


run_vovnet_demo(){

 $PYTEST_CMD models/experimental/vovnet/demo/demo.py

}

run_vit_demo(){

 $PYTEST_CMD models/demos/wormhole/vit/tests/test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py

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
