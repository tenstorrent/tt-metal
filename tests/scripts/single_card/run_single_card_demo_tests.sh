#!/bin/bash
set -e

run_falcon7b_func() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo -k "default_mode_1024_stochastic"

}

run_mistral7b_func() {

  mistral7b=/mnt/MLPerf/tt_dnn-models/Mistral/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db
  mistral_cache=/mnt/MLPerf/tt_dnn-models/Mistral/TT_CACHE/Mistral-7B-Instruct-v0.3
  HF_MODEL=$mistral7b TT_CACHE_PATH=$mistral_cache WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/tt_transformers/tests/test_accuracy.py -k perf --timeout 1200; fail+=$?

}

run_qwen7b_func() {

  HF_MODEL=/mnt/MLPerf/tt_dnn-models/qwen/Qwen2-7B-Instruct WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml MESH_DEVICE=N300 pytest -n auto models/tt_transformers/demo/simple_text_demo.py -k performance-ci-1 --timeout 1800

}

run_segformer_func() {

  #Segformer Segmentation Demo
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings models/demos/segformer/demo/demo_for_semantic_segmentation.py --timeout 600; fail+=$?

  #Segformer Classification Demo
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings models/demos/segformer/demo/demo_for_image_classification.py --timeout 600; fail+=$?

}

run_sentencebert_func() {

  #SentenceBERT Demo
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings models/demos/sentence_bert/demo/demo.py::test_sentence_bert_demo_inference --timeout 600; fail+=$?

  #SentenceBERT eval
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings models/demos/sentence_bert/demo/dataset_evaluation.py::test_sentence_bert_eval --timeout 600; fail+=$?

}

run_yolov11_func() {

  #Yolov11 Demo
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings models/demos/yolov11/demo/demo.py --timeout 600; fail+=$?

}

run_llama3_func() {
  fail=0

  # Llama3 Accuracy tests
  # Llama3.2-1B
  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  # Llama3.2-3B
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  # Llama3.1-8B (11B weights are the same)
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/

  # Run Llama3 accuracy tests for 1B, 3B, 8B weights
  for llama_dir in "$llama1b" "$llama3b" "$llama8b"; do
    LLAMA_DIR=$llama_dir WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/tt_transformers/tests/test_accuracy.py -k perf --timeout 420 || fail=1
    echo "LOG_METAL: Llama3 accuracy tests for $llama_dir completed"
  done

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi

}

run_ufld_v2_func() {
  #ufld_v2
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/ufld_v2/demo/demo.py --timeout 600
  #ufld_v2 eval
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/ufld_v2/demo/dataset_evaluation.py --timeout 1500
}
run_vgg_func() {

  #VGG11/VGG16
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/vgg/demo/demo.py --timeout 600

}

run_bert_tiny_func() {
  fail=0

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/bert_tiny/demo/demo.py --timeout 600 || fail=1

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/wormhole/bert_tiny/demo/demo.py --timeout 600 || fail=1

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi

}

run_bert_func() {
  fail=0

  pytest -n auto --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py -k batch_7 || fail=1
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py -k batch_8 || fail=1

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi

}

run_resnet_stability() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings models/demos/wormhole/resnet50/tests/test_resnet50_stability.py -k "short"

}

run_resnet_func() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings models/demos/wormhole/resnet50/demo/demo.py

}

run_sdxl_func() {
  pytest --disable-warnings models/experimental/stable_diffusion_xl_base/tests/test_sdxl_accuracy.py --start-from=0 --num-prompts=2
}

run_distilbert_func() {
  fail=0

  pytest --disable-warnings models/demos/distilbert/demo/demo.py --timeout 600 || fail=1

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings models/demos/wormhole/distilbert/demo/demo.py --timeout 600 || fail=1

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi

}

run_covnet_mnist_func() {

  pytest --disable-warnings models/demos/convnet_mnist/demo/demo.py --timeout 600

}

run_mnist_func() {

  pytest --disable-warnings models/demos/mnist/demo/demo.py --timeout 600

}

run_squeezebert_func() {

  pytest --disable-warnings models/demos/squeezebert/demo/demo.py --timeout 600

}

run_roberta_func() {

  pytest --disable-warnings models/demos/roberta/demo/demo.py --timeout 600

}

run_stable_diffusion_func() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings --input-path="models/demos/wormhole/stable_diffusion/demo/input_data.json" models/demos/wormhole/stable_diffusion/demo/demo.py::test_demo --timeout 900

}

run_mistral7b_perf() {

  # To ensure a proper perf measurement and dashboard upload of Mistral-7B N150, we have to run them on the N300 perf pipeline for now
  mistral7b=/mnt/MLPerf/tt_dnn-models/Mistral/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db
  mistral_cache=/mnt/MLPerf/tt_dnn-models/Mistral/TT_CACHE/Mistral-7B-Instruct-v0.3
  # Run Mistral-7B-v0.3 for N150
  MESH_DEVICE=N150 HF_MODEL=$mistral7b TT_CACHE_PATH=$mistral_cache WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/tt_transformers/demo/simple_text_demo.py --timeout 600 -k "not performance-ci-stress-1"; fail+=$?
  # Run Mistral-7B-v0.3 for N300
  MESH_DEVICE=N300 HF_MODEL=$mistral7b TT_CACHE_PATH=$mistral_cache WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/tt_transformers/demo/simple_text_demo.py --timeout 600 -k "not performance-ci-stress-1"; fail+=$?

}

run_llama3_perf() {
  fail=0

  # Llama3.2-1B
  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  # Llama3.2-3B
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  # Llama3.1-8B
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/
  # Llama3.2-11B (same tet weights as 8B)
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

  # Run all Llama3 tests for 1B, 3B, 8B weights for N150
  # To ensure a proper perf measurement and dashboard upload of the Llama3 models on a N150, we have to run them on the N300 perf pipeline for now
  for llama_dir in "$llama1b" "$llama3b" "$llama8b"; do
    MESH_DEVICE=N150 LLAMA_DIR=$llama_dir WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/tt_transformers/demo/simple_text_demo.py --timeout 600 -k "not performance-ci-stress-1" || fail=1
    echo "LOG_METAL: Llama3 tests for $llama_dir completed on N150"
  done
  # Run all Llama3 tests for 1B, 3B, 8B and 11B weights
  for llama_dir in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    LLAMA_DIR=$llama_dir WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/tt_transformers/demo/simple_text_demo.py --timeout 600 -k "not performance-ci-stress-1" || fail=1
    echo "LOG_METAL: Llama3 tests for $llama_dir completed"
  done

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi

}

run_falcon7b_perf() {

  # Falcon7b (perf verification for 128/1024/2048 seq lens and output token verification)
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b_common/demo/input_data.json' models/demos/wormhole/falcon7b/demo_wormhole.py

}

run_mamba_perf() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/wormhole/mamba/demo/prompts.json' models/demos/wormhole/mamba/demo/demo.py --timeout 420

}

run_whisper_perf() {

  # Whisper conditional generation
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/whisper/demo/demo.py --input-path="models/demos/whisper/demo/dataset/conditional_generation" -k "conditional_generation"

}

run_yolov9c_perf() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings models/demos/yolov9c/demo/demo.py --timeout 600

}
run_yolov8s_perf() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings models/demos/yolov8s/demo/demo.py --timeout 600

}


run_mobilenetv2_perf(){

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/mobilenetv2/demo/demo.py::test_mobilenetv2_imagenet_demo --timeout 600

}

run_yolov8s_world_perf() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings models/demos/yolov8s_world/demo/demo.py --timeout 600

}

run_vanilla_unet_demo() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/experimental/vanilla_unet/demo/demo.py::test_unet_demo_single_image

}
run_yolov8x_perf() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings models/demos/yolov8x/demo/demo.py --timeout 600

}
run_yolov4_perf() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings models/demos/yolov4/demo.py --timeout 600
}
run_yolov10x_demo() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings  models/demos/yolov10x/demo/demo.py --timeout 600

}

run_yolov7_demo() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/yolov7/demo/demo.py --timeout 600

}

run_vgg_unet_demo() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/vgg_unet/demo/demo.py --timeout 600

}


run_yolov12x_demo() {

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings  models/experimental/yolov12x/demo/demo.py --timeout 600

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
