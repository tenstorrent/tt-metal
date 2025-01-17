#!/bin/bash
set +e

run_common_func_tests() {
  # working on both n150 and n300
  fail=0

  # Falcon7B
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[wormhole_b0-True-user_input0-1-default_mode_1024_stochastic]; fail+=$?

  # Mistral7B
  # Skipping: kills WH cards, check issue #14440
  # WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/wormhole/mistral7b/demo/demo.py --timeout 420; fail+=$?

  # Qwen7B
  QWEN_DIR=/mnt/MLPerf/tt_dnn-models/qwen/Qwen2-7B-Instruct WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml FAKE_DEVICE=N150 pytest -n auto models/demos/qwen/demo/demo.py -k instruct --timeout 420; fail+=$?

  # Llama3 Accuracy tests
  # Llama3.2-1B
  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  # Llama3.2-3B
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  # Llama3.1-8B (11B weights are the same)
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/

  # Run Llama3 accuracy tests for 1B, 3B, 8B weights
  for llama_dir in "$llama1b" "$llama3b" "$llama8b"; do
    LLAMA_DIR=$llama_dir WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/llama3/tests/test_llama_accuracy.py -k perf --timeout 420; fail+=$?
    echo "LOG_METAL: Llama3 accuracy tests for $llama_dir completed"
  done

  #VGG11/VGG16
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/vgg/demo/demo.py --timeout 600; fail+=$?

  #Bert-Tiny
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/bert_tiny/demo/demo.py --timeout 600; fail+=$?

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/wormhole/bert_tiny/demo/demo.py --timeout 600; fail+=$?

  # Bert
  pytest -n auto --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py -k batch_7; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py -k batch_8; fail+=$?

  # Resnet
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings models/demos/wormhole/resnet50/demo/demo.py; fail+=$?

  # Distilbert
  pytest --disable-warnings models/demos/distilbert/demo/demo.py --timeout 600; fail+=$?

  # ConvNet Mnist
  pytest --disable-warnings models/demos/convnet_mnist/demo/demo.py --timeout 600; fail+=$?

  # Mnist
  pytest --disable-warnings models/demos/mnist/demo/demo.py --timeout 600; fail+=$?

  # SqueezeBERT
  pytest --disable-warnings models/demos/squeezebert/demo/demo.py --timeout 600; fail+=$?

  # Distilbert
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings models/demos/wormhole/distilbert/demo/demo.py --timeout 600; fail+=$?

  #RoBERTa
  pytest --disable-warnings models/demos/roberta/demo/demo.py --timeout 600; fail+=$?

  return $fail
}

run_common_perf_tests(){
  # working on both n150 and n300
  fail=0

  # Mistral7B
  # Skipping: kills WH cards, check issue #14440
  # WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/wormhole/mistral7b/demo/demo_with_prefill.py --timeout 420; fail+=$?

  # Mamba
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/wormhole/mamba/demo/prompts.json' models/demos/wormhole/mamba/demo/demo.py --timeout 420; fail+=$?

  return $fail
}

run_n150_tests(){
  fail=0

  run_common_func_tests; fail+=$?
  run_common_perf_tests; fail+=$?

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings --input-path="models/demos/wormhole/stable_diffusion/demo/input_data.json" models/demos/wormhole/stable_diffusion/demo/demo.py::test_demo --timeout 900; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_n300_func_tests() {
  fail=0;

  run_common_func_tests; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_n300_perf_tests(){
  fail=0

  run_common_perf_tests; fail+=$?

  # Llama3.1-8B
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/
  # Llama3.2-1B
  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  # Llama3.2-3B
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

  # Run all Llama3 tests for 1B, 3B, 8B and 11B weights for N150
  # To ensure a proper perf measurement and dashboard upload of the Llama3 models on a N150, we have to run them on the N300 perf pipeline for now
  for llama_dir in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    FAKE_DEVICE=N150 LLAMA_DIR=$llama_dir WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/llama3/demo/demo.py --timeout 600; fail+=$?
    echo "LOG_METAL: Llama3 tests for $llama_dir completed on N150"
  done
  # Run all Llama3 tests for 1B, 3B, 8B and 11B weights
  for llama_dir in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    LLAMA_DIR=$llama_dir WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/llama3/demo/demo.py --timeout 600; fail+=$?
    echo "LOG_METAL: Llama3 tests for $llama_dir completed"
  done

  # Falcon7b (perf verification for 128/1024/2048 seq lens and output token verification)
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b_common/demo/input_data.json' models/demos/wormhole/falcon7b/demo_wormhole.py; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
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
