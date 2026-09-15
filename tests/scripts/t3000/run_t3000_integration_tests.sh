#!/bin/bash
set -eo pipefail

TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache

run_t3000_ethernet_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ethernet_tests"

  pytest tests/tt_metal/microbenchmarks/ethernet/test_ethernet_bidirectional_bandwidth_microbenchmark.py ; fail+=$?
  pytest tests/tt_metal/microbenchmarks/ethernet/test_ethernet_link_ping_latency.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ethernet_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_70b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_70b_tests"

  # Run test_model (decode and prefill) for llama3 70B
  llama70b=meta-llama/Llama-3.1-70B-Instruct
  tt_cache_llama70b=$TT_CACHE_HOME/$llama70b
  HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest models/tt_transformers/tests/test_model.py -k full ; fail+=$?
  HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest models/tt_transformers/tests/test_model_prefill.py -k "performance and not accuracy" ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_70b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

# run_t3000_llama3.2-11b-vision_freq_tests() {
# # Record the start time
#  fail=0
#  start_time=$(date +%s)

#  echo "LOG_METAL: Running run_t3000_llama3.2-11b-vision_freq_tests"

# # Llama3.2-11B
#  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct
#  tt_cache_llama11b=$TT_CACHE_HOME/$llama11b

#  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b  pytest models/tt_transformers/tests/multimodal/test_llama_image_transformer.py ; fail+=$?
#  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest models/tt_transformers/tests/multimodal/test_llama_vision_encoder.py ; fail+=$?
#  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_text.py ; fail+=$?
#  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_vision.py ; fail+=$?

# # Record the end time
#  end_time=$(date +%s)
#  duration=$((end_time - start_time))
#  echo "LOG_METAL: run_t3000_llama3.2-11b-vision_freq_tests $duration seconds to complete"
#  if [[ $fail -ne 0 ]]; then
#    exit 1
#  fi
# }

# run_t3000_spoof_n300_llama3.2-11b-vision_freq_tests() {
#  # Record the start time
#  fail=0
#  start_time=$(date +%s)

#  echo "LOG_METAL: Running run_t3000_spoof_n300_llama3.2-11b-vision_freq_tests"

#  # Llama3.2-11B
#  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct
#  tt_cache_llama11b=$TT_CACHE_HOME/$llama11b
#   # Use MESH_DEVICE env variable to run on an N300 mesh
#  mesh_device=N300

#  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest models/tt_transformers/tests/multimodal/test_llama_image_transformer.py ; fail+=$?
#  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest models/tt_transformers/tests/multimodal/test_llama_vision_encoder.py ; fail+=$?
#  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_text.py ; fail+=$?
#  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_vision.py ; fail+=$?

#  # Record the end time
#  end_time=$(date +%s)
#  duration=$((end_time - start_time))
#  echo "LOG_METAL: run_t3000_spoof_n300_llama3.2-11b-vision_freq_tests $duration seconds to complete"
#  if [[ $fail -ne 0 ]]; then
#    exit 1
#  fi
# }

run_t3000_tteager_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_tteager_tests"

  # distributed layernorm
  pytest tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_tteager_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_trace_stress_tests() {
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_trace_stress_tests"
  NUM_TRACE_LOOPS=15 pytest tests/ttnn/unit_tests/base_functionality/test_multi_device_trace.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))

  echo "LOG_METAL: run_t3000_trace_stress_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tests() {
  # Run ethernet tests
  run_t3000_ethernet_tests

  # Run tteager tests
  run_t3000_tteager_tests

  # Run trace tests
  run_t3000_trace_stress_tests

  # Run llama3-70b tests
  run_t3000_llama3_70b_tests

  # Run Llama3.2-11B Vision tests
  # run_t3000_llama3.2-11b-vision_freq_tests

  # Run Llama3.2-11B Vision tests on spoofed N300
  # run_t3000_spoof_n300_llama3.2-11b-vision_freq_tests

  # Run trace tests
  run_t3000_trace_stress_tests
}

fail=0
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
  export PYTHONPATH=$TT_METAL_HOME

  run_t3000_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
