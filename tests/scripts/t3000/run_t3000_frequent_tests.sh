#!/bin/bash
set -eo pipefail

run_t3000_ethernet_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ethernet_tests"

  pytest -n auto tests/tt_metal/microbenchmarks/ethernet/test_ethernet_bidirectional_bandwidth_microbenchmark.py ; fail+=$?
  pytest -n auto tests/tt_metal/microbenchmarks/ethernet/test_ethernet_ring_latency_microbenchmark.py ; fail+=$?
  pytest -n auto tests/tt_metal/microbenchmarks/ethernet/test_ethernet_link_ping_latency.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ethernet_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_tests"

  # Llama3.2-1B
  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  # Llama3.2-3B
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  # Llama3.1-8B
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/
  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

  # Run test model for llama3 - 1B, 3B, 8B and 11B weights
  for llama_dir in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/tests/test_model.py -k full ; fail+=$?
    LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/tests/test_model_prefill.py ; fail+=$?
    echo "LOG_METAL: Llama3 tests for $llama_dir completed"
  done

  # Run chunked prefill test for llama3-1B
  LLAMA_DIR=$llama1b pytest models/tt_transformers/tests/test_chunked_generation.py; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_tests $duration seconds to complete"
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
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.1-70B-Instruct/
  LLAMA_DIR=$llama70b pytest -n auto models/tt_transformers/tests/test_model.py -k full ; fail+=$?
  LLAMA_DIR=$llama70b pytest -n auto models/tt_transformers/tests/test_model_prefill.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_70b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_90b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_90b_tests"

  # Run test_model (decode and prefill) for llama3 70B
  llama90b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-90B-Vision-Instruct/
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/test_model.py -k quick ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/test_model_prefill.py -k "performance and 1layer" ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_90b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_accuracy_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_accuracy_tests"

  # Llama3.2-1B
  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  # Llama3.2-3B
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  # Llama3.1-8B
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/
  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/
  # Llama3.1-70B
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.1-70B-Instruct/
  # Llama3.2-90B
  llama90b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-90B-Vision-Instruct/

  # Run test accuracy llama3 - 1B, 3B, 8B, 11B and 70B weights
  for llama_dir in "$llama1b" "$llama3b" "$llama8b" "$llama11b" "$llama70b" "$llama90b"; do
    LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/tests/test_accuracy.py -k perf ; fail+=$?
    echo "LOG_METAL: Llama3 accuracy tests for $llama_dir completed"
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_accuracy_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3.2-11b-vision_freq_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3.2-11b-vision_freq_tests"

  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_transformer.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_vision_encoder.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_text.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_vision.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3.2-11b-vision_freq_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_spoof_n300_llama3.2-11b-vision_freq_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_spoof_n300_llama3.2-11b-vision_freq_tests"

  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/
  # Use MESH_DEVICE env variable to run on an N300 mesh
  mesh_device=N300

  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_transformer.py ; fail+=$?
  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_vision_encoder.py ; fail+=$?
  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_text.py ; fail+=$?
  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_vision.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_spoof_n300_llama3.2-11b-vision_freq_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3.2-90b-vision_freq_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3.2-90b-vision_freq_tests"

  # Llama3.2-90B -- use repacked weights when acceptable for faster testing
  llama90b_repacked=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-90B-Vision-Instruct/repacked
  LLAMA_DIR=$llama90b_repacked pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_text.py --timeout 2400; fail+=$?
  LLAMA_DIR=$llama90b_repacked pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_transformer.py ; fail+=$?
  LLAMA_DIR=$llama90b_repacked pytest -n auto models/tt_transformers/tests/multimodal/test_llama_vision_encoder.py ; fail+=$?
  LLAMA_DIR=$llama90b_repacked pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_vision.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3.2-90b-vision_freq_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}


run_t3000_mistral_tests() {

  echo "LOG_METAL: Running run_t3000_mistral_frequent_tests"

  tt_cache_path="/mnt/MLPerf/tt_dnn-models/Mistral/TT_CACHE/Mistral-7B-Instruct-v0.3"
  hf_model="/mnt/MLPerf/tt_dnn-models/Mistral/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest -n auto models/tt_transformers/tests/test_model.py -k full
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest -n auto models/tt_transformers/tests/test_model_prefill.py

}

run_t3000_mixtral_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mixtral_tests"

  # mixtral8x7b 8 chip decode model test (env flags set inside the test)
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_model.py --timeout=600; fail+=$?
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_model_prefill.py --timeout=600; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mixtral_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tteager_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_tteager_tests"

  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_all_gather.py -k post_commit ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_all_gather_matmul.py -k post_commit ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_reduce_scatter_post_commit.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_send_recv_async.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_barrier_t3000_frequent.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_all_reduce_t3000_frequent.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_all_to_all_dispatch_t3000.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_all_to_all_combine_t3000.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_mesh_partition_t3000.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_moe_ccl_end_to_end.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/operations/point_to_point/test_send_receive.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/operations/data_movement/test_moe_expert_token_remap_t3k.py ; fail+=$?

  # distributed layernorm
  pytest tests/ttnn/unit_tests/operations/test_distributed_layernorm.py ; fail+=$?

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
  NUM_TRACE_LOOPS=15 pytest -n auto tests/ttnn/unit_tests/test_multi_device_trace.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))

  echo "LOG_METAL: run_t3000_trace_stress_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_falcon40b_tests() {
  fail=0
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon40b_tests"

  pytest -n auto models/demos/t3000/falcon40b/tests/test_falcon_mlp.py ; fail+=$?
  pytest -n auto models/demos/t3000/falcon40b/tests/test_falcon_attention.py --timeout=480 ; fail+=$?
  pytest -n auto models/demos/t3000/falcon40b/tests/test_falcon_decoder.py --timeout=480 ; fail+=$?
  pytest -n auto models/demos/t3000/falcon40b/tests/test_falcon_causallm.py --timeout=600 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon40b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_resnet_tests() {
  fail=0
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_resnet_tests"

  pytest -n auto models/demos/t3000/resnet50/tests/test_resnet50_performant.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_resnet_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}


run_t3000_sd35large_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_sd35large_tests"

  # Run test_model for sd35 large
  sd35large=/mnt/MLPerf/tt_dnn-models/StableDiffusion_35_Large/
  SD35L_DIR=$sd35large pytest -n auto models/experimental/stable_diffusion_35_large/tests/test_fun_transformer_block.py -k "t3k" ; fail+=$?
  SD35L_DIR=$sd35large pytest -n auto models/experimental/stable_diffusion_35_large/tests/test_fun_patch_embedding.py -k "t3k"; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_sd35large_tests $duration seconds to complete"
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

  # Run falcon40b tests
  run_t3000_falcon40b_tests

  # Run llama3 small (1B, 3B, 8B, 11B) tests
  run_t3000_llama3_tests

  # Run llama3-70b tests
  run_t3000_llama3_70b_tests

  # Run llama3-90b tests
  run_t3000_llama3_90b_tests

  # Run llama3 accuracy tests
  run_t3000_llama3_accuracy_tests

  # Run Llama3.2-11B Vision tests
  run_t3000_llama3.2-11b-vision_freq_tests

  # Run Llama3.2-11B Vision tests on spoofed N300
  run_t3000_spoof_n300_llama3.2-11b-vision_freq_tests

  # Run Llama3.2-90B Vision tests
  run_t3000_llama3.2-90b-vision_freq_tests

  # Run mistral tests
  run_t3000_mistral_tests

  # Run mixtral tests
  run_t3000_mixtral_tests

  # Run resnet tests
  run_t3000_resnet_tests

  # Run sd35_large tests
  run_t3000_sd35large_tests

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
