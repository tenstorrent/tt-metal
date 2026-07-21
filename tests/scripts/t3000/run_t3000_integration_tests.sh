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

run_t3000_resnet_tests() {
  fail=0
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_resnet_tests"

  pytest models/demos/vision/classification/resnet50/ttnn_resnet/tests/test_resnet50_performant.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_resnet_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_dit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)
  test_name=${FUNCNAME[1]}

  echo "LOG_METAL: Running ${test_name}"

  # Run test_model for sd35 large
  for test_cmd in "$@"; do
    pytest ${test_cmd} ; fail+=$?
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: ${test_name} $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_sd35large_tests() {
  run_t3000_dit_tests \
    "models/tt_dit/tests/models/sd35/test_vae_sd35.py -k t3k" \
    "models/tt_dit/tests/models/sd35/test_attention_sd35.py" \
    "models/tt_dit/tests/models/sd35/test_transformer_sd35.py::test_sd35_transformer_block"
}

run_t3000_flux1_tests() {
  run_t3000_dit_tests \
    "models/tt_dit/tests/blocks/test_attention.py::test_attention_flux" \
    "models/tt_dit/tests/blocks/test_transformer_block.py::test_transformer_block_flux -k 2x4"
}

run_t3000_motif_tests() {
  run_t3000_dit_tests \
    "models/tt_dit/tests/blocks/test_attention.py::test_attention_motif" \
    "models/tt_dit/tests/blocks/test_transformer_block.py::test_transformer_block_motif"
}

run_t3000_qwenimage_tests() {
  run_t3000_dit_tests \
    "models/tt_dit/tests/encoders/qwen25vl/test_qwen25vl.py::test_qwen25vl_encoder_pair -k 2x4"
}

run_t3000_wan22_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_wan22_tests"

  # Run test_model for Wan2.2
  export TT_DIT_CACHE_DIR="/tmp/TT_DIT_CACHE"
  pytest models/tt_dit/tests/models/wan2_2/test_rope.py -k "2x4"; fail+=$?
  pytest models/tt_dit/tests/models/wan2_2/test_attention_wan.py -k "2x4sp0tp1"; fail+=$?
  pytest models/tt_dit/tests/models/wan2_2/test_transformer_wan.py -k "transformer_block and 2x4sp0tp1 or short_seq-2x4sp0tp1 and not yes_load_cache and not model_caching" --timeout 600; fail+=$?
  pytest models/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py -k "((test_wan_encoder or test_wan_decoder) and 2x4 and real_weights and check_output and _1f and chunk_1) or (test_wan_decoder_chunked_consistency and 2x4 and bf16 and 5f and 480p)"; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_wan22_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_mochi_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mochi_tests"

  export TT_DIT_CACHE_DIR="/tmp/TT_DIT_CACHE"
  FAKE_DEVICE=T3K pytest models/tt_dit/tests/models/mochi/test_vae_mochi.py -k "(decoder and 1x8 and load_dit and small_latent) or conv3d_1x1x1 or (1x8 and l768 and bf16)" --timeout=1500; fail+=$?
  pytest models/tt_dit/tests/models/mochi/test_attention_mochi.py -k "short_seq"; fail+=$?
  pytest models/tt_dit/tests/models/mochi/test_transformer_mochi.py -k "1x8 or 2x4 and short_seq and not yes_load_cache and not model_caching"; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mochi_tests $duration seconds to complete"
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

  # Run resnet tests
  run_t3000_resnet_tests

  # Run sd35_large tests
  run_t3000_sd35large_tests

  # Run flux1 tests
  run_t3000_flux1_tests

  # Run motif tests
  run_t3000_motif_tests

  # Run trace tests
  run_t3000_trace_stress_tests

  # Run wan22 tests
  run_t3000_wan22_tests

  # Run mochi tests
  run_t3000_mochi_tests

  # Run qwenimage tests
  run_t3000_qwenimage_tests
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
