#!/bin/bash
set -eo pipefail

TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache

run_t3000_ethernet_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ethernet_tests"

  pytest -n auto tests/tt_metal/microbenchmarks/ethernet/test_ethernet_bidirectional_bandwidth_microbenchmark.py ; fail+=$?
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
  llama1b=meta-llama/Llama-3.2-1B-Instruct
  # Llama3.2-3B
  llama3b=meta-llama/Llama-3.2-3B-Instruct
  # Llama3.1-8B
  llama8b=meta-llama/Llama-3.1-8B-Instruct
  # Llama3.2-11B
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct

  # Run test model for llama3 - 1B, 3B, 8B and 11B weights
  for hf_model in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    tt_cache=$TT_CACHE_HOME/$hf_model
    HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/tests/test_model.py -k full ; fail+=$?
    HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/tests/test_model_prefill.py ; fail+=$?
    echo "LOG_METAL: Llama3 tests for $hf_model completed"
  done

  # Run chunked prefill test for llama3-1B
  tt_cache_llama1b=$TT_CACHE_HOME/$llama1b
  HF_MODEL=$llama1b TT_CACHE_PATH=$tt_cache_llama1b pytest models/tt_transformers/tests/test_chunked_generation.py; fail+=$?

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
  llama70b=meta-llama/Llama-3.1-70B-Instruct
  tt_cache_llama70b=$TT_CACHE_HOME/$llama70b
  HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest -n auto models/tt_transformers/tests/test_model.py -k full ; fail+=$?
  HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest -n auto models/tt_transformers/tests/test_model_prefill.py ; fail+=$?

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
  llama90b=meta-llama/Llama-3.2-90B-Vision-Instruct
  tt_cache_llama90b=$TT_CACHE_HOME/$llama90b
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/test_model.py -k quick ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/test_model_prefill.py -k "performance and 1layer" ; fail+=$?

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
  llama1b=meta-llama/Llama-3.2-1B-Instruct
  # Llama3.2-3B
  llama3b=meta-llama/Llama-3.2-3B-Instruct
  # Llama3.1-8B
  llama8b=meta-llama/Llama-3.1-8B-Instruct
  # Llama3.2-11B
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct

  # Run test accuracy llama3 - 1B, 3B, 8B, and 11B weights
  for hf_model in "$llama1b" "$llama3b" "$llama8b" "$llama11b" ; do
    tt_cache=$TT_CACHE_HOME/$hf_model
    HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-token-matching" ; fail+=$?
    echo "LOG_METAL: Llama3 accuracy tests for $hf_model completed"
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3_accuracy_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3_70n90b_accuracy_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3_accuracy_tests"

  # Llama3.1-70B
  llama70b=meta-llama/Llama-3.1-70B-Instruct
  # Llama3.2-90B
  llama90b=meta-llama/Llama-3.2-90B-Vision-Instruct

  # Run test accuracy llama3 - 1B, 3B, 8B, 11B and 70B weights
  for hf_model in "$llama70b" "$llama90b"; do
    tt_cache=$TT_CACHE_HOME/$hf_model
    HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-token-matching" --timeout 4200 ; fail+=$?
    echo "LOG_METAL: Llama3 accuracy tests for $hf_model completed"
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
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct
  tt_cache_llama11b=$TT_CACHE_HOME/$llama11b

  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b  pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_transformer.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_vision_encoder.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_text.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_vision.py ; fail+=$?

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
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct
  tt_cache_llama11b=$TT_CACHE_HOME/$llama11b
  # Use MESH_DEVICE env variable to run on an N300 mesh
  mesh_device=N300

  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_transformer.py ; fail+=$?
  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_vision_encoder.py ; fail+=$?
  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_text.py ; fail+=$?
  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_vision.py ; fail+=$?

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
  llama90b=meta-llama/Llama-3.2-90B-Vision-Instruct
  tt_cache_llama90b=$TT_CACHE_HOME/$llama90b
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_text.py --timeout 2400; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_transformer.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_vision_encoder.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_vision.py ; fail+=$?

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

  hf_model=mistralai/Mistral-7B-Instruct-v0.3
  tt_cache_path=$TT_CACHE_HOME/$hf_model
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
  NUM_TRACE_LOOPS=15 pytest -n auto tests/ttnn/unit_tests/base_functionality/test_multi_device_trace.py ; fail+=$?

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

  pytest -n auto models/demos/ttnn_resnet/tests/test_resnet50_performant.py ; fail+=$?

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
    pytest -n auto ${test_cmd} ; fail+=$?
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
    "models/experimental/tt_dit/tests/models/sd35/test_vae_sd35.py -k t3k" \
    "models/experimental/tt_dit/tests/models/sd35/test_attention_sd35.py" \
    "models/experimental/tt_dit/tests/models/sd35/test_transformer_sd35.py::test_sd35_transformer_block"
}

run_t3000_flux1_tests() {
  run_t3000_dit_tests \
    "models/experimental/tt_dit/tests/blocks/test_attention.py::test_attention_flux" \
    "models/experimental/tt_dit/tests/models/flux1/test_transformer_flux1.py::test_single_transformer_block -k 2x4" \
    "models/experimental/tt_dit/tests/blocks/test_transformer_block.py::test_transformer_block_flux -k 2x4"
}

run_t3000_motif_tests() {
  run_t3000_dit_tests \
    "models/experimental/tt_dit/tests/blocks/test_attention.py::test_attention_motif" \
    "models/experimental/tt_dit/tests/blocks/test_transformer_block.py::test_transformer_block_motif"
}

run_t3000_qwenimage_tests() {
  run_t3000_dit_tests \
    "models/experimental/tt_dit/tests/encoders/qwen25vl/test_qwen25vl.py::test_qwen25vl_encoder_pair -k 2x4"
}

run_t3000_wan22_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_wan22_tests"

  # Run test_model for Wan2.2
  export TT_DIT_CACHE_DIR="/tmp/TT_DIT_CACHE"
  pytest -n auto models/experimental/tt_dit/tests/models/wan2_2/test_rope.py -k "2x4"; fail+=$?
  pytest -n auto models/experimental/tt_dit/tests/models/wan2_2/test_attention_wan.py -k "2x4sp0tp1"; fail+=$?
  pytest -n auto models/experimental/tt_dit/tests/models/wan2_2/test_transformer_wan.py -k "transformer_block and 2x4sp0tp1 or short_seq-2x4sp0tp1 and not yes_load_cache and not model_caching" --timeout 600; fail+=$?
  pytest -n auto models/experimental/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py -k "test_wan_decoder and 2x4 and real_weights and check_output and _1f"; fail+=$?

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
  FAKE_DEVICE=T3K pytest -n auto models/experimental/tt_dit/tests/models/mochi/test_vae_mochi.py -k "decoder and 1link-load_dit-large_latent or conv3d_1x1x1 or -1link-l768" --timeout=1500; fail+=$?
  pytest -n auto models/experimental/tt_dit/tests/models/mochi/test_attention_mochi.py -k "short_seq"; fail+=$?
  pytest -n auto models/experimental/tt_dit/tests/models/mochi/test_transformer_mochi.py -k "1x8 or 2x4 and short_seq and not yes_load_cache and not model_caching"; fail+=$?

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
  run_t3000_llama3_70n90b_accuracy_tests

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
