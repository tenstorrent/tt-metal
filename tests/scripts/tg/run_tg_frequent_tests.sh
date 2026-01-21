#!/bin/bash

run_tg_llama3_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_tg_llama3_tests"

  # Llama3.3-70B
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/

  # Run all Llama3 tests for 8B, 1B, and 3B weights
  # for llama_dir in "$llama1b" "$llama3b" "$llama8b" "$llama11b" "$llama70b"; do
  for llama_dir in "$llama70b"; do
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest --timeout 1800 -n auto models/demos/llama3_70b_galaxy/tests/test_llama_model_nd.py --timeout=1800 ; fail+=$?
    LLAMA_DIR=$llama_dir FAKE_DEVICE=TG pytest --timeout 1800 -n auto models/demos/llama3_70b_galaxy/tests/test_llama_model.py -k full --timeout=1800 ; fail+=$?
    echo "LOG_METAL: Llama3 tests for $llama_dir completed"
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_tg_llama3_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_tg_tests() {

  if [[ "$1" == "llama3" ]]; then
    echo "LOG_METAL: running Llama3 run_tg_frequent_tests"
    run_tg_llama3_tests

  elif [[ "$1" == "resnet50" ]]; then
    echo "LOG_METAL: running resnet50 run_tg_frequent_tests"
    pytest -n auto models/demos/ttnn_resnet/tests/test_resnet50_performant.py ; fail+=$?

  elif [[ "$1" == "unit" ]]; then
    echo "LOG_METAL: running unit/distributed run_tg_frequent_tests"
    ## Force IRAM enabled because these tests mixes fabric and non-fabric ccl tests. The IRAM setting must be consistent
    ## due to the erisc kernel wrapper being affected, and that kernel being persistent through the workload.
    ## The jit build also has different behaviour for IRAM enabled/disabled so we enable it globally.
    TT_METAL_ENABLE_ERISC_IRAM=1 pytest -n auto tests/ttnn/distributed/test_data_parallel_example_TG.py --timeout=900 ; fail+=$?
    TT_METAL_ENABLE_ERISC_IRAM=1 pytest -n auto tests/ttnn/distributed/test_multidevice_TG.py --timeout=900 ; fail+=$?
    TT_METAL_ENABLE_ERISC_IRAM=1 pytest -n auto tests/ttnn/unit_tests/base_functionality/test_multi_device_trace_TG.py --timeout=900 ; fail+=$?

  elif [[ "$1" == "sd35" ]]; then
    echo "LOG_METAL: running stable diffusion 3.5 Large run_tg_frequent_tests"
    pytest -n auto models/experimental/tt_dit/tests/models/sd35/test_vae_sd35.py -k "tg" --timeout=300; fail+=$?
    pytest -n auto models/experimental/tt_dit/tests/models/sd35/test_attention_sd35.py -k "4x4sp0tp1" --timeout=300; fail+=$?
    pytest -n auto models/experimental/tt_dit/tests/models/sd35/test_transformer_sd35.py::test_sd35_transformer_block -k "4x4sp0tp1" --timeout=300; fail+=$?

  elif [[ "$1" == "flux1" ]]; then
    echo "LOG_METAL: running Flux.1 run_tg_frequent_tests"
    HF_HUB_CACHE=/mnt/MLPerf/huggingface/hub pytest -n auto models/experimental/tt_dit/tests/blocks/test_attention.py::test_attention_flux -k "4x" --timeout=300; fail+=$?
    HF_HUB_CACHE=/mnt/MLPerf/huggingface/hub pytest -n auto models/experimental/tt_dit/tests/models/flux1/test_transformer_flux1.py::test_single_transformer_block -k "4x" --timeout=300; fail+=$?
    HF_HUB_CACHE=/mnt/MLPerf/huggingface/hub pytest -n auto models/experimental/tt_dit/tests/blocks/test_transformer_block.py::test_transformer_block_flux -k "4x" --timeout=300; fail+=$?

  elif [[ "$1" == "motif" ]]; then
    echo "LOG_METAL: running Motif run_tg_frequent_tests"
    HF_HUB_CACHE=/mnt/MLPerf/huggingface/hub pytest -n auto models/experimental/tt_dit/tests/blocks/test_attention.py::test_attention_motif -k "4x" --timeout=300; fail+=$?
    HF_HUB_CACHE=/mnt/MLPerf/huggingface/hub pytest -n auto models/experimental/tt_dit/tests/blocks/test_transformer_block.py::test_transformer_block_motif -k "4x" --timeout=300; fail+=$?

  elif [[ "$1" == "wan22" ]]; then
    echo "LOG_METAL: running Wan2.2 run_tg_frequent_tests"
    export TT_DIT_CACHE_DIR="/tmp/TT_DIT_CACHE"
    pytest -n auto models/experimental/tt_dit/tests/models/wan2_2/test_rope.py -k "wh_4x8sp1tp0"; fail+=$?
    pytest -n auto models/experimental/tt_dit/tests/models/wan2_2/test_attention_wan.py -k "wh_4x8sp1tp0"; fail+=$?
    pytest -n auto models/experimental/tt_dit/tests/models/wan2_2/test_transformer_wan.py -k "transformer_block and wh_4x8sp1tp0 or short_seq-wh_4x8sp1tp0 and not yes_load_cache and not model_caching"; fail+=$?
    pytest -n auto models/experimental/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py -k "test_wan_decoder and 4x8 and real_weights and check_output and _1f"; fail+=$?

  elif [[ "$1" == "qwenimage" ]]; then
    echo "LOG_METAL: running QwenImage run_tg_frequent_tests"
    pytest -n auto models/experimental/tt_dit/tests/encoders/qwen25vl/test_qwen25vl.py::test_qwen25vl_encoder_pair -k "4x8"; fail+=$?

  elif [[ "$1" == "mochi" ]]; then
    echo "LOG_METAL: running mochi run_tg_frequent_tests"
    FAKE_DEVICE=TG pytest -n auto models/experimental/tt_dit/tests/models/mochi/test_vae_mochi.py -k "decoder and 4links-load_dit-large_latent or conv3d_1x1x1 or -4links-l768" --timeout=1500; fail+=$?
    pytest -n auto models/experimental/tt_dit/tests/models/mochi/test_attention_mochi.py -k "short_seq and 4x8"; fail+=$?
    pytest -n auto models/experimental/tt_dit/tests/models/mochi/test_transformer_mochi.py -k "4x8 and short_seq and not yes_load_cache and not model_caching"; fail+=$?

  else
    echo "LOG_METAL: Unknown model type: $1"
    return 1
  fi

  if [[ $fail -ne 0 ]]; then
    echo "LOG_METAL: run_tg_frequent_tests failed"
    exit 1
  fi

}

main() {
  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  # Parse the arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      --model)
        model=$2
        shift
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
  done

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_tg_tests "$model"
}

main "$@"
