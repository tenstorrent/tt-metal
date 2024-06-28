
#/bin/bash
set -eo pipefail

run_t3000_ethernet_tests() {
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ethernet_tests"

  pytest tests/tt_metal/microbenchmarks/ethernet/test_ethernet_bidirectional_bandwidth_microbenchmark.py
  pytest tests/tt_metal/microbenchmarks/ethernet/test_ethernet_ring_latency_microbenchmark.py

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ethernet_tests $duration seconds to complete"
}

run_t3000_llama2_70b_tests() {
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama2_70b_tests"

  env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/llama2_70b/tests/test_llama_mlp_t3000.py
  env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/llama2_70b/tests/test_llama_attention_t3000.py
  env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/llama2_70b/tests/test_llama_decoder_t3000.py
  env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/llama2_70b/tests/test_llama_model_t3000.py --timeout=900

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama2_70b_tests $duration seconds to complete"
}

run_t3000_mixtral_tests() {
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mixtral_tests"

  # mixtral8x7b 8 chip decode model test (env flags set inside the test)
  pytest models/demos/t3000/mixtral8x7b/tests/test_mixtral_model.py::test_mixtral_model_inference[wormhole_b0-True-10-1-pcc]

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mixtral_tests $duration seconds to complete"
}

run_t3000_tteager_tests() {
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_tteager_tests"

  pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_all_gather.py -k post_commit
  pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_reduce_scatter_post_commit.py

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_tteager_tests $duration seconds to complete"
}

run_t3000_trace_stress_tests() {
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_trace_stress_tests"

  NUM_TRACE_LOOPS=15 pytest tests/ttnn/unit_tests/test_multi_device_trace.py
  NUM_TRACE_LOOPS=15 WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/ttnn/unit_tests/test_multi_device_trace.py

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_trace_stress_tests $duration seconds to complete"
}


run_t3000_falcon40b_tests() {
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon40b_tests"

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/falcon40b/tests/test_falcon_mlp.py
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/falcon40b/tests/test_falcon_attention.py --timeout=480
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/falcon40b/tests/test_falcon_decoder.py --timeout=480
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/falcon40b/tests/test_falcon_causallm.py --timeout=600

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon40b_tests $duration seconds to complete"
}

run_t3000_tests() {
  # Run ethernet tests
  #run_t3000_ethernet_tests

  # Run tteager tests
  run_t3000_tteager_tests

  # Run trace tests
  run_t3000_trace_stress_tests

  # Run falcon40b tests
  run_t3000_falcon40b_tests

  # Run llama2-70b tests
  run_t3000_llama2_70b_tests

  # Run mixtral tests
  run_t3000_mixtral_tests

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
  export PYTHONPATH=$TT_METAL_HOME

  run_t3000_tests
}

main "$@"
