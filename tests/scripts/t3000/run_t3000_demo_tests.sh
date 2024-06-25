
#/bin/bash
set -eo pipefail

run_t3000_falcon40b_tests() {
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon40b_tests"

  # Falcon40B prefill 60 layer end to end with 10 loops; we need 8x8 grid size
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/falcon40b/tests/ci/test_falcon_end_to_end_60_layer_t3000_prefill_10_loops.py

  # Falcon40B end to end demo (prefill + decode)
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/falcon40b/tests/ci/test_falcon_end_to_end_t3000_demo_loops.py

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon40b_tests $duration seconds to complete"
}

run_t3000_falcon7b_tests(){
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon7b_tests"

  # Falcon7B demo (perf verification for 128/1024/2048 seq lens and output token verification)
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/t3000/falcon7b/input_data_t3000.json' models/demos/t3000/falcon7b/demo_t3000.py::test_demo_multichip[user_input0-8-True-perf_mode_128_stochastic_verify]
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/t3000/falcon7b/input_data_t3000.json' models/demos/t3000/falcon7b/demo_t3000.py::test_demo_multichip[user_input0-8-True-perf_mode_1024_stochastic_verify]
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/t3000/falcon7b/input_data_t3000.json' models/demos/t3000/falcon7b/demo_t3000.py::test_demo_multichip[user_input0-8-True-perf_mode_2048_stochastic_verify]
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/t3000/falcon7b/input_data_t3000.json' models/demos/t3000/falcon7b/demo_t3000.py::test_demo_multichip[user_input0-8-True-default_mode_1024_greedy_verify]

  # Falcon7B perplexity test (prefill and decode)
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/falcon7b/tests/test_perplexity_falcon.py::test_perplexity[True-prefill_seq1024_dram]
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/falcon7b/tests/test_perplexity_falcon.py::test_perplexity[True-decode_1024_l1_sharded]

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon7b_tests $duration seconds to complete"
}

run_t3000_mixtral_tests() {
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mixtral8x7b_tests"

  # mixtral8x7b 8 chip demo test - 100 token generation with general weights (env flags set inside the test)
  pytest models/demos/t3000/mixtral8x7b/demo/demo.py::test_mixtral8x7b_demo[wormhole_b0-True-general_weights]

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mixtral_tests $duration seconds to complete"
}

run_t3000_tests() {
  # Run falcon40b tests
  run_t3000_falcon40b_tests

  # Run falcon7b tests
  run_t3000_falcon7b_tests

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
