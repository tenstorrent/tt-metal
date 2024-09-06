
#/bin/bash

run_t3000_perplexity_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_perplexity_tests"

  # Falcon7B perplexity tests
  # WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/falcon7b_common/tests/perplexity/test_perplexity_falcon.py --timeout=1500 ; fail+=$?

  # Falcon40B perplexity tests
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/falcon40b/tests/test_perplexity_falcon.py -k prefill --timeout=2100 ; fail+=$?

  # Llama-70B perplexity tests
  # WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/llama2_70b/demo/eval_t3000.py --timeout=7200 ; fail+=$?

  # # Mixtral8x7B perplexity tests
  # WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_perplexity.py --timeout=3600 ; fail+=$?
  # WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_topk.py --timeout=3600 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_perplexity_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}


run_t3000_tests() {

  # Run perplexity tests
  run_t3000_perplexity_tests

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

  run_t3000_perplexity_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
