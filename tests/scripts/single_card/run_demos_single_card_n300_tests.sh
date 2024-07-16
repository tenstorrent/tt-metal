#/bin/bash

run_n300_falcon7b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon7b_tests"

  # Perf verification for 128/1024/2048 seq lens
  pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b/demo/input_data.json' models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[user_input0-perf_mode_128_stochastic_verify] ; fail+=$?
  pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b/demo/input_data.json' models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[user_input0-perf_mode_1024_stochastic_verify] ; fail+=$?
  pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b/demo/input_data.json' models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[user_input0-perf_mode_2048_stochastic_verify] ; fail+=$?
  # Output token verification for 32 user prompts
  pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b/demo/input_data.json' models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[user_input0-default_mode_1024_greedy_verify] ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_n300_falcon7b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

# Run tests working on both N150 and N300
source tests/scripts/single_card/run_demos_single_card_n150_tests.sh

# Falcon7B N300 demo tests
run_n300_falcon7b_tests

# Not working on N150, working on N300
unset WH_ARCH_YAML
rm -rf built
pytest -n auto --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py -k batch_7
