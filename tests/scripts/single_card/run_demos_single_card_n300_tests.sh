#/bin/bash

run_n300_falcon7b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon7b_tests"

  # Perf verification for 128/1024/2048 seq lens
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b_common/demo/input_data.json' models/demos/wormhole/falcon7b/demo_wormhole.py ; fail+=$?

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

source tests/scripts/single_card/run_demos_single_card_wh_common.sh

# Falcon7B N300 demo tests
run_n300_falcon7b_tests

# Not working on N150, working on N300
unset WH_ARCH_YAML
rm -rf built
pytest -n auto --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py -k batch_7
