#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [[ $dispatch_mode == "slow" ]]; then
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_frequent
  echo "Running Python API unit tests in SD for frequent..."
  ./tests/scripts/run_python_api_unit_tests.sh
else
  if [[ $tt_arch == "wormhole_b0" ]]; then
    pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_all_gather.py -k nightly
  else
    echo "Frequent API tests are not available for fast dispatch because they're already covered in post-commit"
  fi
fi
