
#/bin/bash
set -eo pipefail

source ./t3000_demo_tests_func.sh

run_t3000_tests() {
  # Run falcon40b tests
  run_t3000_falcon40b_tests

  # Run falcon7b tests
  run_t3000_falcon7b_tests

  # Run mixtral tests
  run_t3000_mixtral_tests
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

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_t3000_tests
}

main "$@"
