#!/bin/bash

set -eo pipefail

validate_env_vars() {
    if [[ -z "$ARCH_NAME" ]]; then
      echo "Must provide ARCH_NAME in environment" 1>&2
      exit 1
    fi
}

set_up_end_to_end_tests_env() {
  cd tests/end_to_end_tests

  python3 -m venv env

  source env/bin/activate

  python -m pip install -r requirements.txt
  python -m pip install ../../metal_libs-*.whl
}

main() {
  validate_env_vars

  set_up_end_to_end_tests_env
}

main "$@"
