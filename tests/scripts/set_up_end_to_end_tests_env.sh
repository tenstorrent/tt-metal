#!/bin/bash

set -eo pipefail

validate_env_vars() {
  if [[ -n "${TT_METAL_HOME}" || -n "${PYTHONPATH}" ]]; then
    echo "TT_METAL_HOME / PYTHONPATH is set. This is not allowed in production environments"
    exit 1
  fi
}

set_up_end_to_end_tests_env() {
  cd tests/end_to_end_tests

  python3 -m venv env

  source env/bin/activate

  python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu

  python3 -m pip install -r requirements.txt
  python3 -m pip install ../../ttnn-*.whl

  cd ../../
  rm -rf tt_metal tt_eager ttnn models
  echo "Showing current directory"
  ls -hal
}

main() {
  validate_env_vars

  set_up_end_to_end_tests_env
}

main "$@"
