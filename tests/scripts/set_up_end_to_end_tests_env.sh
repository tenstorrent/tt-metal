#!/bin/bash

set -eo pipefail

validate_env_vars() {
  if [[ -n "${TT_METAL_HOME}" || -n "${PYTHONPATH}" || -n "${ARCH_NAME}" ]]; then
    echo "TT_METAL_HOME / PYTHONPATH / ARCH_NAME is set. This is not allowed in production environments"
    exit 1
  fi
}

set_up_end_to_end_tests_env() {
  cd tests/end_to_end_tests

  python3 -m venv env

  source env/bin/activate

  python -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu

  python -m pip install -r requirements.txt
  python -m pip install ../../metal_libs-*.whl

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
