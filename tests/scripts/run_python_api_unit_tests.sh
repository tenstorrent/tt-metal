#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi
# Execute TT Eager unit and sweep tests here
env pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/ -xvvv
env pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/ -xvvv

# Execute python model tests here
./tests/scripts/run_python_model_tests.sh

env pytest $TT_METAL_HOME/tests/ttnn/unit_tests -xvvv
