#!/bin/bash
set -eo pipefail

run_ttnn_sweeps() {
  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  export PYTHONPATH=$TT_METAL_HOME
  source python_env/bin/activate

  python tests/ttnn/sweep_tests/run_sweeps.py
}

run_ttnn_sweeps
