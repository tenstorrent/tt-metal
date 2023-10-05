#!/bin/bash
# set -x
set -eo pipefail

run_profiling_test(){
  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  echo "Make sure this test runs in a build with ENABLE_PROFILER=1"
  source build/python_env/bin/activate
  export PYTHONPATH=$TT_METAL_HOME

  env python $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py
}

run_profiling_test
