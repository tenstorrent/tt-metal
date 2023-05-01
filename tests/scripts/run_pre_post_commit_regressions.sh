#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [ "$TT_METAL_ENV" != "dev" ]; then
  echo "Must set TT_METAL_ENV as dev" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

export PYTHONPATH=$TT_METAL_HOME

env ARCH_NAME=grayskull python tests/scripts/run_build_kernels_for_riscv.py -j 16
env python tests/scripts/run_llrt.py --short-driver-tests

./tests/scripts/run_python_api_unit_tests.sh
