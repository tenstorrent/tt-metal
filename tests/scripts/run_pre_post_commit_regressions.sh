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

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

export PYTHONPATH=$TT_METAL_HOME

env python tests/scripts/run_build_kernels_for_riscv.py --tt-arch $ARCH_NAME
env pytest tests/tt_metal/llrt --tt-arch $ARCH_NAME -m post_commit

if [ "$ARCH_NAME" == "grayskull" ]; then
  ./tests/scripts/run_python_api_unit_tests.sh
  env python tests/scripts/run_tt_metal.py

  # Tests profiler module FW side
  ./tests/scripts/run_profiler_regressions.sh FW
else
  ./build/test/tt_metal/test_bcast --arch $ARCH_NAME
  ./build/test/tt_metal/test_reduce_hw --arch $ARCH_NAME
  ./build/test/tt_metal/test_reduce_w --arch $ARCH_NAME
  ./build/test/tt_metal/test_reduce_h --arch $ARCH_NAME
fi

./build/test/tt_metal/unit_tests
./build/test/tt_metal/gtest_unit_tests

echo "Checking docs build..."

cd $TT_METAL_HOME/docs
python -m pip install -r requirements-docs.txt
make clean
make html
