#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

export PYTHONPATH=$TT_METAL_HOME

if [ "$ARCH_NAME" == "grayskull" ]; then
  ./tests/scripts/run_python_api_unit_tests.sh
  env python tests/scripts/run_tt_metal.py
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

# Tests profiler module FW side
# NOTE: Keep this test last as it requires a fresh ENABLE_PROFILER=1 build
echo "Run profiler regression"
cd $TT_METAL_HOME
./tests/scripts/run_profiler_regressions.sh PROFILER
