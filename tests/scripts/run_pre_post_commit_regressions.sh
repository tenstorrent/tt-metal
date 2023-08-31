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
  env python tests/scripts/run_tt_eager.py
else

  ./tests/scripts/run_python_api_unit_tests_wormhole_b0.sh
  ./build/test/tt_metal/test_bcast --arch $ARCH_NAME
  ./build/test/tt_metal/test_reduce_hw --arch $ARCH_NAME
  ./build/test/tt_metal/test_reduce_w --arch $ARCH_NAME
  ./build/test/tt_metal/test_reduce_h --arch $ARCH_NAME
  ./build/test/tt_metal/test_unpack_tilize --arch $ARCH_NAME
  ./build/test/tt_metal/test_unpack_untilize --arch $ARCH_NAME
  ./build/test/tt_metal/test_matmul_single_core_small --arch $ARCH_NAME
  ./build/test/tt_metal/test_matmul_single_core --arch $ARCH_NAME
fi

./build/test/tt_metal/unit_tests # Runs both on WH/GS on CI
./build/test/tt_metal/gtest_unit_tests

echo "Checking docs build..."

cd $TT_METAL_HOME/docs
python -m pip install -r requirements-docs.txt
make clean
make html
