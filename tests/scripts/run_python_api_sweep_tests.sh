#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi
export PYTHONPATH=$TT_METAL_HOME

if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
  env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/ -name 'test_*.py' -a ! -name 'test_sweep_conv_with_address_map.py') -vvv
else
  # Need to remove move for time being since failing
  # Need to run embedding separate due to tests being dependent in session
  env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/ -name 'test_*.py' -a ! -name 'test_sweep_conv_with_address_map.py' -a ! -name 'test_move.py') -vvv
fi

# Test forcing ops to single core
env TT_METAL_SINGLE_CORE_MODE=1 pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_matmul.py::test_run_matmul_test -k BFLOAT16
env TT_METAL_SINGLE_CORE_MODE=1 pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_unpad.py
