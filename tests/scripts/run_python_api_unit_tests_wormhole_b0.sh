#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi


# This must run in slow dispatch mode
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/unit_testing/test_bmm_tilize_untilize.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_broadcast.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_composite.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_eltwise_binary.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_eltwise_ternary.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_eltwise_unary.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_outer.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_pad.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_permute.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_reduce.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_stats.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_sum.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_transpose.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_tilize.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_untilize.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest ${TT_METAL_HOME}/tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_tilize_with_zero_padding.py
