#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
  echo "Only Slow Dispatch mode allowed - Must have TT_METAL_SLOW_DISPATCH_MODE set" 1>&2
  exit 1
fi

# This must run in slow dispatch mode
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_broadcast.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_composite.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_eltwise_binary.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_eltwise_ternary.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_eltwise_unary.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_matmul.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_move.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_outer.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_pad.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_permute.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_reduce.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_stats.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_sum.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_tilize.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_tilize_with_zero_padding.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_transpose.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_unpad.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_untilize.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_conv_with_dtx_cpu_sweep.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_untilize_with_unpadding.py
pytest ${TT_METAL_HOME}/tests/tt_eager/python_api_testing/unit_testing/test_bmm_tilize_untilize.py
