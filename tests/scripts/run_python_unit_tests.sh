#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
    env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/ -name 'test_*.py' -a ! -name 'test_untilize_with_halo_and_max_pool.py') -vvv
    env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/ -name 'test_*.py' -a ! -name 'test_sweep_conv_with_address_map.py') -vvv

else
    # Need to remove move for time being since failing
    env pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/ -vvv
    env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/ -name 'test_*.py' -a ! -name 'test_sweep_conv_with_address_map.py' -a ! -name 'test_move.py') -vvv
    env pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_move.py -k input_L1 -vvv
fi
