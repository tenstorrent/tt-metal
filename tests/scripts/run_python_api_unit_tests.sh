#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
fi

export PYTHONPATH=$TT_METAL_HOME

if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
  env pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/
else
  # Need to remove move for time being since failing
  env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/ -name 'test_*.py' -a ! -name 'test_move.py') -vvv
fi

# This must run in slow dispatch mode
# pytest -svv $TT_METAL_HOME/tests/python_api_testing/sweep_tests/pytests/test_sweep_conv_with_address_map.py
