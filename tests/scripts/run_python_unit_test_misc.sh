#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
    env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/misc/ -name 'test_*.py' -a ! -name 'test_untilize_with_halo_and_max_pool.py' ! -name 'test_moreh*.py') -vvv
else
    # Need to remove move for time being since failing
    env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/misc/ ! -name 'test_moreh*.py') -vvv
fi
