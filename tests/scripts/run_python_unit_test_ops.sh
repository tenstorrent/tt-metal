#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
    env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/loss_ops/) -vvv
    env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/fallback_ops/) -vvv
    env pytest $(find $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/backward_ops/) -vvv
fi
