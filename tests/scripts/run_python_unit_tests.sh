#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
    ./tests/scripts/run_python_unit_test_backward_ops.sh
    ./tests/scripts/run_python_unit_test_fallback_ops.sh
    ./tests/scripts/run_python_unit_test_loss_ops.sh
    # This doesn't exist?
    # ./tests/scripts/run_python_unit_test_misc_ops.sh
    ./tests/scripts/run_python_unit_test_moreh.sh

    if [ "$ARCH_NAME" != "wormhole_b0" ]; then
        # TODO(arakhmati): Run ttnn tests only on graskull until the issue with ttnn.reshape on wormhole is resolved
        env pytest $TT_METAL_HOME/tests/ttnn/unit_tests
    fi
else
    # Need to remove move for time being since failing
    env pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/ -vvv
fi
