#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

env pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/sweep_tests/pytests/ -vvv
