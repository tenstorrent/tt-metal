#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

export PYTHONPATH=$TT_METAL_HOME

pytest $TT_METAL_HOME/tests/python_api_testing/unit_testing/ -vvv
