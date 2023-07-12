#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [ "$TT_METAL_ENV" != "dev" ]; then
  echo "Must set TT_METAL_ENV as dev" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

source build/python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME

pytest $TT_METAL_HOME/tests/python_api_testing/sweep_tests/pytests/ -vvv -k 'test_sweep'
