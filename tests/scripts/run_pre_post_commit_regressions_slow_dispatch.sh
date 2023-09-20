#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
  echo "Only Slow Dispatch mode allowed - Must have TT_METAL_SLOW_DISPATCH_MODE set" 1>&2
  exit 1
fi

cd $TT_METAL_HOME
export PYTHONPATH=$TT_METAL_HOME

if [ "$ARCH_NAME" == "grayskull" ]; then
  ./tests/scripts/run_python_api_unit_tests.sh
  ./tests/scripts/run_python_api_sweep_tests.sh
  ./tests/scripts/run_python_api_model_tests.sh
else
  ./tests/scripts/run_python_api_unit_tests_wormhole_b0.sh
fi
./tests/scripts/run_unit_tests.sh

echo "Checking docs build..."

cd $TT_METAL_HOME/docs
python -m pip install -r requirements-docs.txt
make clean
make html
