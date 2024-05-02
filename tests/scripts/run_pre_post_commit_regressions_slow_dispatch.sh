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

# New FD2 cpp tests.
./tests/scripts/run_cpp_fd2_tests.sh

./tests/scripts/run_cpp_unit_tests.sh
