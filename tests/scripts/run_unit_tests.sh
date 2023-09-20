#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
fi

export PYTHONPATH=$TT_METAL_HOME

if [[-z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
  if [ "$ARCH_NAME" == "grayskull" ]; then
    env python tests/scripts/run_tt_metal.py --dispatch-mode fast
    env python tests/scripts/run_tt_eager.py --dispatch-mode fast
  fi

  ./build/test/tt_metal/unit_tests_fast_dispatch
else
  env python tests/scripts/run_tt_metal.py --dispatch-mode slow
  env python tests/scripts/run_tt_eager.py --dispatch-mode slow
  ./build/test/tt_metal/unit_tests
fi
