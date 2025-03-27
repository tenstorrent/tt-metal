#!/bin/bash

set -eo pipefail

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
    env python3 tests/scripts/run_tt_metal.py --dispatch-mode slow
    env python3 tests/scripts/run_tt_eager.py --dispatch-mode slow
else
    # Enable this on BH after #14613
    if [[ "$ARCH_NAME" == "wormhole_b0" ]]; then
        TT_METAL_GTEST_ETH_DISPATCH=1 ./build/test/tt_metal/unit_tests_dispatch
    fi
    env python3 tests/scripts/run_tt_eager.py --dispatch-mode fast
    env python3 tests/scripts/run_tt_metal.py --dispatch-mode fast
fi
