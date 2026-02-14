#!/bin/bash

set -eo pipefail

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

# Enable this on BH after #14613
if [[ "$ARCH_NAME" == "wormhole_b0" ]]; then
    export TT_METAL_ENABLE_ERISC_IRAM=1
fi

if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
    env python3 tests/scripts/run_tt_eager.py --dispatch-mode slow
else
    # Enable this on BH after #14613
    if [[ "$ARCH_NAME" == "wormhole_b0" ]]; then
        TT_METAL_GTEST_ETH_DISPATCH=1 ./build/test/tt_metal/unit_tests_dispatch
    fi
    env python3 tests/scripts/run_tt_eager.py --dispatch-mode fast
    # Programming example
    # TODO why is this not ran in pr-gate?
    ./build/programming_examples/contributed/vecadd
fi

./build/test/tt_metal/unit_tests_legacy --gtest_shuffle --gtest_filter=-*NIGHTLY_*
