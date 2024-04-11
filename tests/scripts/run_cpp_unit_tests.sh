#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
    ./build/test/tt_metal/unit_tests
    env python tests/scripts/run_tt_metal.py --dispatch-mode slow
    env python tests/scripts/run_tt_eager.py --dispatch-mode slow
else
    ./build/test/tt_metal/unit_tests_fast_dispatch
    if [[ "$ARCH_NAME" == "wormhole_b0" ]]; then
        WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter="-*WatcherFixture.*:*DPrintFixture.*:*DPrintErrorChecking.*:*DPrintFixtureDisableDevices.*"
    fi
    env python tests/scripts/run_tt_eager.py --dispatch-mode fast
    env python tests/scripts/run_tt_metal.py --dispatch-mode fast
fi
