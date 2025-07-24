#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
fi

export TT_METAL_CLEAR_L1=1

cd $TT_METAL_HOME

#############################################
# FABRIC UNIT TESTS                         #
#############################################
echo "Running fabric unit tests now...";

# TODO (issue: #24335) disabled slow dispatch tests for now, need to re-evaluate if need to add in a different pool.
#TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"

# these tests cover mux fixture as well
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"

#############################################
# FABRIC SANITY TESTS                       #
#############################################
echo "Running fabric sanity tests now...";

./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml
