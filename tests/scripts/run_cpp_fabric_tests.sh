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


cd $TT_METAL_HOME

#############################################
# FABRIC UNIT TESTS                         #
#############################################
echo "Running fabric unit tests now...";

# DISABLED: Slow dispatch fabric tests require a dedicated runner pool that
# supports TT_METAL_SLOW_DISPATCH_MODE=1 without impacting fast-dispatch CI.
# Tracked in: https://github.com/tenstorrent/tt-metal/issues/24335
# Re-enable once a suitable runner pool is identified and allocated.
#TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"

# Host side tests that require a card: Topology Mapping in Control Plane
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MeshGraphValidation*"

./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="FabricTrafficGeneratorKernelIntegrationTest.*"

#############################################
# FABRIC SANITY TESTS                       #
#############################################
echo "Running fabric sanity tests now...";

./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_sanity_common.yaml
