#include <gtest/gtest.h>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric.hpp>

#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"  // Fabric2DFixture, RunTestUnicastConnAPI

using tt::tt_fabric::fabric_router_tests::Fabric2DFixture;

TEST_F(Fabric2DFixture, TestPerf) {
    // Call the existing helper unambiguously by namespace-qualifying it
    tt::tt_fabric::fabric_router_tests::RunTestUnicastConnAPI(
        this,
        /*num_hops=*/1
        // optional: , RoutingDirection::EAST, /*use_dram_dst=*/false
    );
}
