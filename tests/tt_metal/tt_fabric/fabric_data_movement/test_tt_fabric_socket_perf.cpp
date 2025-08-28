#include <gtest/gtest.h>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric.hpp>

#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"  // Fabric2DFixture, BaseFabricFixture
#include "tests/tt_metal/tt_fabric/common/utils.hpp"           // find_device_with_neighbor_in_direction

// Bring types/helpers into scope
using tt::tt_fabric::fabric_router_tests::Fabric2DFixture;
using BaseFabricFixture = tt::tt_fabric::fabric_router_tests::BaseFabricFixture;
using tt::tt_fabric::FabricNodeId;
using tt::tt_fabric::MeshId;
using tt::tt_fabric::RoutingDirection;
using tt::tt_fabric::fabric_router_tests::find_device_with_neighbor_in_direction;
using chip_id_t = tt::umd::chip_id_t;

// ---------- FORWARD DECLARATION IN THE CORRECT NAMESPACE ----------
namespace tt::tt_fabric::fabric_router_tests {
void run_unicast_test_bw_chips(
    BaseFabricFixture* fixture,
    tt::umd::chip_id_t src_physical_device_id,
    tt::umd::chip_id_t dst_physical_device_id,
    uint32_t num_hops,
    bool use_dram_dst);
}  // namespace tt::tt_fabric::fabric_router_tests

// Try a single direction; return true if a path was found and test executed.
static bool RunTestUnicastConnAPI_TryDirection(
    BaseFabricFixture* fixture, uint32_t num_hops, RoutingDirection direction, bool use_dram_dst) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    FabricNodeId dst_fabric_node_id(MeshId{0}, 0);
    chip_id_t not_used_1{};
    chip_id_t not_used_2{};

    if (!find_device_with_neighbor_in_direction(
            fixture, src_fabric_node_id, dst_fabric_node_id, not_used_1, not_used_2, direction)) {
        return false;
    }

    chip_id_t src_phys = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    chip_id_t dst_phys = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);

    tt::tt_fabric::fabric_router_tests::run_unicast_test_bw_chips(fixture, src_phys, dst_phys, num_hops, use_dram_dst);
    return true;
}

// Convenience wrapper: try EAST, WEST, NORTH, SOUTH (in that order).
static void RunTestUnicastConnAPI_Local(BaseFabricFixture* fixture, uint32_t num_hops, bool use_dram_dst = false) {
    const RoutingDirection dirs[] = {
        static_cast<RoutingDirection>(tt::tt_fabric::EAST),
        static_cast<RoutingDirection>(tt::tt_fabric::WEST),
        static_cast<RoutingDirection>(tt::tt_fabric::NORTH),
        static_cast<RoutingDirection>(tt::tt_fabric::SOUTH),
    };

    for (auto d : dirs) {
        if (RunTestUnicastConnAPI_TryDirection(fixture, num_hops, d, use_dram_dst)) {
            return;  // success on first available neighbor
        }
    }
    GTEST_SKIP() << "No path found between sender and receiver in any direction (E/W/N/S)";
}

// ---------- TEST ----------
TEST_F(Fabric2DFixture, TestPerf) {
    // One hop to a neighbor in any available direction, DRAM dest off (L1)
    RunTestUnicastConnAPI_Local(this, /*num_hops=*/1);
}
