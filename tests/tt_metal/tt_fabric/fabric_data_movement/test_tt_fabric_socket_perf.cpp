// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/device_pool.hpp>
#include "hostdevcommon/fabric_common.h"
#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "fabric_fixture.hpp"
#include "t3k_mesh_descriptor_chip_mappings.hpp"
#include "utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include "test_common.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "umd/device/types/xy_pair.h"
#include <tt-metalium/utils.hpp>
#include "tests/tt_metal/tt_fabric/common/utils.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

TEST_F(Fabric2DFixture, UnicastRaw_Skeleton) { RunTestUnicastRaw(this); }

TEST_F(Fabric2DFixture, UnicastConn_Skeleton) { RunTestUnicastConnAPI(this, /*num_connections=*/1); }

TEST_F(Fabric2DFixture, UnicastConn_Timed_Skeleton) {
    auto t0 = std::chrono::high_resolution_clock::now();
    RunTestUnicastConnAPI(this, /*num_connections=*/1);
    auto t1 = std::chrono::high_resolution_clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[UnicastConn_Timed_Skeleton] wall_time_s=" << sec << "\n";
}

TEST_F(Fabric2DFixture, UnicastConn_Hardcoded) {
    FabricNodeId src_fabric_node_id{MeshId{0}, /*chip_id=*/0};
    FabricNodeId dst_fabric_node_id{MeshId{0}, /*chip_id=*/1};

    // Map to physical chip ids
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    chip_id_t src_phys = cp.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    chip_id_t dst_phys = cp.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);

    // Hard-code path properties
    uint32_t num_hops = 1;
    bool use_dram_dst = false;  // true to land in DRAM on the dst side

    run_unicast_test_bw_chips(this, src_phys, dst_phys, num_hops, use_dram_dst);
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
