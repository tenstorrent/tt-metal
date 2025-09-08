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
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "umd/device/types/xy_pair.h"
#include <tt-metalium/utils.hpp>

namespace tt::tt_fabric {
namespace fabric_router_tests {

// hack to let topology.cpp to know the binary is a unit test
// https://github.com/tenstorrent/tt-metal/issues/20000
// TODO: delete this once tt_fabric_api.h fully support low latency feature
extern "C" bool isFabricUnitTest();
bool isFabricUnitTest() { return true; }

using tt::tt_metal::ShardedBufferConfig;
using tt::tt_metal::ShardOrientation;
using tt::tt_metal::ShardSpecBuffer;

std::shared_ptr<tt_metal::distributed::MeshBuffer> PrepareBuffer(
    std::shared_ptr<tt_metal::distributed::MeshDevice> device,
    uint32_t size,
    CoreRangeSet& logical_crs,
    std::vector<uint32_t>& fill_data) {
    auto shard_parameters = ShardSpecBuffer(logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    tt_metal::distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = size,
        .buffer_type = tt_metal::BufferType::L1,
        .sharding_args = tt_metal::BufferShardingArgs(shard_parameters, tt_metal::TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false};

    tt_metal::distributed::ReplicatedBufferConfig global_buffer_config{
        .size = size,
    };
    auto buffer = tt_metal::distributed::MeshBuffer::create(global_buffer_config, device_local_config, device.get());
    tt_metal::distributed::WriteShard(
        device->mesh_command_queue(), buffer, fill_data, tt::tt_metal::distributed::MeshCoordinate({0, 0}), true);
    return buffer;
}

void RunGetNextHopRouterDirectionTest(BaseFabricFixture* fixture, bool is_multi_mesh = false) {
    CoreCoord logical_core = {0, 0};
    const auto& devices = fixture->get_devices();
    const size_t NUM_DEVICES = devices.size();
    bool invalid_test_scenario = !is_multi_mesh && NUM_DEVICES < 2;
    if (invalid_test_scenario) {
        GTEST_SKIP() << "Test requires at least 2 devices, found " << NUM_DEVICES;
    }

    std::vector<tt::tt_metal::Program> programs(NUM_DEVICES);
    std::vector<std::shared_ptr<tt_metal::distributed::MeshBuffer>> result_buffers(NUM_DEVICES);
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    for (size_t src_idx = 0; src_idx < NUM_DEVICES; src_idx++) {
        const auto& src_device = devices[src_idx];
        auto src_fabric_node_id =
            control_plane.get_fabric_node_id_from_physical_chip_id(src_device->get_devices()[0]->id());
        uint32_t src_fabric_chip_id = src_fabric_node_id.chip_id;

        uint32_t result_size = NUM_DEVICES * sizeof(uint32_t);
        std::vector<uint32_t> result_buffer_data(NUM_DEVICES, 0);
        CoreRangeSet core_range = {logical_core};
        result_buffers[src_idx] = PrepareBuffer(src_device, result_size, core_range, result_buffer_data);
        programs[src_idx] = tt::tt_metal::CreateProgram();

        uint32_t result_addr = result_buffers[src_idx]->address();
        std::vector<uint32_t> runtime_args = {
            *src_fabric_node_id.mesh_id,         // src_mesh_id
            src_fabric_chip_id,                  // src_chip_id
            result_addr,                         // result_addr
            static_cast<uint32_t>(NUM_DEVICES),  // num_devices
        };

        // Add mesh_id and chip_id pairs for all destinations
        for (size_t dst_idx = 0; dst_idx < NUM_DEVICES; dst_idx++) {
            auto dst_fabric_node_id =
                control_plane.get_fabric_node_id_from_physical_chip_id(devices[dst_idx]->get_devices()[0]->id());
            runtime_args.push_back(*dst_fabric_node_id.mesh_id);  // dst_mesh_id
            runtime_args.push_back(dst_fabric_node_id.chip_id);   // dst_chip_id
        }

        auto kernel = tt_metal::CreateKernel(
            programs[src_idx],
            "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_get_next_hop_router_direction.cpp",
            {logical_core},
            tt_metal::DataMovementConfig{});

        tt_metal::SetRuntimeArgs(programs[src_idx], kernel, logical_core, runtime_args);
    }

    for (size_t src_idx = 0; src_idx < NUM_DEVICES; src_idx++) {
        fixture->RunProgramNonblocking(devices[src_idx], programs[src_idx]);
    }
    for (size_t src_idx = 0; src_idx < NUM_DEVICES; src_idx++) {
        fixture->WaitForSingleProgramDone(devices[src_idx], programs[src_idx]);
    }

    for (size_t src_idx = 0; src_idx < NUM_DEVICES; src_idx++) {
        const auto& src_device = devices[src_idx];
        auto src_fabric_node_id =
            control_plane.get_fabric_node_id_from_physical_chip_id(src_device->get_devices()[0]->id());

        std::vector<uint32_t> result_data;
        tt::tt_metal::distributed::ReadShard(
            src_device->mesh_command_queue(),
            result_data,
            result_buffers[src_idx],
            tt::tt_metal::distributed::MeshCoordinate({0, 0}));
        for (size_t dst_idx = 0; dst_idx < NUM_DEVICES; dst_idx++) {
            auto dst_fabric_node_id =
                control_plane.get_fabric_node_id_from_physical_chip_id(devices[dst_idx]->get_devices()[0]->id());
            uint32_t actual_direction = result_data[dst_idx];
            if (src_fabric_node_id == dst_fabric_node_id) {
                // Self-routing should return INVALID_DIRECTION
                EXPECT_EQ(actual_direction, (uint32_t)eth_chan_magic_values::INVALID_DIRECTION);
            } else {
                auto expected_direction = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id)
                                              .value_or(RoutingDirection::NONE);

                if (expected_direction != RoutingDirection::NONE) {
                    // Route exists - should return valid direction
                    auto expected_eth_direction = control_plane.routing_direction_to_eth_direction(expected_direction);
                    EXPECT_EQ(actual_direction, expected_eth_direction);
                } else {
                    // No route exists - should return INVALID_DIRECTION
                    EXPECT_EQ(actual_direction, (uint32_t)eth_chan_magic_values::INVALID_DIRECTION);
                }
            }
        }
    }
}

std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> GenerateAllValidCombinations(
    BaseFabricFixture* fixture) {
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> combinations;
    const auto& devices = fixture->get_devices();

    if (devices.empty()) {
        return combinations;
    }

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto src_fabric_node_id =
        control_plane.get_fabric_node_id_from_physical_chip_id(devices[0]->get_devices()[0]->id());
    auto mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);

    uint32_t ns_dim = mesh_shape[0];
    uint32_t ew_dim = mesh_shape[1];

    for (uint32_t north = 0; north < ns_dim; north++) {
        for (uint32_t south = 0; south < ns_dim; south++) {
            if (north + south >= ns_dim) {
                continue;
            }

            for (uint32_t east = 0; east < ew_dim; east++) {
                for (uint32_t west = 0; west < ew_dim; west++) {
                    if (east + west >= ew_dim) {
                        continue;
                    }

                    if (north + south + east + west > 0) {
                        combinations.emplace_back(north, south, east, west);
                    }
                }
            }
        }
    }

    return combinations;
}

TEST_F(Fabric2DFixture, TestUnicastRaw) {
    for (uint32_t i = 0; i < 10; i++) {
        RunTestUnicastRaw(this);
    }
}

TEST_F(Fabric2DFixture, TestUnicastConnAPI) { RunTestUnicastConnAPI(this, 1); }

TEST_F(Fabric2DFixture, TestUnicastConnAPIDRAM) { RunTestUnicastConnAPI(this, 1, RoutingDirection::E, true); }

TEST_F(Fabric2DFixture, TestUnicastConnAPIRandom) {
    for (uint32_t i = 0; i < 10; i++) {
        RunTestUnicastConnAPIRandom(this);
    }
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_1W1E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 1, RoutingDirection::E, 1);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_1W2E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 1, RoutingDirection::E, 2);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_2W1E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 2, RoutingDirection::E, 1);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_2W2E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 2, RoutingDirection::E, 2);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_3W3E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 3, RoutingDirection::E, 3);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_4W3E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 4, RoutingDirection::E, 3);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_3W4E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 3, RoutingDirection::E, 4);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_1N2S) {
    RunTestMCastConnAPI(this, RoutingDirection::N, 1, RoutingDirection::S, 2);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_2N1S) {
    RunTestMCastConnAPI(this, RoutingDirection::N, 2, RoutingDirection::S, 1);
}

TEST_F(NightlyFabric2DFixture, Test2DMCast) {
    auto valid_combinations = GenerateAllValidCombinations(this);
    for (const auto& [north, south, east, west] : valid_combinations) {
        RunTest2DMCastConnAPI(this, north, south, east, west);
    }
}

// 2D topology Linear API tests (using 1D Linear API semantics)
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocUnicastWrite1D) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_WRITE, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocUnicastWrite1DMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocUnicastWrite1DWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        true);
}

TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocInlineWrite1D) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocInlineWrite1DMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocInlineWrite1DWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        true);
}

// Unicast Scatter Write
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocScatterWrite1D) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_SCATTER_WRITE, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocScatterWrite1DMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocScatterWrite1DWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        true);
}

// Unicast Atomic Inc
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocAtomicInc1D) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocAtomicInc1DMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocAtomicInc1DWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        true);
}

// Unicast Fused Atomic Inc
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocFusedAtomicInc1D) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_FUSED_UNICAST_ATOMIC_INC, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocFusedAtomicInc1DMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocFusedAtomicInc1DWithState) {
    FabricUnicastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        true);
}

TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocUnicastWrite1D) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_UNICAST_WRITE, cfg);
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocUnicastWrite1DMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocUnicastWrite1DWithState) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)},
        true);
}

// Multicast Inline Write
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocInlineWrite1D) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_UNICAST_INLINE_WRITE, cfg);
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocInlineWrite1DMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocInlineWrite1DWithState) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)},
        true);
}

// Multicast Scatter Write
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocScatterWrite1D) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_UNICAST_SCATTER_WRITE, cfg);
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocScatterWrite1DMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocScatterWrite1DWithState) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)},
        true);
}

// Multicast Atomic Inc
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocAtomicInc1D) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_UNICAST_ATOMIC_INC, cfg);
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocAtomicInc1DMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocAtomicInc1DWithState) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)},
        true);
}

// Multicast Fused Atomic Inc
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocFusedAtomicInc1D) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_FUSED_UNICAST_ATOMIC_INC, cfg);
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocFusedAtomicInc1DMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocFusedAtomicInc1DWithState) {
    FabricMulticastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)},
        true);
}

// 2D DYNAMIC topology Linear API tests (using 1D Linear API semantics)
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocUnicastWrite1D) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_WRITE, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocUnicastWrite1DMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocUnicastWrite1DWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        true);
}

TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocInlineWrite1D) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocInlineWrite1DMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocInlineWrite1DWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        true);
}

// Unicast Scatter Write
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocScatterWrite1D) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_SCATTER_WRITE, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocScatterWrite1DMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocScatterWrite1DWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        true);
}

// Unicast Atomic Inc
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocAtomicInc1D) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocAtomicInc1DMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocAtomicInc1DWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        true);
}

// Unicast Fused Atomic Inc
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocFusedAtomicInc1D) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_FUSED_UNICAST_ATOMIC_INC, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocFusedAtomicInc1DMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricUnicastNocFusedAtomicInc1DWithState) {
    FabricUnicastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        true);
}

TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocUnicastWrite1D) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_UNICAST_WRITE, cfg);
    }
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocUnicastWrite1DMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocUnicastWrite1DWithState) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)},
        true);
}

// Multicast Inline Write
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocInlineWrite1D) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_UNICAST_INLINE_WRITE, cfg);
    }
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocInlineWrite1DMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocInlineWrite1DWithState) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)},
        true);
}

// Multicast Scatter Write
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocScatterWrite1D) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_UNICAST_SCATTER_WRITE, cfg);
    }
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocScatterWrite1DMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocScatterWrite1DWithState) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)},
        true);
}

// Multicast Atomic Inc
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocAtomicInc1D) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_UNICAST_ATOMIC_INC, cfg);
    }
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocAtomicInc1DMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocAtomicInc1DWithState) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)},
        true);
}

// Multicast Fused Atomic Inc
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocFusedAtomicInc1D) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_FUSED_UNICAST_ATOMIC_INC, cfg);
    }
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocFusedAtomicInc1DMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DDynamicFixture, TestLinearFabricMulticastNocFusedAtomicInc1DWithState) {
    FabricMulticastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)},
        true);
}

// 1D Routing Validation Test
TEST_F(Fabric1DFixture, TestGetNextHopRouterDirection1D) { RunGetNextHopRouterDirectionTest(this, false); }

// 2D Dynamic Routing Unicast Tests
TEST_F(Fabric2DDynamicFixture, TestUnicastRaw) {
    for (uint32_t i = 0; i < 10; i++) {
        RunTestUnicastRaw(this);
    }
}

// 2D Dynamic Routing Unicast Tests
TEST_P(T3kCustomMeshGraphFabric2DDynamicFixture, TestUnicastRaw) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    CustomMeshGraphFabric2DDynamicFixture::SetUp(
        mesh_graph_desc_path, get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
    for (uint32_t i = 0; i < 10; i++) {
        RunTestUnicastRaw(this);
    }
}

TEST_F(Fabric2DDynamicFixture, TestGetNextHopRouterDirection1MeshAllToAll) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::tt_metal::ClusterType::TG) {
        GTEST_SKIP() << "Test not applicable for TG cluster type";
    }
    RunGetNextHopRouterDirectionTest(this, false);
}

// Multi-Mesh Test - Using parameterized test with connected mesh descriptor
TEST_P(T3kCustomMeshGraphFabric2DDynamicFixture, TestGetNextHopRouterDirectionMultiMesh) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    CustomMeshGraphFabric2DDynamicFixture::SetUp(
        mesh_graph_desc_path, get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
    RunGetNextHopRouterDirectionTest(this, true);
}

// Skipping other t3k configs because multi-mesh in single process isn't supported
INSTANTIATE_TEST_SUITE_P(
    T3kCustomMeshGraphFabric2DDynamicTests,
    T3kCustomMeshGraphFabric2DDynamicFixture,
    ::testing::Values(t3k_mesh_descriptor_chip_mappings[0]));

TEST_F(Fabric2DDynamicFixture, TestUnicastConnAPI) { RunTestUnicastConnAPI(this, 1); }

TEST_F(Fabric2DDynamicFixture, TestUnicastConnAPIDRAM) { RunTestUnicastConnAPI(this, 1, RoutingDirection::E, true); }

// 2D Dynamic Routing Unidirectional mcast tests (no turns)
TEST_F(Fabric2DDynamicFixture, TestLineMcastE2Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastE3Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastE7Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastW2Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastW3Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastW7Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 7};
    RunTestLineMcast(this, {routing_info});
}

// 2D Dynamic Routing Unidirectional mcast tests (with turns)
TEST_F(Fabric2DDynamicFixture, TestLineMcastN1HopE3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN1HopE7Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, n_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastN2HopE3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN2HopE7Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, n_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS1HopE3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastS1HopE7Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, s_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS2HopE3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastS2HopE7Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, s_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastN1HopW3Hops) {
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {w_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN1HopW7Hops) {
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 7};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {w_routing_info, n_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS1HopW3Hops) {
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastS1HopW7Hops) {
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 7};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {w_routing_info, s_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS1HopE1HopW1Hop) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastN1HopE1HopW1Hop) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS1HopE2HopsW1Hop) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastN1HopE2HopsW1Hop) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS1HopE1HopW2Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastN1HopE1HopW2Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS2HopsE4HopsW3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 4};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastS2HopsE3HopsW4Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 4};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastS3HopsE4HopsW3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 4};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 3};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastS3HopsE3HopsW4Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 4};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 3};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN2HopsE4HopsW3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 4};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN2HopsE3HopsW4Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 4};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN3HopsE4HopsW3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 4};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 3};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN3HopsE3HopsW4Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 4};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 3};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
