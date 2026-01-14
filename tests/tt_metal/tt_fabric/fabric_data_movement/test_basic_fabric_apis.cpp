// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
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

#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
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
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <umd/device/types/xy_pair.hpp>
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric::fabric_router_tests {

// hack to let topology.cpp to know the binary is a unit test
// https://github.com/tenstorrent/tt-metal/issues/20000
// TODO: delete this once tt_fabric_api.h fully support low latency feature
extern "C" bool isFabricUnitTest();
bool isFabricUnitTest() { return true; }

using tt::tt_metal::HalProgrammableCoreType;
using tt::tt_metal::KernelHandle;
using tt::tt_metal::ShardOrientation;
using tt::tt_metal::ShardSpecBuffer;

std::shared_ptr<tt_metal::distributed::MeshBuffer> PrepareBuffer(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
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
void RunSetUnicastRouteTest(
    BaseFabricFixture* fixture,
    bool is_multi_mesh = false,
    HalProgrammableCoreType core_type = HalProgrammableCoreType::TENSIX) {
    const auto& devices = fixture->get_devices();
    const size_t NUM_DEVICES = devices.size();
    bool invalid_test_scenario = !is_multi_mesh && NUM_DEVICES < 2;
    if (invalid_test_scenario) {
        GTEST_SKIP() << "Test requires at least 2 devices, found " << NUM_DEVICES;
    }

    // Select appropriate logical core based on core type - this will be device-specific
    std::vector<CoreCoord> logical_cores(NUM_DEVICES);
    for (size_t dev_idx = 0; dev_idx < NUM_DEVICES; dev_idx++) {
        if (core_type == HalProgrammableCoreType::IDLE_ETH) {
            // Use first available IDLE_ETH core for each device
            auto idle_eth_cores = devices[dev_idx]->get_devices()[0]->get_inactive_ethernet_cores();
            if (idle_eth_cores.empty()) {
                GTEST_SKIP() << "No IDLE_ETH cores available on device " << dev_idx;
            }
            if (!fixture->slow_dispatch_) {
                GTEST_SKIP() << "IDLE_ETH core test requires TT_METAL_SLOW_DISPATCH_MODE enabled in fixture";
            }
            logical_cores[dev_idx] = *idle_eth_cores.begin();
        } else {
            logical_cores[dev_idx] = {0, 0};
        }
    }

    std::vector<tt::tt_metal::Program> programs(NUM_DEVICES);
    std::vector<uint32_t> result_addrs(NUM_DEVICES);  // Store fixed addresses for each device
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Get mesh shape to determine if it's 2D fabric
    auto src_fabric_node_id =
        control_plane.get_fabric_node_id_from_physical_chip_id(devices[0]->get_devices()[0]->id());
    auto mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    bool is_2d_fabric = topology == Topology::Mesh;
    uint32_t ew_dim = is_2d_fabric ? mesh_shape[1] : 0;

    // Query actual route buffer size from fabric context (topology-aware)
    uint32_t MAX_ROUTE_BUFFER_SIZE =
        is_2d_fabric ? fabric_context.get_2d_pkt_hdr_route_buffer_size() : SINGLE_ROUTE_SIZE_1D;
    uint32_t RESULT_SIZE_PER_DEVICE = (MAX_ROUTE_BUFFER_SIZE * 2);  // 2 route buffers
    // 0x100000 (1MB) is safe on Tensix L1
    uint32_t FABRIC_TEST_BUFFER_BASE_ADDR = 0x100000;
    if (core_type == HalProgrammableCoreType::IDLE_ETH) {
        FABRIC_TEST_BUFFER_BASE_ADDR = tt_metal::MetalContext::instance().hal().get_dev_addr(
            HalProgrammableCoreType::IDLE_ETH, tt_metal::HalL1MemAddrType::UNRESERVED);
    }

    for (size_t src_idx = 0; src_idx < NUM_DEVICES; src_idx++) {
        const auto& src_device = devices[src_idx];
        auto src_fabric_node_id =
            control_plane.get_fabric_node_id_from_physical_chip_id(src_device->get_devices()[0]->id());
        uint32_t src_fabric_chip_id = src_fabric_node_id.chip_id;

        uint32_t result_size = NUM_DEVICES * RESULT_SIZE_PER_DEVICE * sizeof(uint32_t);
        uint32_t result_addr = FABRIC_TEST_BUFFER_BASE_ADDR + (src_idx * result_size);
        result_addrs[src_idx] = result_addr;  // Store for later use

        // Skip MeshBuffer creation - directly use fixed address for experimental measurement
        // This bypasses host-side device memory management for raw address access
        programs[src_idx] = tt::tt_metal::CreateProgram();

        std::vector<uint32_t> runtime_args = {
            *src_fabric_node_id.mesh_id,         // src_mesh_id
            src_fabric_chip_id,                  // src_chip_id
            result_addr,                         // result_addr
            static_cast<uint32_t>(NUM_DEVICES),  // num_devices
            ew_dim,                              // ew_dim
        };

        // Add mesh_id and chip_id pairs for all destinations
        for (size_t dst_idx = 0; dst_idx < NUM_DEVICES; dst_idx++) {
            auto dst_fabric_node_id =
                control_plane.get_fabric_node_id_from_physical_chip_id(devices[dst_idx]->get_devices()[0]->id());
            runtime_args.push_back(*dst_fabric_node_id.mesh_id);  // dst_mesh_id
            runtime_args.push_back(dst_fabric_node_id.chip_id);   // dst_chip_id
        }

        std::map<std::string, std::string> defines = {};
        if (is_2d_fabric) {
            defines["FABRIC_2D"] = "";
        }

        KernelHandle kernel;
        if (core_type == HalProgrammableCoreType::IDLE_ETH) {
            kernel = tt_metal::CreateKernel(
                programs[src_idx],
                "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_fabric_set_unicast_route.cpp",
                {logical_cores[src_idx]},
                tt_metal::EthernetConfig{
                    .eth_mode = tt_metal::Eth::IDLE,
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .defines = defines});
        } else {
            kernel = tt_metal::CreateKernel(
                programs[src_idx],
                "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_fabric_set_unicast_route.cpp",
                {logical_cores[src_idx]},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .defines = defines});
        }

        tt_metal::SetRuntimeArgs(programs[src_idx], kernel, logical_cores[src_idx], runtime_args);
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

        uint32_t result_size = NUM_DEVICES * RESULT_SIZE_PER_DEVICE * sizeof(uint32_t);
        std::vector<uint32_t> result_data;

        // Use tt_metal detail API to read from device L1 memory directly
        // Note: This is experimental and bypasses safety checks
        CoreType read_core_type = (core_type == HalProgrammableCoreType::IDLE_ETH) ? CoreType::ETH : CoreType::WORKER;
        tt::tt_metal::detail::ReadFromDeviceL1(
            src_device->get_devices()[0],
            logical_cores[src_idx],
            result_addrs[src_idx],
            result_size,
            result_data,
            read_core_type);

        for (size_t dst_idx = 0; dst_idx < NUM_DEVICES; dst_idx++) {
            auto dst_fabric_node_id =
                control_plane.get_fabric_node_id_from_physical_chip_id(devices[dst_idx]->get_devices()[0]->id());
            if (!is_2d_fabric && std::abs(
                                     static_cast<long>(src_fabric_node_id.chip_id) -
                                     static_cast<long>(dst_fabric_node_id.chip_id)) >= MAX_CHIPS_LOWLAT_1D) {
                // Skip 1D route buffer comparison if src and dst are more than 16 chips apart
                continue;
            }

            uint32_t result_offset = dst_idx * RESULT_SIZE_PER_DEVICE;
            // Compare route buffers
            bool route_buffers_match = true;
            for (uint32_t i = 0; i < MAX_ROUTE_BUFFER_SIZE; i++) {
                uint32_t actual_byte = result_data[result_offset + i];
                uint32_t expected_byte = result_data[result_offset + MAX_ROUTE_BUFFER_SIZE + i];
                if (actual_byte != expected_byte) {
                    route_buffers_match = false;
                    break;
                }
            }

            EXPECT_TRUE(route_buffers_match)
                << "Route buffer mismatch for [" << *src_fabric_node_id.mesh_id << "/" << src_fabric_node_id.chip_id
                << "] -> [" << *dst_fabric_node_id.mesh_id << "/" << dst_fabric_node_id.chip_id << "]";
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

TEST_F(Fabric2DFixture, TestUnicastRaw_3N) {
    RunTestUnicastRaw2D(this, 3, RoutingDirection::N, 0, RoutingDirection::E);
}

TEST_F(Fabric2DFixture, TestUnicastRaw_3E) {
    RunTestUnicastRaw2D(this, 0, RoutingDirection::N, 3, RoutingDirection::E);
}

TEST_F(Fabric2DFixture, TestUnicastRaw_3N3E) {
    RunTestUnicastRaw2D(this, 3, RoutingDirection::N, 3, RoutingDirection::E);
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

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_1N1E1W) { RunTest2DMCastConnAPI(this, 1, 0, 1, 1); }

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_7N3E) { RunTest2DMCastConnAPI(this, 7, 0, 3, 0); }

TEST_F(NightlyFabric2DFixture, Test2DMCast) {
    auto valid_combinations = GenerateAllValidCombinations(this);
    for (const auto& [north, south, east, west] : valid_combinations) {
        RunTest2DMCastConnAPI(this, north, south, east, west);
    }
}

// 2D topology Mesh API tests
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocUnicastWrite) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_WRITE, {std::make_tuple(dir, 1)}, FabricApiType::Mesh);
    }
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocUnicastWriteMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Mesh);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocUnicastWriteWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Mesh,
        true);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricMulticastNocUnicastWriteWithState) {
    Fabric2DMulticastCommon(
        this,
        NOC_UNICAST_WRITE,
        {
            {std::make_tuple(RoutingDirection::N, 0, 1), std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::W, 0, 1)},
        },
        true);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricMulticastNocUnicastWrite) {
    Fabric2DMulticastCommon(
        this,
        NOC_UNICAST_WRITE,
        {
            {std::make_tuple(RoutingDirection::N, 0, 1), std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::W, 0, 1)},
        },
        false);
}

TEST_F(NightlyFabric2DFixture, TestMeshFabricMulticastNocInlineUnicastWrite) {
    Fabric2DMulticastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {
            {std::make_tuple(RoutingDirection::N, 0, 1), std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::W, 0, 1)},
        },
        false);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricMulticastNocInlineUnicastWriteWithState) {
    Fabric2DMulticastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {
            {std::make_tuple(RoutingDirection::N, 0, 1), std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::W, 0, 1)},
        },
        true);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricMulticastNocScatterUnicastWrite) {
    Fabric2DMulticastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {
            {std::make_tuple(RoutingDirection::N, 0, 1), std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::W, 0, 1)},
        },
        false);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricMulticastNocScatterUnicastWriteWithState) {
    Fabric2DMulticastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {
            {std::make_tuple(RoutingDirection::N, 0, 1), std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::W, 0, 1)},
        },
        true);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricMulticastNocAtomicInc) {
    Fabric2DMulticastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {
            {std::make_tuple(RoutingDirection::N, 0, 1), std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::W, 0, 1)},
        },
        false);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricMulticastNocAtomicIncWithState) {
    Fabric2DMulticastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {
            {std::make_tuple(RoutingDirection::N, 0, 1), std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::W, 0, 1)},
        },
        true);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricMulticastNocFusedAtomicInc) {
    Fabric2DMulticastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {
            {std::make_tuple(RoutingDirection::N, 0, 1), std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::W, 0, 1)},
        },
        false);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricMulticastNocFusedAtomicIncWithState) {
    Fabric2DMulticastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {
            {std::make_tuple(RoutingDirection::N, 0, 1), std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::W, 0, 1)},
        },
        true);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocInlineWrite) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, {std::make_tuple(dir, 1)}, FabricApiType::Mesh);
    }
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocInlineWriteMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Mesh);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocInlineWriteWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Mesh,
        true);
}

// Nightly Mux Mode Tests - test mux extension for 1D
TEST_F(NightlyFabric1DTensixFixture, TestLinearFabricMulticastNocMux) {
    std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>> configs = {
        std::make_tuple(RoutingDirection::E, 1, 2),
        std::make_tuple(RoutingDirection::E, 1, 3),
        std::make_tuple(RoutingDirection::W, 1, 2),
        std::make_tuple(RoutingDirection::W, 1, 3),
        std::make_tuple(RoutingDirection::N, 1, 1),
        std::make_tuple(RoutingDirection::S, 1, 1)};
    for (const auto& config : configs) {
        auto [dir, start, range] = config;
        log_info(tt::LogTest, "Testing Multicast Mux 1D: Dir={}, Start={}, Range={}", dir, start, range);
        log_info(tt::LogTest, "  Type: NOC_UNICAST_WRITE");
        FabricMulticastCommon(this, NOC_UNICAST_WRITE, {config});
        log_info(tt::LogTest, "  Type: NOC_UNICAST_INLINE_WRITE");
        FabricMulticastCommon(this, NOC_UNICAST_INLINE_WRITE, {config});
        log_info(tt::LogTest, "  Type: NOC_UNICAST_ATOMIC_INC");
        FabricMulticastCommon(this, NOC_UNICAST_ATOMIC_INC, {config});
    }
}
// Nightly Mux Mode Tests - test mux extension for 2D
TEST_F(NightlyFabric2DTensixFixture, TestMeshFabricMulticastNocMux) {
    std::vector<std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>>> all_multicast_configs = {
        // North + East + West combination
        {
            {std::make_tuple(RoutingDirection::N, 0, 1), std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::W, 0, 1)},
        },
        {
            {std::make_tuple(RoutingDirection::N, 0, 1), std::make_tuple(RoutingDirection::E, 0, 1)},
            {std::make_tuple(RoutingDirection::E, 0, 1)},
            {std::make_tuple(RoutingDirection::W, 0, 2)},
        },
        // South + East + West combination
        {
            {std::make_tuple(RoutingDirection::S, 0, 1), std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::E, 0, 2)},
            {std::make_tuple(RoutingDirection::W, 0, 1)},
        },
        {
            {std::make_tuple(RoutingDirection::S, 0, 1), std::make_tuple(RoutingDirection::E, 0, 1)},
            {std::make_tuple(RoutingDirection::E, 0, 1)},
            {std::make_tuple(RoutingDirection::W, 0, 2)},
        },
    };
    for (const auto& multicast_configs : all_multicast_configs) {
        log_info(tt::LogTest, "Testing Mesh Multicast Mux 2D - Config {}", multicast_configs);
        log_info(tt::LogTest, "  Type: NOC_UNICAST_WRITE");
        Fabric2DMulticastCommon(this, NOC_UNICAST_WRITE, multicast_configs, false);
        log_info(tt::LogTest, "  Type: NOC_UNICAST_INLINE_WRITE");
        Fabric2DMulticastCommon(this, NOC_UNICAST_INLINE_WRITE, multicast_configs, false);
        log_info(tt::LogTest, "  Type: NOC_UNICAST_ATOMIC_INC");
        Fabric2DMulticastCommon(this, NOC_UNICAST_ATOMIC_INC, multicast_configs, false);
    }
}

// UDM Mode Tests - test udm api changes for 2D
TEST_F(Fabric2DUDMModeFixture, TestUDMFabricUnicastWriteEast) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(RoutingDirection::E, 1));
}
TEST_F(Fabric2DUDMModeFixture, TestUDMFabricUnicastWriteWest) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(RoutingDirection::W, 1));
}
TEST_F(Fabric2DUDMModeFixture, TestUDMFabricUnicastInlineWriteEast) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(RoutingDirection::E, 1));
}
TEST_F(Fabric2DUDMModeFixture, TestUDMFabricUnicastInlineWriteWest) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(RoutingDirection::W, 1));
}
TEST_F(Fabric2DUDMModeFixture, TestUDMFabricUnicastAtomicIncEast) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(RoutingDirection::E, 1));
}
TEST_F(Fabric2DUDMModeFixture, TestUDMFabricUnicastAtomicIncWest) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(RoutingDirection::W, 1));
}
TEST_F(Fabric2DUDMModeFixture, TestUDMFabricReadEast) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(RoutingDirection::E, 1));
}
TEST_F(Fabric2DUDMModeFixture, TestUDMFabricReadWest) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(RoutingDirection::W, 1));
}

// Nightly UDM Mode Tests - test udm api changes for 2D
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteEast) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(RoutingDirection::E, 2));
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteWest) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(RoutingDirection::W, 2));
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteNorth) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(RoutingDirection::N, 1));
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteSouth) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(RoutingDirection::S, 1));
}

// Nightly UDM Mode Dual RISC Tests - test both BRISC and NCRISC accessing fabric simultaneously
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteDualRisc) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Dual RISC test does not support wormhole";
    }
    UDMFabricUnicastCommon(
        this, NOC_UNICAST_WRITE, std::make_tuple(RoutingDirection::E, 1), std::nullopt, std::nullopt, true);
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastInlineWriteDualRisc) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Dual RISC test does not support wormhole";
    }
    UDMFabricUnicastCommon(
        this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(RoutingDirection::E, 1), std::nullopt, std::nullopt, true);
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastAtomicIncDualRisc) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Dual RISC test does not support wormhole";
    }
    UDMFabricUnicastCommon(
        this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(RoutingDirection::E, 1), std::nullopt, std::nullopt, true);
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricReadDualRisc) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Dual RISC test does not support wormhole";
    }
    UDMFabricUnicastCommon(
        this, NOC_UNICAST_READ, std::make_tuple(RoutingDirection::E, 1), std::nullopt, std::nullopt, true);
}

// Nightly UDM Mode Tests - test udm inline write api changes for 2D
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastInlineWriteEast) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(RoutingDirection::E, 2));
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastInlineWriteWest) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(RoutingDirection::W, 2));
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastInlineWriteNorth) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(RoutingDirection::N, 1));
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastInlineWriteSouth) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(RoutingDirection::S, 1));
}

// Nightly UDM Mode Tests - test udm atomic inc api changes for 2D
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastAtomicIncEast) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(RoutingDirection::E, 2));
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastAtomicIncWest) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(RoutingDirection::W, 2));
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastAtomicIncNorth) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(RoutingDirection::N, 1));
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastAtomicIncSouth) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(RoutingDirection::S, 1));
}

// Nightly UDM Mode Read Tests - test udm read api for 2D
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricReadEast) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(RoutingDirection::E, 2));
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricReadWest) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(RoutingDirection::W, 2));
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricReadNorth) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(RoutingDirection::N, 1));
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricReadSouth) {
    UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(RoutingDirection::S, 1));
}

// UDM Mode Write Tests with explicit src/dest node IDs - test specific node pairs that has traffic turns
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteFromNode0) {
    for (uint32_t dst : {5u, 6u, 7u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(0u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(0u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(0u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteFromNode1) {
    for (uint32_t dst : {4u, 6u, 7u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(1u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(1u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(1u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteFromNode2) {
    for (uint32_t dst : {4u, 5u, 7u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(2u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(2u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(2u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteFromNode3) {
    for (uint32_t dst : {4u, 5u, 6u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(3u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(3u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(3u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteFromNode4) {
    for (uint32_t dst : {1u, 2u, 3u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(4u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(4u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(4u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteFromNode5) {
    for (uint32_t dst : {0u, 2u, 3u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(5u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(5u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(5u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteFromNode6) {
    for (uint32_t dst : {0u, 1u, 3u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(6u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(6u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(6u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteFromNode7) {
    for (uint32_t dst : {0u, 1u, 2u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(7u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, std::make_tuple(7u, dst));
        UDMFabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, std::make_tuple(7u, dst));
    }
}
// UDM Mode Read Tests with explicit src/dest node IDs - test specific node pairs that has traffic turns
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastReadFromNode0) {
    for (uint32_t dst : {5u, 6u, 7u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(0u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastReadFromNode1) {
    for (uint32_t dst : {4u, 6u, 7u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(1u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastReadFromNode2) {
    for (uint32_t dst : {4u, 5u, 7u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(2u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastReadFromNode3) {
    for (uint32_t dst : {4u, 5u, 6u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(3u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastReadFromNode4) {
    for (uint32_t dst : {1u, 2u, 3u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(4u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastReadFromNode5) {
    for (uint32_t dst : {0u, 2u, 3u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(5u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastReadFromNode6) {
    for (uint32_t dst : {0u, 1u, 3u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(6u, dst));
    }
}
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastReadFromNode7) {
    for (uint32_t dst : {0u, 1u, 2u}) {
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(7u, dst));
    }
}

// Helper to generate all worker coordinate pairs in the compute grid (sender coord == receiver coord)
std::vector<std::pair<CoreCoord, CoreCoord>> GetAllWorkerCoordPairs(CoreCoord grid_size) {
    std::vector<std::pair<CoreCoord, CoreCoord>> pairs;
    for (size_t x = 0; x < grid_size.x; x++) {
        for (size_t y = 0; y < grid_size.y; y++) {
            CoreCoord coord{x, y};
            pairs.push_back({coord, coord});
        }
    }
    return pairs;
}

// UDM Mode Worker Coordinate Tests - test fabric communication with all workers simultaneously
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteAllWorkerCoords) {
    auto grid_size = get_devices()[0]->get_devices()[0]->compute_with_storage_grid_size();
    auto all_worker_pairs = GetAllWorkerCoordPairs(grid_size);
    log_info(tt::LogTest, "Testing {} worker pairs for write operations", all_worker_pairs.size());
    for (uint32_t dst : {5u, 6u, 7u}) {
        log_info(tt::LogTest, "  Sender at fabric node 0 and receiver at fabric node {}", dst);
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(0u, dst), std::nullopt, all_worker_pairs);
    }
}

TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastReadAllWorkerCoords) {
    auto grid_size = get_devices()[0]->get_devices()[0]->compute_with_storage_grid_size();
    auto all_worker_pairs = GetAllWorkerCoordPairs(grid_size);
    log_info(tt::LogTest, "Testing {} worker pairs for read operations", all_worker_pairs.size());
    for (uint32_t dst : {5u, 6u, 7u}) {
        log_info(tt::LogTest, "  Sender at fabric node 0 and receiver at fabric node {}", dst);
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(0u, dst), std::nullopt, all_worker_pairs);
    }
}

TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteAllWorkerCoordsDualRisc) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Dual RISC test does not support wormhole";
    }
    auto grid_size = get_devices()[0]->get_devices()[0]->compute_with_storage_grid_size();
    auto all_worker_pairs = GetAllWorkerCoordPairs(grid_size);
    log_info(tt::LogTest, "Testing {} worker pairs for write operations", all_worker_pairs.size());
    for (uint32_t dst : {5u, 6u, 7u}) {
        log_info(tt::LogTest, "  Sender at fabric node 0 and receiver at fabric node {}", dst);
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(0u, dst), std::nullopt, all_worker_pairs);
    }
}

TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastReadAllWorkerCoordsDualRisc) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Dual RISC test does not support wormhole";
    }
    auto grid_size = get_devices()[0]->get_devices()[0]->compute_with_storage_grid_size();
    auto all_worker_pairs = GetAllWorkerCoordPairs(grid_size);
    log_info(tt::LogTest, "Testing {} worker pairs for read operations", all_worker_pairs.size());
    for (uint32_t dst : {5u, 6u, 7u}) {
        log_info(tt::LogTest, "  Sender at fabric node 0 and receiver at fabric node {}", dst);
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(0u, dst), std::nullopt, all_worker_pairs);
    }
}

// UDM Mode All-to-All Tests - all devices send to all other devices simultaneously
// Senders are on top half of compute grid, receivers are on bottom half
// Each receiver receives from N-1 senders at different L1 locations
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteAllToAll) {
    UDMFabricUnicastAllToAllCommon(this, NOC_UNICAST_WRITE);
}

TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastInlineWriteAllToAll) {
    UDMFabricUnicastAllToAllCommon(this, NOC_UNICAST_INLINE_WRITE);
}

TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastAtomicIncAllToAll) {
    UDMFabricUnicastAllToAllCommon(this, NOC_UNICAST_ATOMIC_INC);
}

TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastReadAllToAll) {
    UDMFabricUnicastAllToAllCommon(this, NOC_UNICAST_READ);
}

TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastWriteAllToAllDualRisc) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Dual RISC test does not support wormhole";
    }
    UDMFabricUnicastAllToAllCommon(this, NOC_UNICAST_WRITE, true);
}

TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastInlineWriteAllToAllDualRisc) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Dual RISC test does not support wormhole";
    }
    UDMFabricUnicastAllToAllCommon(this, NOC_UNICAST_INLINE_WRITE, true);
}

TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastAtomicIncAllToAllDualRisc) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Dual RISC test does not support wormhole";
    }
    UDMFabricUnicastAllToAllCommon(this, NOC_UNICAST_ATOMIC_INC, true);
}

TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricUnicastReadAllToAllDualRisc) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Dual RISC test does not support wormhole";
    }
    UDMFabricUnicastAllToAllCommon(this, NOC_UNICAST_READ, true);
}

// Mux-to-Mux Forwarding Tests - test the mux's ability to forward packets to the correct downstream mux
// These tests intentionally send packets with a non-optimal initial direction to verify mux forwarding works
// Test cases cover scenarios where the worker sends a packet to a mux in a different direction,
// and the mux must forward it to the correct downstream mux based on the packet's initial_direction field

// Node 0 has neighbors in E and S directions
// Forwarding to East destinations (1, 2, 3) via non-East initial direction (S)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode0ForwardEast) {
    for (auto initial_dir : {RoutingDirection::S}) {
        log_info(tt::LogTest, "Node 0: Testing forward to East via initial_dir={}", initial_dir);
        for (uint32_t dst : {1u, 2u, 3u}) {
            log_info(tt::LogTest, "  Node 0->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(0u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 0->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(0u, dst), initial_dir);
        }
    }
}

// Forwarding to South destinations (4, 5, 6, 7) via non-South initial direction (E)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode0ForwardSouth) {
    for (auto initial_dir : {RoutingDirection::E}) {
        log_info(tt::LogTest, "Node 0: Testing forward to South via initial_dir={}", initial_dir);
        for (uint32_t dst : {4u, 5u, 6u, 7u}) {
            log_info(tt::LogTest, "  Node 0->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(0u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 0->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(0u, dst), initial_dir);
        }
    }
}

// Node 1 has neighbors in E, W, and S directions - test all 9 combinations
// Forwarding to East destinations (2, 3) via non-East initial directions (W, S)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode1ForwardEast) {
    for (auto initial_dir : {RoutingDirection::W, RoutingDirection::S}) {
        log_info(tt::LogTest, "Node 1: Testing forward to East via initial_dir={}", initial_dir);
        for (uint32_t dst : {2u, 3u}) {
            log_info(tt::LogTest, "  Node 1->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(1u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 1->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(1u, dst), initial_dir);
        }
    }
}

// Forwarding to West destinations (0) via non-West initial directions (E, S)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode1ForwardWest) {
    for (auto initial_dir : {RoutingDirection::E, RoutingDirection::S}) {
        log_info(tt::LogTest, "Node 1: Testing forward to West via initial_dir={}", initial_dir);
        log_info(tt::LogTest, "  Node 1->Node 0: Write");
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(1u, 0u), initial_dir);
        log_info(tt::LogTest, "  Node 1->Node 0: Read");
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(1u, 0u), initial_dir);
    }
}

// Forwarding to South destinations (4, 5, 6, 7) via non-South initial directions (E, W)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode1ForwardSouth) {
    for (auto initial_dir : {RoutingDirection::E, RoutingDirection::W}) {
        log_info(tt::LogTest, "Node 1: Testing forward to South via initial_dir={}", initial_dir);
        for (uint32_t dst : {4u, 5u, 6u, 7u}) {
            log_info(tt::LogTest, "  Node 1->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(1u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 1->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(1u, dst), initial_dir);
        }
    }
}

// Node 2 has neighbors in E, W, and S directions - test all 9 combinations
// Forwarding to East destinations (3) via non-East initial directions (W, S)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode2ForwardEast) {
    for (auto initial_dir : {RoutingDirection::W, RoutingDirection::S}) {
        log_info(tt::LogTest, "Node 2: Testing forward to East via initial_dir={}", initial_dir);
        log_info(tt::LogTest, "  Node 2->Node 3: Write");
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(2u, 3u), initial_dir);
        log_info(tt::LogTest, "  Node 2->Node 3: Read");
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(2u, 3u), initial_dir);
    }
}

// Forwarding to West destinations (0, 1) via non-West initial directions (E, S)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode2ForwardWest) {
    for (auto initial_dir : {RoutingDirection::E, RoutingDirection::S}) {
        log_info(tt::LogTest, "Node 2: Testing forward to West via initial_dir={}", initial_dir);
        for (uint32_t dst : {0u, 1u}) {
            log_info(tt::LogTest, "  Node 2->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(2u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 2->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(2u, dst), initial_dir);
        }
    }
}

// Forwarding to South destinations (6, 7) via non-South initial directions (E, W)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode2ForwardSouth) {
    for (auto initial_dir : {RoutingDirection::E, RoutingDirection::W}) {
        log_info(tt::LogTest, "Node 2: Testing forward to South via initial_dir={}", initial_dir);
        for (uint32_t dst : {6u, 7u}) {
            log_info(tt::LogTest, "  Node 2->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(2u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 2->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(2u, dst), initial_dir);
        }
    }
}

// Node 3 has neighbors in W and S directions
// Forwarding to West destinations (0, 1, 2) via non-West initial direction (S)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode3ForwardWest) {
    for (auto initial_dir : {RoutingDirection::S}) {
        log_info(tt::LogTest, "Node 3: Testing forward to West via initial_dir={}", initial_dir);
        for (uint32_t dst : {0u, 1u, 2u}) {
            log_info(tt::LogTest, "  Node 3->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(3u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 3->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(3u, dst), initial_dir);
        }
    }
}

// Forwarding to South destinations (4, 5, 6, 7) via non-South initial direction (W)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode3ForwardSouth) {
    for (auto initial_dir : {RoutingDirection::W}) {
        log_info(tt::LogTest, "Node 3: Testing forward to South via initial_dir={}", initial_dir);
        for (uint32_t dst : {4u, 5u, 6u, 7u}) {
            log_info(tt::LogTest, "  Node 3->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(3u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 3->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(3u, dst), initial_dir);
        }
    }
}

// Node 4 has neighbors in E and N directions
// Forwarding to East destinations (5, 6, 7) via non-East initial direction (N)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode4ForwardEast) {
    for (auto initial_dir : {RoutingDirection::N}) {
        log_info(tt::LogTest, "Node 4: Testing forward to East via initial_dir={}", initial_dir);
        for (uint32_t dst : {5u, 6u, 7u}) {
            log_info(tt::LogTest, "  Node 4->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(4u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 4->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(4u, dst), initial_dir);
        }
    }
}

// Forwarding to North destinations (0, 1, 2, 3) via non-North initial direction (E)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode4ForwardNorth) {
    for (auto initial_dir : {RoutingDirection::E}) {
        log_info(tt::LogTest, "Node 4: Testing forward to North via initial_dir={}", initial_dir);
        for (uint32_t dst : {0u, 1u, 2u, 3u}) {
            log_info(tt::LogTest, "  Node 4->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(4u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 4->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(4u, dst), initial_dir);
        }
    }
}

// Node 5 has neighbors in E, W, and N directions - test all 9 combinations
// Forwarding to East destinations (6, 7) via non-East initial directions (W, N)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode5ForwardEast) {
    for (auto initial_dir : {RoutingDirection::W, RoutingDirection::N}) {
        log_info(tt::LogTest, "Node 5: Testing forward to East via initial_dir={}", initial_dir);
        for (uint32_t dst : {6u, 7u}) {
            log_info(tt::LogTest, "  Node 5->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(5u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 5->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(5u, dst), initial_dir);
        }
    }
}

// Forwarding to West destinations (4) via non-West initial directions (E, N)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode5ForwardWest) {
    for (auto initial_dir : {RoutingDirection::E, RoutingDirection::N}) {
        log_info(tt::LogTest, "Node 5: Testing forward to West via initial_dir={}", initial_dir);
        log_info(tt::LogTest, "  Node 5->Node 4: Write");
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(5u, 4u), initial_dir);
        log_info(tt::LogTest, "  Node 5->Node 4: Read");
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(5u, 4u), initial_dir);
    }
}

// Forwarding to North destinations (0, 1, 2, 3) via non-North initial directions (E, W)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode5ForwardNorth) {
    for (auto initial_dir : {RoutingDirection::E, RoutingDirection::W}) {
        log_info(tt::LogTest, "Node 5: Testing forward to North via initial_dir={}", initial_dir);
        for (uint32_t dst : {0u, 1u, 2u, 3u}) {
            log_info(tt::LogTest, "  Node 5->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(5u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 5->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(5u, dst), initial_dir);
        }
    }
}

// Node 6 has neighbors in E, W, and N directions - test all 9 combinations
// Forwarding to East destinations (7) via non-East initial directions (W, N)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode6ForwardEast) {
    for (auto initial_dir : {RoutingDirection::W, RoutingDirection::N}) {
        log_info(tt::LogTest, "Node 6: Testing forward to East via initial_dir={}", initial_dir);
        log_info(tt::LogTest, "  Node 6->Node 7: Write");
        UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(6u, 7u), initial_dir);
        log_info(tt::LogTest, "  Node 6->Node 7: Read");
        UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(6u, 7u), initial_dir);
    }
}

// Forwarding to West destinations (4, 5) via non-West initial directions (E, N)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode6ForwardWest) {
    for (auto initial_dir : {RoutingDirection::E, RoutingDirection::N}) {
        log_info(tt::LogTest, "Node 6: Testing forward to West via initial_dir={}", initial_dir);
        for (uint32_t dst : {4u, 5u}) {
            log_info(tt::LogTest, "  Node 6->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(6u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 6->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(6u, dst), initial_dir);
        }
    }
}

// Forwarding to North destinations (0, 1, 2, 3) via non-North initial directions (E, W)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode6ForwardNorth) {
    for (auto initial_dir : {RoutingDirection::E, RoutingDirection::W}) {
        log_info(tt::LogTest, "Node 6: Testing forward to North via initial_dir={}", initial_dir);
        for (uint32_t dst : {0u, 1u, 2u, 3u}) {
            log_info(tt::LogTest, "  Node 6->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(6u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 6->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(6u, dst), initial_dir);
        }
    }
}

// Node 7 has neighbors in W and N directions
// Forwarding to West destinations (4, 5, 6) via non-West initial direction (N)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode7ForwardWest) {
    for (auto initial_dir : {RoutingDirection::N}) {
        log_info(tt::LogTest, "Node 7: Testing forward to West via initial_dir={}", initial_dir);
        for (uint32_t dst : {4u, 5u, 6u}) {
            log_info(tt::LogTest, "  Node 7->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(7u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 7->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(7u, dst), initial_dir);
        }
    }
}

// Forwarding to North destinations (0, 1, 2, 3) via non-North initial direction (W)
TEST_F(NightlyFabric2DUDMModeFixture, TestUDMFabricMuxToMuxNode7ForwardNorth) {
    for (auto initial_dir : {RoutingDirection::W}) {
        log_info(tt::LogTest, "Node 7: Testing forward to North via initial_dir={}", initial_dir);
        for (uint32_t dst : {0u, 1u, 2u, 3u}) {
            log_info(tt::LogTest, "  Node 7->Node {}: Write", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_WRITE, std::make_tuple(7u, dst), initial_dir);
            log_info(tt::LogTest, "  Node 7->Node {}: Read", dst);
            UDMFabricUnicastCommon(this, NOC_UNICAST_READ, std::make_tuple(7u, dst), initial_dir);
        }
    }
}

// Unicast Scatter Write
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocScatterWrite) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_SCATTER_WRITE, {std::make_tuple(dir, 1)}, FabricApiType::Mesh);
    }
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocScatterWriteMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Mesh);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocScatterWriteWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Mesh,
        true);
}

// Unicast Atomic Inc
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocAtomicInc) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, {std::make_tuple(dir, 1)}, FabricApiType::Mesh);
    }
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocAtomicIncMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Mesh);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocAtomicIncWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Mesh,
        true);
}

// Unicast Fused Atomic Inc
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocFusedAtomicInc) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_FUSED_UNICAST_ATOMIC_INC, {std::make_tuple(dir, 1)}, FabricApiType::Mesh);
    }
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocFusedAtomicIncMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Mesh);
}
TEST_F(NightlyFabric2DFixture, TestMeshFabricUnicastNocFusedAtomicIncWithState) {
    FabricUnicastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Mesh,
        true);
}

// 2D topology Linear API tests (using 1D Linear API semantics)
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocUnicastWrite) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_WRITE, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocUnicastWriteMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocUnicastWriteWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Linear,
        true);
}

TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocInlineWrite) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_INLINE_WRITE, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocInlineWriteMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocInlineWriteWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Linear,
        true);
}

// Unicast Scatter Write
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocScatterWrite) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_SCATTER_WRITE, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocScatterWriteMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocScatterWriteWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Linear,
        true);
}

// Unicast Atomic Inc
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocAtomicInc) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_UNICAST_ATOMIC_INC, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocAtomicIncMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocAtomicIncWithState) {
    FabricUnicastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Linear,
        true);
}

// Unicast Fused Atomic Inc
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocFusedAtomicInc) {
    for (auto dir : {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
        FabricUnicastCommon(this, NOC_FUSED_UNICAST_ATOMIC_INC, {std::make_tuple(dir, 1)});
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocFusedAtomicIncMultiDir) {
    FabricUnicastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricUnicastNocFusedAtomicIncWithState) {
    FabricUnicastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1),
         std::make_tuple(RoutingDirection::W, 2),
         std::make_tuple(RoutingDirection::N, 1),
         std::make_tuple(RoutingDirection::S, 1)},
        FabricApiType::Linear,
        true);
}

TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocUnicastWrite) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_UNICAST_WRITE, cfg);
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocUnicastWriteMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocUnicastWriteWithState) {
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
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocInlineWrite) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_UNICAST_INLINE_WRITE, cfg);
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocInlineWriteMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_INLINE_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocInlineWriteWithState) {
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
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocScatterWrite) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_UNICAST_SCATTER_WRITE, cfg);
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocScatterWriteMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_SCATTER_WRITE,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocScatterWriteWithState) {
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
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocAtomicInc) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_UNICAST_ATOMIC_INC, cfg);
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocAtomicIncMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocAtomicIncWithState) {
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
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocFusedAtomicInc) {
    std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>> pairs = {
        {std::make_tuple(RoutingDirection::E, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
        {std::make_tuple(RoutingDirection::N, 1, 1), std::make_tuple(RoutingDirection::W, 1, 2)},
    };
    for (auto& cfg : pairs) {
        FabricMulticastCommon(this, NOC_FUSED_UNICAST_ATOMIC_INC, cfg);
    }
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocFusedAtomicIncMultiDir) {
    FabricMulticastCommon(
        this,
        NOC_FUSED_UNICAST_ATOMIC_INC,
        {std::make_tuple(RoutingDirection::E, 1, 1),
         std::make_tuple(RoutingDirection::W, 1, 2),
         std::make_tuple(RoutingDirection::N, 1, 1),
         std::make_tuple(RoutingDirection::S, 1, 1)});
}
TEST_F(NightlyFabric2DFixture, TestLinearFabricMulticastNocFusedAtomicIncWithState) {
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
TEST_P(T3kCustomMeshGraphFabric2DFixture, TestUnicastRaw) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    CustomMeshGraphFabric2DFixture::SetUp(
        mesh_graph_desc_path, get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
    for (uint32_t i = 0; i < 10; i++) {
        RunTestUnicastRaw(this);
    }
}

TEST_F(Fabric2DFixture, TestGetNextHopRouterDirection1MeshAllToAll) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::tt_metal::ClusterType::TG) {
        GTEST_SKIP() << "Test not applicable for TG cluster type";
    }
    RunGetNextHopRouterDirectionTest(this, false);
}

// Multi-Mesh Test - Using parameterized test with connected mesh descriptor
TEST_P(T3kCustomMeshGraphFabric2DFixture, TestGetNextHopRouterDirectionMultiMesh) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    CustomMeshGraphFabric2DFixture::SetUp(
        mesh_graph_desc_path, get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
    RunGetNextHopRouterDirectionTest(this, true);
}

// Skipping other t3k configs because multi-mesh in single process isn't supported
INSTANTIATE_TEST_SUITE_P(
    T3kCustomMeshGraphFabric2DTests,
    T3kCustomMeshGraphFabric2DFixture,
    ::testing::Values(t3k_mesh_descriptor_chip_mappings[0]));

// 2D Routing Unidirectional mcast tests (no turns)
TEST_F(Fabric2DFixture, TestLineMcastE2Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastE3Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastE7Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastW2Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastW3Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastW7Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 7};
    RunTestLineMcast(this, {routing_info});
}

// 2D Dynamic Routing Unidirectional mcast tests (with turns)
TEST_F(Fabric2DFixture, TestLineMcastN1HopE3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastN1HopE7Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, n_routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastN2HopE3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastN2HopE7Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, n_routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastS1HopE3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastS1HopE7Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, s_routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastS2HopE3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastS2HopE7Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, s_routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastN1HopW3Hops) {
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {w_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastN1HopW7Hops) {
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 7};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {w_routing_info, n_routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastS1HopW3Hops) {
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastS1HopW7Hops) {
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 7};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {w_routing_info, s_routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastS1HopE1HopW1Hop) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastN1HopE1HopW1Hop) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastS1HopE2HopsW1Hop) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastN1HopE2HopsW1Hop) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastS1HopE1HopW2Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastN1HopE1HopW2Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(Fabric2DFixture, TestLineMcastS2HopsE4HopsW3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 4};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastS2HopsE3HopsW4Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 4};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastS3HopsE4HopsW3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 4};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 3};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastS3HopsE3HopsW4Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 4};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 3};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastN2HopsE4HopsW3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 4};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastN2HopsE3HopsW4Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 4};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastN3HopsE4HopsW3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 4};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 3};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DFixture, TestLineMcastN3HopsE3HopsW4Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 4};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 3};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(Fabric1DFixture, TestSetUnicastRoute) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::T3K) {
        GTEST_SKIP() << "Test applicable only on T3K";
    }
    RunSetUnicastRouteTest(this, false);
}

TEST_F(Fabric1DFixture, TestSetUnicastRouteIdleEth) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::T3K) {
        GTEST_SKIP() << "Test applicable only on T3K";
    }
    RunSetUnicastRouteTest(this, false, HalProgrammableCoreType::IDLE_ETH);
}

// 1 mesh all-to-all
TEST_F(Fabric2DFixture, TestSetUnicastRoute) { RunSetUnicastRouteTest(this, false); }

TEST_F(Fabric2DFixture, TestSetUnicastRouteIdleEth) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::T3K) {
        GTEST_SKIP() << "Test applicable only on T3K";
    }
    RunSetUnicastRouteTest(this, false, HalProgrammableCoreType::IDLE_ETH);
}

// Test for std::atomic_exchange on kernels using BRISC and NCRISC
}  // namespace tt::tt_fabric::fabric_router_tests
