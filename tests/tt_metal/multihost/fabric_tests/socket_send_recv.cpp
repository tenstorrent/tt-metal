// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "multihost_fabric_fixtures.hpp"
#include "tests/tt_metal/multihost/fabric_tests/socket_send_recv_utils.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <random>
#include <algorithm>

namespace tt::tt_fabric::fabric_router_tests::multihost {

using namespace multihost_utils;

template <typename FixtureType>
class MultiHostSocketTest : public FixtureType, public ::testing::WithParamInterface<SocketTestConfig> {
public:
    void RunTest() {
        auto config = GetParam();

        log_info(tt::LogTest, "Socket Test Variant: {} ", get_test_variant_name(config.variant));
        log_info(tt::LogTest, "Socket Buffer Size: {} bytes", config.socket_fifo_size);
        log_info(tt::LogTest, "Socket Page Size: {} bytes", config.socket_page_size);
        log_info(tt::LogTest, "Data Size: {} bytes", config.data_size);
        log_info(
            tt::LogTest,
            "Host Rank: {}",
            *tt::tt_metal::distributed::multihost::DistributedContext::get_current_world()->rank());

        // Call the appropriate test function based on variant
        switch (config.variant) {
            case TestVariant::SINGLE_CONN_BWD:
                test_multi_mesh_single_conn_bwd(
                    this->mesh_device_,
                    config.socket_fifo_size,
                    config.socket_page_size,
                    config.data_size,
                    config.system_config);
                break;
            case TestVariant::SINGLE_CONN_FWD:
                test_multi_mesh_single_conn_fwd(
                    this->mesh_device_,
                    config.socket_fifo_size,
                    config.socket_page_size,
                    config.data_size,
                    config.system_config);
                break;
            case TestVariant::MULTI_CONN_FWD:
                test_multi_mesh_multi_conn_fwd(
                    this->mesh_device_,
                    config.socket_fifo_size,
                    config.socket_page_size,
                    config.data_size,
                    config.system_config);
                break;
            case TestVariant::MULTI_CONN_BIDIR:
                test_multi_mesh_multi_conn_bidirectional(
                    this->mesh_device_,
                    config.socket_fifo_size,
                    config.socket_page_size,
                    config.data_size,
                    config.system_config);
                break;
        }
    }
};

std::vector<SocketTestConfig> generate_socket_test_configs(SystemConfig system_config) {
    std::vector<SocketTestConfig> configs;

    std::vector<uint32_t> fifo_sizes = {1024, 2048, 4096, 512, 1024};
    std::vector<uint32_t> page_sizes = {64, 256, 1088, 128, 128};
    std::vector<uint32_t> data_sizes = {2048, 4096, 78336, 16384, 4096};
    std::vector<TestVariant> variants = {
        TestVariant::SINGLE_CONN_BWD,
        TestVariant::SINGLE_CONN_FWD,
        TestVariant::MULTI_CONN_FWD,
        TestVariant::MULTI_CONN_BIDIR};

    for (int config_idx = 0; config_idx < fifo_sizes.size(); ++config_idx) {
        for (const auto& variant : variants) {
            configs.push_back(
                {.socket_fifo_size = fifo_sizes[config_idx],
                 .socket_page_size = page_sizes[config_idx],
                 .data_size = data_sizes[config_idx],
                 .variant = variant,
                 .system_config = system_config});
        }
    }
    return configs;
}

template <typename ParamType>
std::string generate_multihost_socket_test_name(const testing::TestParamInfo<ParamType>& info) {
    return get_test_variant_name(info.param.variant) + "_" + get_system_config_name(info.param.system_config) + "_" +
           std::to_string(info.param.socket_fifo_size) + "_" + std::to_string(info.param.socket_page_size) + "_" +
           std::to_string(info.param.data_size);
}

using MultiHostSocketTestSplitT3K = MultiHostSocketTest<MeshDeviceSplit2x2Fixture>;
using MultiHostSocketTestDualT3K = MultiHostSocketTest<MeshDeviceDual2x4Fixture>;
using MeshDeviceNanoExabox2x4Fixture = MultiHostSocketTest<MeshDeviceNanoExabox2x4Fixture>;
using MeshDeviceNanoExabox1x8Fixture = MultiHostSocketTest<MeshDeviceNanoExabox1x8Fixture>;
using MultiHostSocketTestExabox = MultiHostSocketTest<MeshDeviceExaboxFixture>;
using MultiHostSocketTestSplitGalaxy = MultiHostSocketTest<SplitGalaxyMeshDeviceFixture>;

TEST_P(MultiHostSocketTestSplitT3K, SocketTests) { RunTest(); }

TEST_P(MultiHostSocketTestDualT3K, SocketTests) { RunTest(); }

TEST_P(MeshDeviceNanoExabox2x4Fixture, SocketTests) { RunTest(); }

TEST_P(MeshDeviceNanoExabox1x8Fixture, SocketTests) { RunTest(); }

TEST_P(MultiHostSocketTestExabox, SocketTests) { RunTest(); }

TEST_P(MultiHostSocketTestSplitGalaxy, SocketTests) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(
    MultiHostSocketTestsSplitT3K,
    MultiHostSocketTestSplitT3K,
    ::testing::ValuesIn(generate_socket_test_configs(SystemConfig::SPLIT_T3K)),
    generate_multihost_socket_test_name<MultiHostSocketTestSplitT3K::ParamType>);

INSTANTIATE_TEST_SUITE_P(
    MultiHostSocketTestsDualT3K,
    MultiHostSocketTestDualT3K,
    ::testing::ValuesIn(generate_socket_test_configs(SystemConfig::DUAL_T3K)),
    generate_multihost_socket_test_name<MultiHostSocketTestDualT3K::ParamType>);

INSTANTIATE_TEST_SUITE_P(
    MeshDeviceNanoExabox2x4Fixture,
    MeshDeviceNanoExabox2x4Fixture,
    ::testing::ValuesIn(generate_socket_test_configs(SystemConfig::NANO_EXABOX)),
    generate_multihost_socket_test_name<MultiHostSocketTestDualT3K::ParamType>);

INSTANTIATE_TEST_SUITE_P(
    MeshDeviceNanoExabox1x8Fixture,
    MeshDeviceNanoExabox1x8Fixture,
    ::testing::ValuesIn(generate_socket_test_configs(SystemConfig::NANO_EXABOX)),
    generate_multihost_socket_test_name<MultiHostSocketTestDualT3K::ParamType>);

INSTANTIATE_TEST_SUITE_P(
    MultiHostSocketTestsExabox,
    MultiHostSocketTestExabox,
    ::testing::ValuesIn(generate_socket_test_configs(SystemConfig::EXABOX)),
    generate_multihost_socket_test_name<MultiHostSocketTestExabox::ParamType>);

INSTANTIATE_TEST_SUITE_P(
    MultiHostSocketTestsSplitGalaxy,
    MultiHostSocketTestSplitGalaxy,
    ::testing::ValuesIn(generate_socket_test_configs(SystemConfig::SPLIT_GALAXY)),
    generate_multihost_socket_test_name<MultiHostSocketTestSplitGalaxy::ParamType>);

TEST_F(MeshDeviceNanoExabox2x4Fixture, MultiContextSocketHandshake) {
    std::vector<int> sender_node_ranks_ctx0 = {0, 2, 3, 4};
    uint32_t recv_rank_ctx0 = 1;

    std::vector<int> ctx1_ranks = sender_node_ranks_ctx0;
    std::vector<int> sender_node_ranks_ctx1 = {0, 2, 3};
    uint32_t recv_rank_ctx1 = 1;

    const auto& distributed_ctx0 = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    std::unordered_map<uint32_t, tt_metal::distributed::MeshSocket> sockets_ctx0;
    std::unordered_map<uint32_t, tt_metal::distributed::MeshSocket> sockets_ctx1;

    auto socket_connection = tt_metal::distributed::SocketConnection(
        tt_metal::distributed::MeshCoreCoord(MeshCoordinate(0, 0), tt_metal::CoreCoord(0, 0)),
        tt_metal::distributed::MeshCoreCoord(MeshCoordinate(0, 0), tt_metal::CoreCoord(0, 0)));

    auto socket_mem_config = tt_metal::distributed::SocketMemoryConfig(tt_metal::BufferType::L1, 1024);

    // Initialize sockets in context0 namespace
    if (*distributed_ctx0->rank() == recv_rank_ctx0) {
        for (const auto& sender_rank : sender_node_ranks_ctx0) {
            tt_metal::distributed::SocketConfig socket_config(
                {socket_connection},
                socket_mem_config,
                tt_metal::distributed::multihost::Rank{sender_rank},
                distributed_ctx0->rank(),
                distributed_ctx0);
            sockets_ctx0.emplace(sender_rank, tt_metal::distributed::MeshSocket(mesh_device_, socket_config));
        }
    } else if (
        std::find(sender_node_ranks_ctx0.begin(), sender_node_ranks_ctx0.end(), *distributed_ctx0->rank()) !=
        sender_node_ranks_ctx0.end()) {
        tt_metal::distributed::SocketConfig socket_config(
            {socket_connection},
            socket_mem_config,
            distributed_ctx0->rank(),
            tt_metal::distributed::multihost::Rank{recv_rank_ctx0},
            distributed_ctx0);
        sockets_ctx0.emplace(recv_rank_ctx0, tt_metal::distributed::MeshSocket(mesh_device_, socket_config));
    }
    // Initialize sockets in context1 namespace
    if (std::find(ctx1_ranks.begin(), ctx1_ranks.end(), *distributed_ctx0->rank()) != ctx1_ranks.end()) {
        auto distributed_ctx1 = distributed_ctx0->create_sub_context(ctx1_ranks);
        if (*distributed_ctx1->rank() == recv_rank_ctx1) {
            for (const auto& sender_rank : sender_node_ranks_ctx1) {
                tt_metal::distributed::SocketConfig socket_config(
                    {socket_connection},
                    socket_mem_config,
                    tt_metal::distributed::multihost::Rank{sender_rank},
                    distributed_ctx1->rank(),
                    distributed_ctx1);
                sockets_ctx1.emplace(sender_rank, tt_metal::distributed::MeshSocket(mesh_device_, socket_config));
            }
        } else if (
            std::find(sender_node_ranks_ctx1.begin(), sender_node_ranks_ctx1.end(), *distributed_ctx1->rank()) !=
            sender_node_ranks_ctx1.end()) {
            tt_metal::distributed::SocketConfig socket_config(
                {socket_connection},
                socket_mem_config,
                distributed_ctx1->rank(),
                tt_metal::distributed::multihost::Rank{recv_rank_ctx1},
                distributed_ctx1);
            sockets_ctx1.emplace(recv_rank_ctx1, tt_metal::distributed::MeshSocket(mesh_device_, socket_config));
        }
    }
}

TEST_F(SplitGalaxyMeshDeviceFixture, SocketSubContextValidation) {
    // Create a SubContext with sender rank = 0 and receiver rank = 2
    // This validates that the Distributed Context Rank Translation functionality
    // supported by sockets works correctly.
    // Test 1:
    // Ranks corresponding to the sub context will be passed into the socket config.
    // Mesh Ids will be derived based on the translated values of these ranks.
    // Test 2:
    // Mesh Ids will be passed into the socket config. A translation table converting
    // their global host ranks to the sub context ranks will be generated.
    // Test 3:
    // An Invalid Socket Connection (that uses coordinates outside the sub context) will be passed into the socket
    // config.
    using namespace tt_metal::distributed::multihost;
    std::vector<int> handshake_ranks = {0, 2};
    const auto& parent_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    auto socket_connection = tt_metal::distributed::SocketConnection(
        tt_metal::distributed::MeshCoreCoord(MeshCoordinate(0, 0), tt_metal::CoreCoord(0, 0)),
        tt_metal::distributed::MeshCoreCoord(MeshCoordinate(0, 0), tt_metal::CoreCoord(0, 0)));

    auto invalid_socket_connection = tt_metal::distributed::SocketConnection(
        tt_metal::distributed::MeshCoreCoord(MeshCoordinate(3, 3), tt_metal::CoreCoord(0, 0)),
        tt_metal::distributed::MeshCoreCoord(MeshCoordinate(3, 3), tt_metal::CoreCoord(0, 0)));

    auto socket_mem_config = tt_metal::distributed::SocketMemoryConfig(tt_metal::BufferType::L1, 1024);

    if (parent_context->rank() == Rank{0}) {
        auto sub_context = parent_context->create_sub_context(handshake_ranks);
        tt_metal::distributed::SocketConfig socket_config_0(
            {socket_connection}, socket_mem_config, Rank{0}, Rank{1}, sub_context);
        tt_metal::distributed::SocketConfig socket_config_1(
            {socket_connection}, socket_mem_config, MeshId{0}, MeshId{1}, sub_context);
        tt_metal::distributed::SocketConfig socket_config_2(
            {invalid_socket_connection}, socket_mem_config, Rank{0}, Rank{1}, sub_context);
        auto send_socket_0 = tt_metal::distributed::MeshSocket(mesh_device_, socket_config_0);
        auto send_socket_1 = tt_metal::distributed::MeshSocket(mesh_device_, socket_config_1);
        EXPECT_THROW(tt_metal::distributed::MeshSocket(mesh_device_, socket_config_2), std::exception);
    } else if (parent_context->rank() == Rank{2}) {
        auto sub_context = parent_context->create_sub_context(handshake_ranks);
        tt_metal::distributed::SocketConfig socket_config_0(
            {socket_connection}, socket_mem_config, Rank{0}, Rank{1}, sub_context);
        tt_metal::distributed::SocketConfig socket_config_1(
            {socket_connection}, socket_mem_config, MeshId{0}, MeshId{1}, sub_context);
        tt_metal::distributed::SocketConfig socket_config_2(
            {invalid_socket_connection}, socket_mem_config, Rank{0}, Rank{1}, sub_context);
        auto recv_socket_0 = tt_metal::distributed::MeshSocket(mesh_device_, socket_config_0);
        auto recv_socket_1 = tt_metal::distributed::MeshSocket(mesh_device_, socket_config_1);
        EXPECT_THROW(tt_metal::distributed::MeshSocket(mesh_device_, socket_config_2), std::exception);
    }
    parent_context->barrier();
}

// Test that socket creation works correctly when the sender and receiver ranks are provided.
TEST_F(SplitGalaxyMeshDeviceFixture, RankBasedSocketCreation) {
    using namespace tt_metal::distributed::multihost;
    auto& metal_context = tt::tt_metal::MetalContext::instance();
    const auto& distributed_context = metal_context.global_distributed_context();

    uint32_t socket_fifo_size = 1024;
    auto sender_rank_0 = Rank{0};
    auto recv_rank_0 = Rank{2};
    auto sender_rank_1 = Rank{1};
    auto recv_rank_1 = Rank{3};

    auto socket_mem_config = tt_metal::distributed::SocketMemoryConfig(tt_metal::BufferType::L1, socket_fifo_size);
    auto socket_connection_0 = tt_metal::distributed::SocketConnection(
        tt_metal::distributed::MeshCoreCoord(MeshCoordinate(0, 0), tt_metal::CoreCoord(0, 0)),
        tt_metal::distributed::MeshCoreCoord(MeshCoordinate(0, 0), tt_metal::CoreCoord(0, 0)));
    auto socket_connection_1 = tt_metal::distributed::SocketConnection(
        tt_metal::distributed::MeshCoreCoord(MeshCoordinate(3, 3), tt_metal::CoreCoord(0, 0)),
        tt_metal::distributed::MeshCoreCoord(MeshCoordinate(3, 3), tt_metal::CoreCoord(0, 0)));

    tt_metal::distributed::SocketConfig socket_config_0(
        {socket_connection_0}, socket_mem_config, sender_rank_0, recv_rank_0);
    tt_metal::distributed::SocketConfig socket_config_1(
        {socket_connection_1}, socket_mem_config, sender_rank_1, recv_rank_1);
    if (*distributed_context.rank() == 0) {
        auto send_socket = tt_metal::distributed::MeshSocket(mesh_device_, socket_config_0);
    } else if (*distributed_context.rank() == 2) {
        auto recv_socket = tt_metal::distributed::MeshSocket(mesh_device_, socket_config_0);
    } else if (*distributed_context.rank() == 1) {
        auto send_socket = tt_metal::distributed::MeshSocket(mesh_device_, socket_config_1);
    } else if (*distributed_context.rank() == 3) {
        auto recv_socket = tt_metal::distributed::MeshSocket(mesh_device_, socket_config_1);
    }
    distributed_context.barrier();
}

// Generate a random pairing of sender and receiver device coordinates.
std::vector<tt_metal::distributed::SocketConnection> generate_random_socket_connections(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    tt_fabric::MeshId sender_mesh_id,
    tt_fabric::MeshId recv_mesh_id) {
    std::mt19937 gen = std::mt19937(sync_seed_across_ranks(sender_mesh_id, recv_mesh_id));
    auto mesh_device_shape = mesh_device->shape();
    auto sender_core = CoreCoord(0, 0);
    auto recv_core = CoreCoord(0, 0);

    std::set<MeshCoordinate> generated_sender_device_coords;
    std::set<MeshCoordinate> generated_recv_device_coords;
    std::vector<tt_metal::distributed::SocketConnection> socket_connections;

    while (generated_sender_device_coords.size() < mesh_device_shape[0] * mesh_device_shape[1]) {
        auto sender_device_coord = MeshCoordinate(gen() % mesh_device_shape[0], gen() % mesh_device_shape[1]);
        auto recv_device_coord = MeshCoordinate(gen() % mesh_device_shape[0], gen() % mesh_device_shape[1]);
        while (generated_sender_device_coords.contains(sender_device_coord)) {
            sender_device_coord = MeshCoordinate(gen() % mesh_device_shape[0], gen() % mesh_device_shape[1]);
        }
        while (generated_recv_device_coords.contains(recv_device_coord)) {
            recv_device_coord = MeshCoordinate(gen() % mesh_device_shape[0], gen() % mesh_device_shape[1]);
        }
        generated_sender_device_coords.insert(sender_device_coord);
        generated_recv_device_coords.insert(recv_device_coord);
        socket_connections.push_back(tt_metal::distributed::SocketConnection(
            tt_metal::distributed::MeshCoreCoord(sender_device_coord, sender_core),
            tt_metal::distributed::MeshCoreCoord(recv_device_coord, recv_core)));
    }
    return socket_connections;
}

TEST_F(SplitGalaxyMeshDeviceFixture, BigMeshSocketRandomConnections) {
    auto& metal_context = tt::tt_metal::MetalContext::instance();
    const auto& distributed_context = metal_context.global_distributed_context();

    uint32_t socket_fifo_size = 1024;
    auto sender_mesh_id = tt::tt_fabric::MeshId{0};
    auto recv_mesh_id = tt::tt_fabric::MeshId{1};
    uint32_t data_size = 8192;
    uint32_t page_size = 64;
    uint32_t num_txns = 20;

    auto socket_mem_config = tt_metal::distributed::SocketMemoryConfig(tt_metal::BufferType::L1, socket_fifo_size);

    tt_metal::distributed::SocketConfig socket_config(
        {generate_random_socket_connections(mesh_device_, sender_mesh_id, recv_mesh_id)},
        socket_mem_config,
        sender_mesh_id,
        recv_mesh_id);

    auto local_mesh_binding = tt::tt_metal::MetalContext::instance().get_control_plane().get_local_mesh_id_bindings();
    TT_FATAL(local_mesh_binding.size() == 1, "Local mesh binding must be exactly one.");

    if (local_mesh_binding[0] == sender_mesh_id) {
        auto send_socket = tt_metal::distributed::MeshSocket(mesh_device_, socket_config);
        test_socket_send_recv(mesh_device_, send_socket, data_size, page_size, num_txns, std::nullopt);
    } else {
        auto recv_socket = tt_metal::distributed::MeshSocket(mesh_device_, socket_config);
        test_socket_send_recv(mesh_device_, recv_socket, data_size, page_size, num_txns, std::nullopt);
    }
    distributed_context.barrier();
}

}  // namespace tt::tt_fabric::fabric_router_tests::multihost
