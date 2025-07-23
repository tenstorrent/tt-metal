// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <stdint.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "multihost_fabric_fixtures.hpp"
#include "tests/tt_metal/multihost/fabric_tests/socket_send_recv_utils.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric.hpp>

#include <random>
#include <algorithm>

namespace tt::tt_fabric {
namespace fabric_router_tests::multihost {

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

TEST_P(MultiHostSocketTestSplitT3K, SocketTests) { RunTest(); }

TEST_P(MultiHostSocketTestDualT3K, SocketTests) { RunTest(); }

TEST_P(MeshDeviceNanoExabox2x4Fixture, SocketTests) { RunTest(); }

TEST_P(MeshDeviceNanoExabox1x8Fixture, SocketTests) { RunTest(); }

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

TEST_F(MeshDeviceNanoExabox2x4Fixture, MultiContextSocketHandshake) {
    std::vector<int> sender_node_ranks_ctx0 = {0, 2, 3, 4};
    uint32_t recv_rank_ctx0 = 1;

    std::vector<int> ctx1_ranks = sender_node_ranks_ctx0;
    std::vector<int> sender_node_ranks_ctx1 = {0, 2, 3};
    uint32_t recv_rank_ctx1 = 1;

    auto distributed_ctx0 = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    std::unordered_map<uint32_t, tt_metal::distributed::MeshSocket> sockets_ctx0;
    std::unordered_map<uint32_t, tt_metal::distributed::MeshSocket> sockets_ctx1;

    auto socket_connection = tt_metal::distributed::SocketConnection{
        .sender_core = {MeshCoordinate(0, 0), tt_metal::CoreCoord(0, 0)},
        .receiver_core = {MeshCoordinate(0, 0), tt_metal::CoreCoord(0, 0)}};

    auto socket_mem_config = tt_metal::distributed::SocketMemoryConfig{
        .socket_storage_type = tt_metal::BufferType::L1,
        .fifo_size = 1024,
    };

    // Initialize sockets in context0 namespace
    if (*distributed_ctx0->rank() == recv_rank_ctx0) {
        for (const auto& sender_rank : sender_node_ranks_ctx0) {
            tt_metal::distributed::SocketConfig socket_config = {
                .socket_connection_config = {socket_connection},
                .socket_mem_config = socket_mem_config,
                .sender_rank = tt_metal::distributed::multihost::Rank{sender_rank},
                .receiver_rank = distributed_ctx0->rank(),
                .distributed_context = distributed_ctx0};
            sockets_ctx0.emplace(sender_rank, tt_metal::distributed::MeshSocket(mesh_device_, socket_config));
        }
    } else if (
        std::find(sender_node_ranks_ctx0.begin(), sender_node_ranks_ctx0.end(), *distributed_ctx0->rank()) !=
        sender_node_ranks_ctx0.end()) {
        tt_metal::distributed::SocketConfig socket_config = {
            .socket_connection_config = {socket_connection},
            .socket_mem_config = socket_mem_config,
            .sender_rank = distributed_ctx0->rank(),
            .receiver_rank = tt_metal::distributed::multihost::Rank{recv_rank_ctx0},
            .distributed_context = distributed_ctx0};
        sockets_ctx0.emplace(recv_rank_ctx0, tt_metal::distributed::MeshSocket(mesh_device_, socket_config));
    }
    // Initialize sockets in context1 namespace
    if (std::find(ctx1_ranks.begin(), ctx1_ranks.end(), *distributed_ctx0->rank()) != ctx1_ranks.end()) {
        auto distributed_ctx1 = distributed_ctx0->create_sub_context(ctx1_ranks);
        if (*distributed_ctx1->rank() == recv_rank_ctx1) {
            for (const auto& sender_rank : sender_node_ranks_ctx1) {
                tt_metal::distributed::SocketConfig socket_config = {
                    .socket_connection_config = {socket_connection},
                    .socket_mem_config = socket_mem_config,
                    .sender_rank = tt_metal::distributed::multihost::Rank{sender_rank},
                    .receiver_rank = distributed_ctx1->rank(),
                    .distributed_context = distributed_ctx1};
                sockets_ctx1.emplace(sender_rank, tt_metal::distributed::MeshSocket(mesh_device_, socket_config));
            }
        } else if (
            std::find(sender_node_ranks_ctx1.begin(), sender_node_ranks_ctx1.end(), *distributed_ctx1->rank()) !=
            sender_node_ranks_ctx1.end()) {
            tt_metal::distributed::SocketConfig socket_config = {
                .socket_connection_config = {socket_connection},
                .socket_mem_config = socket_mem_config,
                .sender_rank = distributed_ctx1->rank(),
                .receiver_rank = tt_metal::distributed::multihost::Rank{recv_rank_ctx1},
                .distributed_context = distributed_ctx1};
            sockets_ctx1.emplace(recv_rank_ctx1, tt_metal::distributed::MeshSocket(mesh_device_, socket_config));
        }
    }
}

}  // namespace fabric_router_tests::multihost
}  // namespace tt::tt_fabric
