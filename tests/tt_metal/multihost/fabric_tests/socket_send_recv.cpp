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

using MultiHostSocketTestSplitT3K = MultiHostSocketTest<MeshDeviceSplit2x4Fixture>;
using MultiHostSocketTestDualT3K = MultiHostSocketTest<MeshDeviceDual2x4Fixture>;
using MultiHostSocketTestNanoExabox = MultiHostSocketTest<MeshDeviceNanoExaboxFixture>;

TEST_P(MultiHostSocketTestSplitT3K, SocketTests) { RunTest(); }

TEST_P(MultiHostSocketTestDualT3K, SocketTests) { RunTest(); }

TEST_P(MultiHostSocketTestNanoExabox, SocketTests) { RunTest(); }

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
    MultiHostSocketTestsNanoExabox,
    MultiHostSocketTestNanoExabox,
    ::testing::ValuesIn(generate_socket_test_configs(SystemConfig::NANO_EXABOX)),
    generate_multihost_socket_test_name<MultiHostSocketTestDualT3K::ParamType>);

}  // namespace fabric_router_tests::multihost
}  // namespace tt::tt_fabric
