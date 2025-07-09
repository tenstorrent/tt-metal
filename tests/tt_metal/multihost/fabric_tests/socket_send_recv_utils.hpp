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

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric.hpp>

namespace tt::tt_fabric {
namespace fabric_router_tests::multihost {

namespace multihost_utils {

// System Types currently supported for testing
enum class SystemConfig { SPLIT_T3K, DUAL_T3K, NANO_EXABOX };

// Socket Test Variants
enum class TestVariant { SINGLE_CONN_BWD, SINGLE_CONN_FWD, MULTI_CONN_FWD, MULTI_CONN_BIDIR };

std::string get_system_config_name(SystemConfig system_config) {
    switch (system_config) {
        case SystemConfig::SPLIT_T3K: return "SplitT3K";
        case SystemConfig::DUAL_T3K: return "DualT3K";
        case SystemConfig::NANO_EXABOX: return "NanoExabox";
        default: return "Unknown";
    }
}

std::string get_test_variant_name(TestVariant variant) {
    switch (variant) {
        case TestVariant::SINGLE_CONN_BWD: return "MultiMeshSingleConnectionBwd";
        case TestVariant::SINGLE_CONN_FWD: return "MultiMeshSingleConnectionFwd";
        case TestVariant::MULTI_CONN_FWD: return "MultiMeshMultiConnectionFwd";
        case TestVariant::MULTI_CONN_BIDIR: return "MultiConnectionBidirectional";
        default: return "Unknown";
    }
}

// Configuration for Multi-Host Socket Tests
struct SocketTestConfig {
    uint32_t socket_fifo_size;
    uint32_t socket_page_size;
    uint32_t data_size;
    TestVariant variant;
    SystemConfig system_config;
};

void test_multi_mesh_single_conn_bwd(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config);

void test_multi_mesh_single_conn_fwd(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config);

void test_multi_mesh_multi_conn_fwd(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config);

void test_multi_mesh_multi_conn_bidirectional(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config);

}  // namespace multihost_utils

}  // namespace fabric_router_tests::multihost
}  // namespace tt::tt_fabric
