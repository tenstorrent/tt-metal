// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <chrono>
#include <gtest/gtest.h>
#include <stdint.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace tt::tt_fabric::fabric_router_tests::multihost::multihost_utils {

// System Types currently supported for testing
enum class SystemConfig { SPLIT_T3K, DUAL_T3K, NANO_EXABOX, EXABOX, SPLIT_GALAXY };

// Socket Test Variants
enum class TestVariant { SINGLE_CONN_BWD, SINGLE_CONN_FWD, MULTI_CONN_FWD, MULTI_CONN_BIDIR };

std::string get_system_config_name(SystemConfig system_config);

std::string get_test_variant_name(TestVariant variant);

// Core socket send/recv test function
bool test_socket_send_recv(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device_,
    tt::tt_metal::distributed::MeshSocket& socket,
    uint32_t data_size,
    uint32_t page_size,
    uint32_t num_txns = 20,
    std::optional<std::mt19937> gen = std::nullopt);

// Configuration for Multi-Host Socket Tests
struct SocketTestConfig {
    uint32_t socket_fifo_size;
    uint32_t socket_page_size;
    uint32_t data_size;
    TestVariant variant;
    SystemConfig system_config;
};

void test_multi_mesh_single_conn_bwd(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config);

void test_multi_mesh_single_conn_fwd(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config);

void test_multi_mesh_multi_conn_fwd(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config);

void test_multi_mesh_multi_conn_bidirectional(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config);

uint32_t sync_seed_across_ranks(tt_fabric::MeshId sender_mesh_id, tt_fabric::MeshId recv_mesh_id);

}  // namespace tt::tt_fabric::fabric_router_tests::multihost::multihost_utils
