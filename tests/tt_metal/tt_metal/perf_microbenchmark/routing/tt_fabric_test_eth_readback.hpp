// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/fabric.hpp>

// Include the test device and fixture headers
#include "tt_fabric_test_device_setup.hpp"
#include "tt_fabric_test_interfaces.hpp"

using FabricNodeId = tt::tt_fabric::FabricNodeId;
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using CoreCoord = tt::tt_metal::CoreCoord;
using TestDevice = tt::tt_fabric::fabric_tests::TestDevice;
using TestFixture = tt::tt_fabric::fabric_tests::TestFixture;
using RoutingDirection = tt::tt_fabric::RoutingDirection;

/**
 * @brief Result structure containing buffer data and location metadata for a single ethernet core
 */
struct EthCoreBufferResult {
    MeshCoordinate coord;                  // Device mesh coordinate
    FabricNodeId fabric_node_id;           // Fabric node ID
    CoreCoord eth_core;                    // Ethernet core coordinate
    tt::tt_fabric::chan_id_t eth_channel;  // Ethernet channel ID
    RoutingDirection direction;            // Direction this core serves
    uint32_t link_index;                   // Link index within direction
    std::vector<uint32_t> buffer_data;     // The actual buffer contents
};

/**
 * @brief Helper class for reading/writing buffers to ethernet cores
 *
 * This class encapsulates the common operations for reading and clearing
 * buffers on active ethernet cores across all test devices.
 */
class EthCoreBufferReadback {
public:
    /**
     * @brief Constructor
     * @param test_devices Reference to the map of test devices
     * @param fixture Reference to the test fixture
     */
    EthCoreBufferReadback(
        const std::unordered_map<MeshCoordinate, TestDevice>& test_devices,
        TestFixture& fixture);

    /**
     * @brief Clear buffer on all active ethernet cores
     * @param address The buffer address to clear
     * @param buffer_size The size of the buffer to clear
     */
    void clear_buffer(uint32_t address, size_t buffer_size);

    /**
     * @brief Read buffer from all active ethernet cores
     * @param address The buffer address to read from
     * @param buffer_size The size of the buffer to read
     * @return Vector of results containing buffer data and location metadata for each core
     */
    std::vector<EthCoreBufferResult> read_buffer(uint32_t address, size_t buffer_size);

private:
    const std::unordered_map<MeshCoordinate, TestDevice>& test_devices_;
    TestFixture& fixture_;
};
