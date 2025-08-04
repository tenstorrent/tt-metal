// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/fabric_edm_types.hpp>
#include "lite_fabric_constants.hpp"

namespace lite_fabric {

/*

Initialization process for Lite Fabric

    1. Host writes the lite fabric kernel to an arbitrary active ethernet core on MMIO capable chips. This
    is designated as the Primary core with an initial state of ETH_INIT_LOCAL. This core will launch
    lite fabric kernels on other active ethernet cores on the same chip with an initial state of
ETH_INIT_LOCAL_HANDSHAKE.

    2. The primary core will stall for the ETH_INIT_LOCAL_HANDSHAKE cores to be ready

    3. Primary core transitions state to ETH_INIT_NEIGHBOUR. It will launch a primary lite fabric kernel on the eth
device.

    4. Subordinate core transitions state to ETH_INIT_NEIGHBOUR_HANDSHAKE

    5. The primary lite fabric kernel on the eth device will launch lite fabric kernels on other active ethernet cores
on the eth device with an initial state of ETH_INIT_LOCAL_HANDSHAKE

*/

enum class InitState : uint16_t {
    // Unknown initial state
    UNKNOWN = 0,
    // Indicates that this is written directly from host
    ETH_INIT_FROM_HOST,
    // Write kernel to local ethernet cores and wait for ack
    ETH_INIT_LOCAL,
    // Wait for ack from connected ethernet core
    ETH_HANDSHAKE_NEIGHBOUR,
    // Write primary kernel to connected ethernet core and wait for ack
    ETH_INIT_NEIGHBOUR,
    // Wait for ack from local ethernet cores
    ETH_HANDSHAKE_LOCAL,
    // Ready for traffic
    READY,
};

struct LiteFabricConfig {
    // Starting address of the Lite Fabric binary to be copied locally and to the neighbour.
    volatile uint32_t binary_addr = 0;

    // Size of the Lite Fabric binary.
    volatile uint32_t binary_size = 0;

    // Bit N is 1 if channel N is an active ethernet core. Relies on eth_chan_to_noc_xy to
    // get the ethernet core coordinate.
    volatile uint32_t eth_chans_mask = 0;

    unsigned char padding0[4];

    // Subordinate cores on the same chip increment this value when they are ready. The primary core
    // will stall until this value shows all eth cores are ready.
    volatile uint32_t primary_local_handshake = 0;

    unsigned char padding1[12];

    // Becomes 1 when the neighbour is ready
    volatile uint32_t neighbour_handshake = 0;

    // X coordinate of the primary eth core on this chip
    volatile uint8_t primary_eth_core_x = 0;

    // Y coordinate of the primary eth core on this chip
    volatile uint8_t primary_eth_core_y = 0;

    // This is the local primary core
    volatile uint16_t is_primary = false;

    // This is on the MMIO
    volatile uint16_t is_mmio = false;

    volatile InitState initial_state = InitState::UNKNOWN;

    volatile InitState current_state = InitState::UNKNOWN;

    // Set to 1 to enable routing
    volatile uint32_t routing_enabled = 1;

    unsigned char padding2[14];
} __attribute__((packed));

struct LiteFabricMemoryMap {
    lite_fabric::LiteFabricConfig config;
    tt::tt_fabric::EDMChannelWorkerLocationInfo sender_location_info;
    uint32_t sender_flow_control_semaphore;
    unsigned char padding0[12];
    uint32_t sender_connection_live_semaphore;
    unsigned char padding1[12];
    uint32_t worker_semaphore;
    unsigned char padding2[12];
    unsigned char sender_channel_buffer[CHANNEL_BUFFER_SLOTS * CHANNEL_BUFFER_SIZE];
    unsigned char receiver_channel_buffer[CHANNEL_BUFFER_SLOTS * CHANNEL_BUFFER_SIZE];
};

static_assert(sizeof(LiteFabricConfig) % 16 == 0);
static_assert(offsetof(LiteFabricConfig, primary_local_handshake) % 16 == 0);
static_assert(offsetof(LiteFabricConfig, neighbour_handshake) % 16 == 0);

}  // namespace lite_fabric
