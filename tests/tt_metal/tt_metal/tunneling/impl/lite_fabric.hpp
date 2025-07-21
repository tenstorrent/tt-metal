// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "assert.hpp"
#include "fabric_edm_types.hpp"

#if defined(KERNEL_BUILD)

#include "debug/waypoint.h"
#include "debug/assert.h"
#include "eth_chan_noc_mapping.h" // eth_chan_to_noc_xy
#include "tt_metal/api/tt-metalium/fabric_edm_types.hpp" // WorkerXY

#endif

enum class LiteFabricState : uint32_t {
    // Unknown state
    UNKNOWN = 0,
    // Initialize other ethernet cores on the same chip which is MMIO capable
    MMIO_ETH_INIT_NEIGHBOUR,
    // Do handshake with other ethernet cores on the same chip
    LOCAL_HANDSHAKE,
    // Initialize other ethernet cores on the same chip which is not MMIO capable
    NON_MMIO_ETH_INIT_LOCAL_ETHS,
    // Do handshake with the neighbor chip
    NEIGHBOUR_HANDSHAKE,
    // Done handhsake
    DONE_HANDSHAKE,
    // Ready for packets
    READY_FOR_PACKETS,
    // Terminated
    TERMINATED,
};

struct LiteFabricConfig {
    // Original state of the lite fabric.
    volatile LiteFabricState original_state = LiteFabricState::UNKNOWN;
    // Current state of the lite fabric.
    volatile LiteFabricState current_state = LiteFabricState::UNKNOWN;
    // Address of the lite fabric binary.
    volatile uint32_t binary_address = 0;
    // Size of the lite fabric binary.
    volatile uint32_t binary_size_bytes = 0;
    // Ethernet channels mask. 1 means to initialize. 0 means to skip.
    volatile uint32_t eth_chans_mask = 0;
    // Number of local ethernet cores. Handshake is complete when the number of handshake acks are equal
    // to this value.
    volatile uint32_t num_local_eths = 0;
    // X coordinate of the primary ethernet core. Subordinate lite fabric cores will send their handshake to this core
    volatile uint8_t primary_eth_core_x = 0;
    // Y coordinate of the primary ethernet core. Subordinate lite fabric cores will send their handshake to this core
    volatile uint8_t primary_eth_core_y = 0;
    // Count of local neighbour handshake acks.
    volatile uint32_t local_neighbour_handshake = 0;
    // Count of remote neighbour handshake acks.
    volatile uint32_t remote_neighbour_handshake = 0;
    // Non zero termination signal will terminate the lite fabric.
    volatile uint32_t termination_signal = 0;
    // Non zero if the lite fabric is on a MMIO capable chip.
    volatile bool on_mmio_chip = false;

    // Padding
    uint32_t padding[5];
};

static_assert(sizeof(struct LiteFabricConfig) % 16 == 0, "LiteFabricConfig must be 16 bytes aligned");

tt::tt_fabric::WorkerXY get_ethernet_core(uint32_t eth_chan) {
#if defined(KERNEL_BUILD)
    return tt::tt_fabric::WorkerXY(eth_chan_to_noc_xy[eth_chan].x, eth_chan_to_noc_xy[eth_chan].y);
#else
    TT_THROW("get_ethernet_core(uint32_t) cannot be called from the host");
#endif
}

void init(LiteFabricConfig* config) {
#if defined(KERNEL_BUILD)
    // State machine for initializing lite fabric on active ethernet cores
    while (config->current_state != LiteFabricState::DONE_HANDSHAKE) {
        switch (config->current_state) {
            case LiteFabricState::UNKNOWN:
            {
                WAYPOINT("LFI0");
                break;
            }
            case LiteFabricState::MMIO_ETH_INIT_NEIGHBOUR:
            {
                WAYPOINT("LFI1");
                break;
            }
            case LiteFabricState::LOCAL_HANDSHAKE:
            {
                WAYPOINT("LFI2");
                break;
            }
            case LiteFabricState::NON_MMIO_ETH_INIT_LOCAL_ETHS:
            {
                WAYPOINT("LFI3");
                break;
            }
                break;
            case LiteFabricState::NEIGHBOUR_HANDSHAKE:
            {
                WAYPOINT("LFI4");
                break;
            }
            case LiteFabricState::DONE_HANDSHAKE:
            {
                WAYPOINT("LFI5");
                break;
            }
            case LiteFabricState::READY_FOR_PACKETS:
            {
                WAYPOINT("LFI6");
                break;
            }
            case LiteFabricState::TERMINATED:
            {
                WAYPOINT("LFI7");
                break;
            }
        }
    }
#else
    TT_THROW("init(LiteFabricConfig*) cannot be called from the host");
#endif
}
