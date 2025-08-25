// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Basic handshake (only supports tunnel depth of 1)
// Expects the host to write the lite fabric kernel to the MMIO device
// 1. Each MMIO kernel Mi copies itself to the neighbour (over ethernet) Ni
// 2. Ni will wait for go signal from Mi
// 3. Mi sends first handshake to Ni
// 4. Ni returns ack and is ready. Now Mi is also ready.

#pragma once

#include <cstdint>
#include <stddef.h>
#include "blackhole/eth_chan_noc_mapping.h"
#include "dataflow_api_addrgen.h"
#include "ethernet/dataflow_api.h"
#include "dataflow_api.h"
#include "ethernet/tunneling.h"
#include "tt_metal/lite_fabric/hw/inc/lf_dev_mem_map.hpp"
#include "risc_common.h"
#include "host_interface.hpp"
#include "tt_metal/lite_fabric/hw/inc/risc_interface.hpp"

namespace lite_fabric {

static_assert(sizeof(uint32_t) == sizeof(uintptr_t));

void wait_val(uint32_t addr, uint32_t val) {
    do {
        invalidate_l1_cache();
    } while (reinterpret_cast<volatile uint32_t*>(addr)[0] != val);
}

void routing_init(volatile lite_fabric::FabricLiteConfig* config_struct) {
    invalidate_l1_cache();
    // This value should not be used. It comes from metal.
    // auto my_y = get_absolute_logical_y();
    int number_of_other_eth_chs = __builtin_popcount(config_struct->eth_chans_mask) - 1;
    ASSERT(number_of_other_eth_chs > 0);

    // Send the binary over ethernet to the connected core
    const auto eth_send_binary = [=]() {
        internal_::eth_send_packet<false>(
            0, LITE_FABRIC_DATA_START >> 4, LITE_FABRIC_DATA_START >> 4, LITE_FABRIC_DATA_SIZE >> 4);
        internal_::eth_send_packet<false>(
            0, config_struct->binary_addr >> 4, config_struct->binary_addr >> 4, config_struct->binary_size >> 4);
    };

    const auto eth_send_config = [=]() {
        internal_::eth_send_packet<false>(
            0,
            (uintptr_t)config_struct >> 4,
            (uintptr_t)config_struct >> 4,
            sizeof(lite_fabric::FabricLiteConfig) >> 4);
    };

    auto original_init_state = config_struct->initial_state;
    bool is_mmio = config_struct->is_mmio;
    bool is_primary = config_struct->is_mmio;
    while (config_struct->current_state != lite_fabric::InitState::READY) {
        invalidate_l1_cache();

        switch (config_struct->current_state) {
            case lite_fabric::InitState::UNKNOWN: {
                do {
                    invalidate_l1_cache();
                } while (config_struct->current_state == lite_fabric::InitState::UNKNOWN);
                break;
            }
            case lite_fabric::InitState::ETH_INIT_FROM_HOST: {
                break;
            }
            case lite_fabric::InitState::ETH_INIT_LOCAL: {
                break;
            }
            case lite_fabric::InitState::ETH_HANDSHAKE_NEIGHBOUR: {
                auto handshake_addr = (uintptr_t)&config_struct->neighbour_handshake;
                auto local_handshake_addr = (uintptr_t)&config_struct->primary_local_handshake;

                if (is_mmio) {
                    wait_val(handshake_addr, 1);
                    // Safe to modify config_struct now
                    config_struct->primary_local_handshake = 2;
                    internal_::eth_send_packet(0, local_handshake_addr >> 4, handshake_addr >> 4, 1);

                    // Wait for ack
                    wait_val(handshake_addr, 3);
                } else {
                    // Send first signal to mmio to indicate we have started
                    config_struct->primary_local_handshake = 1;
                    internal_::eth_send_packet(0, local_handshake_addr >> 4, handshake_addr >> 4, 1);

                    // wait for signal from mmio
                    wait_val(handshake_addr, 2);

                    // send ack to mmio
                    config_struct->primary_local_handshake = 3;
                    internal_::eth_send_packet(0, local_handshake_addr >> 4, handshake_addr >> 4, 1);
                }
                config_struct->current_state = lite_fabric::InitState::READY;
                break;
            }
            case lite_fabric::InitState::ETH_INIT_NEIGHBOUR: {
                ASSERT(is_primary);
                ASSERT(is_mmio);
                config_struct->is_primary = false;
                config_struct->is_mmio = false;
                config_struct->routing_enabled = 1;
                config_struct->current_state = lite_fabric::InitState::ETH_HANDSHAKE_NEIGHBOUR;
                config_struct->initial_state = lite_fabric::InitState::ETH_HANDSHAKE_NEIGHBOUR;
                ConnectedRisc1Interface::assert_connected_dm1_reset();
                ConnectedRisc1Interface::set_pc(LITE_FABRIC_TEXT_START);
                eth_send_config();
                eth_send_binary();
                ConnectedRisc1Interface::deassert_connected_dm1_reset();
                break;
            }
            case lite_fabric::InitState::ETH_HANDSHAKE_LOCAL: {
                break;
            }
            default: {
                ASSERT(false);
                while (true) {
                };
            }
        }
    }
}

}  // namespace lite_fabric
