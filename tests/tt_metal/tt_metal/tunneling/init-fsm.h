// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "blackhole/eth_chan_noc_mapping.h"
#include "dataflow_api_addrgen.h"
#include "debug/assert.h"
#include "debug/dprint.h"
#include "ethernet/dataflow_api.h"
#include "dataflow_api.h"
#include "risc_common.h"
#include "fabric_lite.hpp"

void routing_init(volatile LiteFabricConfig* config_struct) {
    auto my_y = get_absolute_logical_y();
    int number_of_other_eth_chs = __builtin_popcount(config_struct->eth_chans_mask) - 1;
    ASSERT(number_of_other_eth_chs > 0);

    while (config_struct->current_state != InitState::READY) {
        invalidate_l1_cache();

        switch (config_struct->current_state) {
            case InitState::UNKNOWN: {
                // This should be initialized to a known value from the host or primary on the previous chip
                ASSERT(false);
                while (true) {
                    __asm__ volatile("nop");
                }
                break;
            }
            case InitState::ETH_INIT_FROM_HOST: {
                config_struct->current_state = InitState::ETH_INIT_LOCAL;
                break;
            }
            case InitState::ETH_INIT_LOCAL: {
                ASSERT(config_struct->is_primary);
                // Copy binaries to local ethernet cores
                // Memory maps are assumed to be the same so the same addresses can be used
                // Metal firmware is assumed to be running so a fake launch message is sent
                // Setup the state to be copied to the next cores
                InitState original_init_state = config_struct->initial_state;
                config_struct->current_state = InitState::ETH_HANDSHAKE_LOCAL;
                config_struct->initial_state = InitState::ETH_HANDSHAKE_LOCAL;
                config_struct->is_primary = false;

                uint32_t eth_mask = config_struct->eth_chans_mask;
                uint32_t y = 0;
                while (eth_mask > 0) {
                    if (y == my_y) {
                        eth_mask = eth_mask >> 1;
                        continue;
                    }
                    bool enabled = eth_mask & 1;
                    auto dest_xy = eth_chan_to_noc_xy[noc_index][y];

                    uint64_t dst_config_addr = get_noc_addr(dest_xy, reinterpret_cast<uint32_t>(config_struct));
                    uint64_t dst_binary_addr = get_noc_addr(dest_xy, config_struct->binary_addr);

                    // Configuration struct
                    noc_async_write(
                        reinterpret_cast<uint32_t>(config_struct), dst_config_addr, sizeof(LiteFabricConfig));

                    // Binary
                    noc_async_write(config_struct->binary_addr, dst_binary_addr, config_struct->binary_size);

                    noc_async_write_barrier();

                    eth_mask = eth_mask >> 1;
                }

                // Next state: Wait for local cores to be ready
                // Primary cores will stall for primary to be ready before incrementing local handshake
                // so this wont reset to 0 by accident
                config_struct->current_state = InitState::ETH_HANDSHAKE_LOCAL;
                config_struct->primary_local_handshake = 0;
                config_struct->neighbour_handshake = 0;
                config_struct->initial_state = original_init_state;
                config_struct->is_primary = true;
                break;
            }
            case InitState::ETH_HANDSHAKE_NEIGHBOUR: {
                break;
            }
            case InitState::ETH_INIT_NEIGHBOUR: {
                ASSERT(config_struct->is_primary);
                // Copy binary and metadata over ethernet
                // No need to change the is_primary flag. The new one will be the primary on the other chip
                // New primary will have to initialize it's local cores as well
                auto original_init_state = config_struct->initial_state;
                config_struct->current_state = InitState::ETH_INIT_LOCAL;
                config_struct->initial_state = InitState::ETH_INIT_LOCAL;

                config_struct->current_state = InitState::ETH_INIT_NEIGHBOUR;
                break;
            }
            case InitState::READY: {
                break;
            }
            case InitState::ETH_HANDSHAKE_LOCAL: {
                if (config_struct->is_primary) {
                    // Tell subordinate cores we are ready to receive their handshake
                    uint32_t subordinate_local_handshake_addr = reinterpret_cast<uint32_t>(config_struct) +
                                                                offsetof(LiteFabricConfig, subordinate_local_handshake);

                    uint32_t eth_mask = config_struct->eth_chans_mask;
                    uint32_t y = 0;
                    while (eth_mask > 0) {
                        if (y == my_y) {
                            eth_mask = eth_mask >> 1;
                            continue;
                        }
                        bool enabled = eth_mask & 1;
                        auto dest_xy = eth_chan_to_noc_xy[noc_index][y];

                        noc_semaphore_inc(get_noc_addr(dest_xy, subordinate_local_handshake_addr), 1);

                        eth_mask = eth_mask >> 1;
                    }

                    while (config_struct->primary_local_handshake != number_of_other_eth_chs) {
                        invalidate_l1_cache();
                        __asm__ volatile("nop");
                    }

                    // Wait for all other ethernet cores to be ready
                    // Next state: Initialize connected eth core
                    config_struct->current_state = InitState::ETH_INIT_NEIGHBOUR;
                } else {
                    // Wait for ready from primary and then tell primary we are ready
                    while (config_struct->primary_local_handshake != 1) {
                        invalidate_l1_cache();
                        __asm__ volatile("nop");
                    }

                    // Ack with primary
                    uint32_t primary_local_handshake_addr =
                        reinterpret_cast<uint32_t>(config_struct) + offsetof(LiteFabricConfig, primary_local_handshake);
                    noc_semaphore_inc(
                        get_noc_addr(
                            config_struct->primary_eth_core_x,
                            config_struct->primary_eth_core_y,
                            primary_local_handshake_addr),
                        1);

                    // Wait for all other ethernet cores to be ready
                    // Next state: Wait for remote / connected ethernet cores to handshake
                    config_struct->current_state = InitState::ETH_HANDSHAKE_NEIGHBOUR;
                }
                break;
            }
        }
    }
}
