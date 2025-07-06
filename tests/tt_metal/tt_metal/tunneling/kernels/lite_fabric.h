// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tunneling {

enum LiteFabricInitState : uint8_t {
    MMIO_ETH_INIT_NEIGHBOUR = 0,
    LOCAL_HANDSHAKE = 1,
    NON_MMIO_ETH_INIT_LOCAL_ETHS = 2,
    NEIGHBOUR_HANDSHAKE = 3,
    DONE = 4,
    UNKNOWN = 5,
};

// need to put this at a 16B aligned address
struct lite_fabric_config_t {
    volatile uint32_t binary_address = 0;     // only used by eths that set up kernels
    volatile uint32_t binary_size_bytes = 0;  // only used by eths that set up kernels
    volatile uint32_t eth_chans_mask = 0;
    volatile uint32_t num_local_eths = 0;
    volatile uint32_t primary_local_handshake = 0;      // where subordinate eths will signal
    volatile uint32_t subordinate_local_handshake = 0;  // where primary eth will signal to subordinates
    volatile uint8_t primary_eth_core_x = 0;
    volatile uint8_t primary_eth_core_y = 0;
    volatile LiteFabricInitState init_state = LiteFabricInitState::UNKNOWN;
    volatile uint8_t multi_eth_cores_setup = 1;  // test mode only
    volatile uint32_t pad0 = 0;
    volatile uint32_t local_neighbour_handshake = 0;  // this needs to be 16B aligned
    volatile uint32_t pad1[3] = {0};
    volatile uint32_t remote_neighbour_handshake = 0;  // this needs to be 16B aligned
    volatile uint8_t pad_final[12] = {0};              // padding to make size 64B
} __attribute__((packed));

}  // namespace tunneling
