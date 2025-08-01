// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/assert.h"
#include "risc_common.h"
#include "tt_metal/hw/inc/ethernet/tunneling.h"

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

constexpr uint32_t SOFT_RESET_ADDR = 0xFFB121B0;

template <uint32_t TXQ = 0>
inline void send_packet(uint32_t src_addr, uint32_t dst_addr, uint32_t n) {
    do {
        invalidate_l1_cache();
    } while (internal_::eth_txq_is_busy(TXQ));
    internal_::eth_send_packet(TXQ, src_addr >> 4, dst_addr >> 4, n);
}

inline void assert_connected_dm1_reset() {
    constexpr uint32_t k_ResetValue = 0x47000;
    internal_::eth_write_remote_reg(0, SOFT_RESET_ADDR, k_ResetValue);
}

inline void deassert_connected_dm1_reset() {
    constexpr uint32_t k_ResetValue = 0;
    internal_::eth_write_remote_reg(0, SOFT_RESET_ADDR, k_ResetValue);
}

void kernel_main() {
    auto test_ptr = reinterpret_cast<volatile uint32_t*>(0x20000);
    test_ptr[0] = 0x11112222;

    while (true) {
        __asm__ volatile("nop");
        __asm__ volatile("nop");
        __asm__ volatile("nop");
        __asm__ volatile("nop");
        __asm__ volatile("nop");
        test_ptr[0] = 0x11112222;
        __asm__ volatile("nop");
    }
}
