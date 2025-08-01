// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "debug/assert.h"
#include "tt_metal/hw/inc/ethernet/tunneling.h"

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

template <uint32_t TXQ>
inline void send_packet(uint32_t src_word_addr, uint32_t dst_word_addr, uint32_t n) {
    while (internal_::eth_txq_is_busy(TXQ)) {
    };
    internal_::eth_send_packet_bytes_unsafe(TXQ, src_word_addr, dst_word_addr, n);
}

constexpr uint32_t SOFT_RESET_ADDR = 0xFFB121B0;

void kernel_main() {}
