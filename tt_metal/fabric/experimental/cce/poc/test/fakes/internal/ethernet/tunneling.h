// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace internal_ {

inline bool eth_txq_is_busy(uint32_t) { return false; }

// Loopback: copies SRAM to SRAM on the same CCE.
void eth_send_packet_fake(uint32_t src_word_addr, uint32_t dest_word_addr, uint32_t num_words);

template <bool ctx_switch = true>
void eth_send_packet(uint32_t /*q_num*/, uint32_t src_word_addr, uint32_t dest_word_addr, uint32_t num_words) {
    eth_send_packet_fake(src_word_addr, dest_word_addr, num_words);
}

}  // namespace internal_
