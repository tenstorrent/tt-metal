// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt_stl/assert.hpp>

namespace ttml::core::distributed {

// How the sequence is split over the CP axis. With N CP devices, Contiguous gives device d global block d of N;
// ZigZag gives it blocks d and 2N-1-d of 2N, which balances causal attention across the ring.
enum class CpSharding : uint8_t { Contiguous = 0, ZigZag = 1 };

// Global position of the token at `device_order_pos` when the sequence is laid out over the CP devices in order.
inline uint32_t cp_global_position(CpSharding sharding, uint32_t device_order_pos, uint32_t seq_len, uint32_t cp_size) {
    switch (sharding) {
        case CpSharding::Contiguous: return device_order_pos;
        case CpSharding::ZigZag: {
            const uint32_t block = seq_len / (2U * cp_size);
            const uint32_t slot = device_order_pos / block;
            const uint32_t rank = slot / 2U;
            const uint32_t global_block = (slot % 2U == 0U) ? rank : 2U * cp_size - 1U - rank;
            return global_block * block + device_order_pos % block;
        }
    }
    TT_THROW("Unknown CP sharding {}", static_cast<int>(sharding));
}

// cp_global_position is a permutation of the sequence only for lengths the sharding can split evenly.
inline void validate_cp_sequence_length(CpSharding sharding, uint32_t seq_len, uint32_t cp_size) {
    switch (sharding) {
        case CpSharding::Contiguous: return;
        case CpSharding::ZigZag:
            TT_FATAL(
                seq_len % (2U * cp_size) == 0U,
                "Zig-zag CP sharding needs sequence_length ({}) divisible by 2 * cp_size ({})",
                seq_len,
                2U * cp_size);
            return;
    }
    TT_THROW("Unknown CP sharding {}", static_cast<int>(sharding));
}

}  // namespace ttml::core::distributed
