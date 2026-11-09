// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttml::ops::distributed {

// How the global sequence is laid out across the CP ring.
// - Contiguous: rank d holds tokens [d*S_local, (d+1)*S_local).
// - Striped (causal load balancing): global token p lives on rank p mod N as local row p div N.
//   Every rank runs one causal block of the same shape at every ring step: on its query for K/V
//   from ranks <= itself, and on its query shifted up one row (strict causal) for later ranks.
//   The caller (data loader, RoPE positions) must lay tokens out this way; ring attention only
//   schedules the work.
enum class RingAttentionLayout : uint8_t { Contiguous = 0, Striped = 1 };

// Global sequence position of the token at position `device_order_pos` of the striped layout, where
// device order is the rank-major concatenation of the local sequences (what sharding the sequence dim
// over the CP axis produces): rank d's local row l, at device_order_pos = d * (seq_len / cp_size) + l,
// is global token l * cp_size + d.
inline uint32_t striped_global_position(uint32_t device_order_pos, uint32_t seq_len, uint32_t cp_size) {
    const uint32_t local_len = seq_len / cp_size;
    return (device_order_pos % local_len) * cp_size + device_order_pos / local_len;
}

}  // namespace ttml::ops::distributed
