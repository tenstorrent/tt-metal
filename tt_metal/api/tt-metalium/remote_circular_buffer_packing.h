// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/circular_buffer_constants.h>

// Pack/unpack helpers for RemoteSenderCBInterface::num_receivers_and_remote_pages_sent_ptr.
// The REMOTE_CB_PACKED_* constants they use live in circular_buffer_constants.h.
//
// Declared here (host + device shared) so host code that composes a receiver-CB config
// (e.g. GlobalCircularBuffer's DRAM-sender state block) can pack the field with the same
// scheme the kernel unpacks, without pulling in the kernel-only
// circular_buffer_interface.h header.

inline constexpr std::uint32_t remote_cb_num_receivers(std::uint32_t packed) {
    return packed >> REMOTE_CB_PACKED_COUNT_SHIFT;
}
inline constexpr std::uint32_t remote_cb_remote_pages_sent_ptr(std::uint32_t packed) {
    return packed & REMOTE_CB_PACKED_ADDR_MASK;
}
inline constexpr std::uint32_t remote_cb_pack(std::uint32_t num_receivers, std::uint32_t remote_pages_sent_ptr) {
    return (num_receivers << REMOTE_CB_PACKED_COUNT_SHIFT) | (remote_pages_sent_ptr & REMOTE_CB_PACKED_ADDR_MASK);
}
