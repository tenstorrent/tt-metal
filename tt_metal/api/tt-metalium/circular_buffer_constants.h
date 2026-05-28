// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// IMPORTANT: This file is included by BOTH host compilation AND device JIT compilation
//
// Host compilation context:
//   - ARCH_WORMHOLE is NEVER defined
//   - NUM_CIRCULAR_BUFFERS uses the maximum across all architectures
//   - Host-side arrays/vectors are sized for this maximum
//
// Device compilation context:
//   - ARCH_WORMHOLE is defined ONLY when compiling for Wormhole
//   - Wormhole has fewer CBs due to limited TRISC memory (2KB)
//   - Blackhole supports the full CB count
//
// Why this works safely:
//   - Host allocates space for the maximum CB count in all data structures
//   - Runtime validation (via hal.get_arch_num_circular_buffers()) prevents using
//     CB indices beyond the device's actual limit
//   - Device firmware only processes CBs valid for that architecture
//
// For NEW CODE:
//   DO NOT USE NUM_CIRCULAR_BUFFERS to get the actual device limit
//   USE: tt::tt_metal::hal::get_arch_num_circular_buffers() instead (See tt_metal/api/tt-metalium/hal.hpp)
//
// TODO: This is TEMPORARY code structure - eventually will be replaced by Dataflow Buffers (DFBs)

#if defined(ARCH_WORMHOLE)
// Device compilation for Wormhole (limited by 2KB TRISC memory)
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
#else
// Blackhole device and HOST compilation (uses max for array sizing)
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 64;
#endif
constexpr static std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4;
constexpr static std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_WORD_SIZE = 16;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4;

// L1 addresses fit in 24 bits (< 2 MB) and num_receivers fits in 8 bits, so the two
// can share a 32-bit slot in RemoteSenderCBInterface::num_receivers_and_remote_pages_sent_ptr.
// Declared here (host + device shared) so host code that composes the receiver-CB
// config (e.g. GlobalCircularBuffer's DRAM-sender state block) can pack the field
// without including the kernel-only circular_buffer_interface.h header.
constexpr static std::uint32_t REMOTE_CB_PACKED_ADDR_MASK = 0x00FFFFFFu;
constexpr static std::uint32_t REMOTE_CB_PACKED_COUNT_SHIFT = 24;

inline constexpr std::uint32_t remote_cb_num_receivers(std::uint32_t packed) {
    return packed >> REMOTE_CB_PACKED_COUNT_SHIFT;
}
inline constexpr std::uint32_t remote_cb_remote_pages_sent_ptr(std::uint32_t packed) {
    return packed & REMOTE_CB_PACKED_ADDR_MASK;
}
inline constexpr std::uint32_t remote_cb_pack(std::uint32_t num_receivers, std::uint32_t remote_pages_sent_ptr) {
    return (num_receivers << REMOTE_CB_PACKED_COUNT_SHIFT) | (remote_pages_sent_ptr & REMOTE_CB_PACKED_ADDR_MASK);
}
