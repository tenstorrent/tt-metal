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
//   - Blackhole reserves the last two counter pairs for DM/compute scratch synchronization
//
// Why this works safely:
//   - Host allocates space for the maximum CB count in all data structures
//   - Runtime validation (via hal.get_num_dataflow_buffers()) prevents using
//     CB indices beyond the device's actual limit
//   - Device firmware only processes CBs valid for that architecture
//
// For NEW CODE:
//   DO NOT USE NUM_CIRCULAR_BUFFERS to get the actual device limit
//   USE: tt::tt_metal::hal::get_num_dataflow_buffers() instead (See tt_metal/api/tt-metalium/hal.hpp)
//
// TODO: This is TEMPORARY code structure - eventually will be replaced by Dataflow Buffers (DFBs)

// Blackhole stream-counter resources: CBs own [0, 62), scratch sync owns [62, 64).
inline constexpr std::uint32_t BLACKHOLE_NUM_CB_COUNTERS = 64;
inline constexpr std::uint32_t BLACKHOLE_NUM_SCRATCH_SYNC_CHANNELS = 2;
inline constexpr std::uint32_t BLACKHOLE_NUM_CIRCULAR_BUFFERS =
    BLACKHOLE_NUM_CB_COUNTERS - BLACKHOLE_NUM_SCRATCH_SYNC_CHANNELS;

#if defined(ARCH_WORMHOLE)
// Device compilation for Wormhole (limited by 2KB TRISC memory)
inline constexpr std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
#else
// Blackhole device and HOST compilation (uses max for array sizing)
inline constexpr std::uint32_t NUM_CIRCULAR_BUFFERS = 64;
#endif
// Device-side configuration indexing follows the usable limit reported by the host HAL.
// Keep NUM_CIRCULAR_BUFFERS unchanged for storage and initialization of all counter pairs.
#if defined(ARCH_BLACKHOLE)
inline constexpr std::uint32_t NUM_USABLE_CIRCULAR_BUFFERS = BLACKHOLE_NUM_CIRCULAR_BUFFERS;
#else
inline constexpr std::uint32_t NUM_USABLE_CIRCULAR_BUFFERS = NUM_CIRCULAR_BUFFERS;
#endif

inline constexpr std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4;
inline constexpr std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;
inline constexpr std::uint32_t CIRCULAR_BUFFER_COMPUTE_WORD_SIZE = 16;
inline constexpr std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4;
