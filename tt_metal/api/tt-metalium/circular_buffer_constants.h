// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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
