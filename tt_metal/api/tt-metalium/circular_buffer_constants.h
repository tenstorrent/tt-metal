// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// IMPORTANT: This file is included by BOTH host compilation AND device JIT compilation
// Host compilation context (e.g., CircularBufferConfig, program building):
//   - ARCH_WORMHOLE is NEVER defined
//   - NUM_CIRCULAR_BUFFERS = 64
//   - Host-side arrays/vectors are sized for 64 CBs (maximum across all architectures)
//
// Device compilation context (firmware kernels):
//   - ARCH_WORMHOLE is defined ONLY when compiling for Wormhole devices
//   - NUM_CIRCULAR_BUFFERS = 32 on Wormhole, 64 on Blackhole
//   - Device firmware uses the correct per-architecture limit
//
// Why this works safely:
//   - Host allocates space for up to 64 CBs in all data structures
//   - Runtime validation (via hal.get_arch_num_circular_buffers()) prevents using
//     CB indices >= 32 on Wormhole, so extra host slots remain unused
//   - Device firmware only processes the CBs valid for that architecture
//
// For NEW CODE:
//   DO NOT USE NUM_CIRCULAR_BUFFERS to get the actual device limit.
//   USE: tt::tt_metal::hal::get_arch_num_circular_buffers() instead (See tt_metal/api/tt-metalium/hal.hpp)
//
// Why this split exists:
//   - Host must allocate space for worst-case (64 CBs) in data structures
//   - Runtime validation prevents using invalid CB indices on each architecture
//   - Device firmware only processes CBs valid for that specific architecture
//
// TODO: This is TEMPORARY code structure : eventually will be replaced by Dataflow Buffers (DFBs)

// Bit width of CB mask (uint64_t)
constexpr static std::uint32_t CB_MASK_WIDTH = 64;

#if defined(ARCH_WORMHOLE)
// Device compilation for Wormhole: 32 CBs (limited by 2KB TRISC memory)
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
#else
// Blackhole (64 CBs with 4KB TRISC) and HOST (uses max for array sizing)
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 64;
#endif
constexpr static std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4;
constexpr static std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_WORD_SIZE = 16;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4;

static_assert(NUM_CIRCULAR_BUFFERS <= CB_MASK_WIDTH, "Maximum CB count exceeds mask width");
