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

// The real silicon counts. These are the ENFORCED cap (via
// hal::get_arch_num_circular_buffers) and must stay independent of the array sizing below,
// which an emule debug build may raise.
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS_WORMHOLE = 32;
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS_SILICON_MAX = 64;

#if defined(EMULE_CB_CEILING)
// Emule build only: CBs are host memory, so a debug ceiling can size arrays past any arch.
// Build-WIDE define (cmake/project_options.cmake) — never per-target, or CircularBufferConfig's
// sizeof would differ between translation units. Applies to Wormhole too: sizing, not the cap.
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = EMULE_CB_CEILING;
#elif defined(ARCH_WORMHOLE)
// Device compilation for Wormhole (limited by 2KB TRISC memory)
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = NUM_CIRCULAR_BUFFERS_WORMHOLE;
#else
// Blackhole device and HOST compilation (uses max for array sizing)
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = NUM_CIRCULAR_BUFFERS_SILICON_MAX;
#endif
constexpr static std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4;
constexpr static std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_WORD_SIZE = 16;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4;
