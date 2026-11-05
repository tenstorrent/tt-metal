// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-only header. Device (kernel/firmware) code must use api/circular_buffer_constants.h from
// tt_metal/hw/inc instead: the two sides are compiled by different compilers and must not share this file.
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#error "tt-metalium/circular_buffer_constants.h is host-only; include \"api/circular_buffer_constants.h\" in device code"
#endif

#include <cstdint>

// Host compilation context:
//   - ARCH_WORMHOLE is NEVER defined
//   - NUM_CIRCULAR_BUFFERS is the maximum across all architectures
//   - Host-side arrays/vectors are sized for this maximum
//
// Why this works safely:
//   - Host allocates space for the maximum CB count in all data structures
//   - Runtime validation (via hal.get_num_dataflow_buffers()) prevents using
//     CB indices beyond the device's actual limit
//
// For NEW CODE:
//   DO NOT USE NUM_CIRCULAR_BUFFERS to get the actual device limit
//   USE: tt::tt_metal::hal::get_num_dataflow_buffers() instead (See tt_metal/api/tt-metalium/hal.hpp)
//
// TODO: This is TEMPORARY code structure - eventually will be replaced by Dataflow Buffers (DFBs)

inline constexpr std::uint32_t NUM_CIRCULAR_BUFFERS = 64;
inline constexpr std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4;
inline constexpr std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;
inline constexpr std::uint32_t CIRCULAR_BUFFER_COMPUTE_WORD_SIZE = 16;
inline constexpr std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4;
