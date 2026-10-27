// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffer constants for device (kernel/firmware) code.
//
// This is the device-side counterpart of the host header tt-metalium/circular_buffer_constants.h. The two files
// are deliberately separate: one is compiled by the host compiler, the other by the RISC-V kernel compiler, and
// neither may include the other.
//
// Device compilation context:
//   - ARCH_WORMHOLE is defined ONLY when compiling for Wormhole
//   - Wormhole has fewer CBs due to limited TRISC memory (2KB)
//   - Blackhole supports the full CB count
//
// TODO: This is TEMPORARY code structure - eventually will be replaced by Dataflow Buffers (DFBs)

#if defined(ARCH_WORMHOLE)
// Wormhole (limited by 2KB TRISC memory)
inline constexpr std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
#else
inline constexpr std::uint32_t NUM_CIRCULAR_BUFFERS = 64;
#endif
inline constexpr std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4;
inline constexpr std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;
inline constexpr std::uint32_t CIRCULAR_BUFFER_COMPUTE_WORD_SIZE = 16;
inline constexpr std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4;
