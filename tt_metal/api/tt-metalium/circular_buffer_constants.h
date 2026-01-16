// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// TODO: This file is used by both host and device.
// TODO: This should be in the tt::tt_metal namespace.

// Circular buffer limits per architecture
constexpr static std::uint32_t CB_MASK_WIDTH = 64;              // For 64-bit mask operations
constexpr static std::uint32_t MAX_CIRCULAR_BUFFERS = 64;       // Maximum (Blackhole)
constexpr static std::uint32_t WORMHOLE_CIRCULAR_BUFFERS = 32;  // Wormhole limit (TRISC 2KB)

#if defined(ARCH_WORMHOLE)
// Device compilation for Wormhole
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = WORMHOLE_CIRCULAR_BUFFERS;
#else
// Blackhole and host (for array sizing)
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = MAX_CIRCULAR_BUFFERS;
#endif
constexpr static std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4;
constexpr static std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_WORD_SIZE = 16;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4;

static_assert(MAX_CIRCULAR_BUFFERS <= CB_MASK_WIDTH, "Maximum CB count exceeds mask width");
