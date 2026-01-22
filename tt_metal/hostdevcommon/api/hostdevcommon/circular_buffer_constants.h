// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Circular buffer constants shared between host and device code.

#pragma once

#include <cstdint>

// These constants are intentionally in the global namespace to match the original API.
// The original file did not use any namespace.
constexpr std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
constexpr std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4;
constexpr std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;
constexpr std::uint32_t CIRCULAR_BUFFER_COMPUTE_WORD_SIZE = 16;
constexpr std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4;
