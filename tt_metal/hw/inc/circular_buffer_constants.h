// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
constexpr static std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4;
constexpr static std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_WORD_SIZE = 16;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4;
