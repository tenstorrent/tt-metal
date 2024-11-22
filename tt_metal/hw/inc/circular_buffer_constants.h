// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
constexpr static std::uint32_t UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG = 4;
constexpr static std::uint32_t CIRCULAR_BUFFER_WORD_SIZE_BYTES = 16;
constexpr static std::uint32_t CIRCULAR_BUFFER_LOG2_WORD_SIZE_BYTES = 4;
