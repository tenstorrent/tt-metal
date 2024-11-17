// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

constexpr std::uint8_t NumTensixDispatchClasses = 3;
constexpr std::uint8_t noc_size_x = 13;
constexpr std::uint8_t noc_size_y = 12;
#define ALLOCATOR_ALIGNMENT 32
#define LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT 5
