// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

/*
 * This file contains values that are visible to both host and device compiled code.
 */

constexpr static std::uint32_t INVALID = 0;
constexpr static std::uint32_t VALID = 1;
constexpr static std::size_t DEFAULT_L1_SMALL_SIZE = 0;  //(1 << 15);  // 32KB
constexpr static std::size_t DEFAULT_TRACE_REGION_SIZE = 0;
