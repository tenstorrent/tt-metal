// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstddef>
#include <cstdint>

/*
 * This file contains values that are visible to both host and device compiled code.
 */

constexpr static std::uint32_t INVALID = 0;
constexpr static std::uint32_t VALID = 1;
constexpr static std::size_t DEFAULT_L1_SMALL_SIZE = 0;  //(1 << 15);  // 32KB
constexpr static std::size_t DEFAULT_TRACE_REGION_SIZE = 0;
constexpr static std::size_t DEFAULT_WORKER_L1_SIZE = 0;  // Size is dynamically determined based on the device type.
constexpr static std::uint32_t DISPATCH_MAX_MESSAGE_ENTRIES = 8;
constexpr static std::uint32_t MAX_NUM_HW_CQS = 2;
