// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

// Sentinel used in launch_msg rta_offset/crta_offset to mark no args present shared by host and firmware.
constexpr uint16_t RTA_CRTA_NO_ARGS_SENTINEL = 0xFFFF;
// Watcher debug pattern: upper 16 bits mark uninitialized RTA slots
// Lower 16 bits contain random value to prevent accidental matches
constexpr uint32_t WATCHER_RTA_UNSET_PATTERN = 0xBEEF0000;
