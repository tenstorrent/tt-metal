/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>

/*
* This file contains values that are visible to both host and device compiled code.
*/

constexpr static std::uint32_t INVALID = 0;
constexpr static std::uint32_t VALID = 1;
constexpr static std::uint32_t NOTIFY_HOST_KERNEL_COMPLETE_VALUE = 512;
// DRAM -> L1 and L1 -> DRAM transfers need to have 32B alignment, which means:
// DRAM_buffer_addr % 32 == L1_buffer_addr % 32, or
// DRAM_buffer_addr % 32 == L1_buffer_addr % 32 == 0
constexpr static std::uint32_t ADDRESS_ALIGNMENT = 32;
// Minimum size (in bytes) that will be allocated by host memory allocator
constexpr static std::uint32_t MIN_ALLOCATABLE_L1_SIZE_BYTES = 32;
constexpr static std::uint32_t MIN_ALLOCATABLE_DRAM_SIZE_BYTES = 1024;
