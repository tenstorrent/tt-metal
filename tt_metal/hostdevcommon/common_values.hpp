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
constexpr static std::uint32_t NOTIFY_HOST_KERNEL_COMPLETE_VALUE = 512;
// DRAM -> L1 and L1 -> DRAM transfers need to have 32B alignment, which means:
// DRAM_buffer_addr % 32 == L1_buffer_addr % 32, or
// DRAM_buffer_addr % 32 == L1_buffer_addr % 32 == 0
constexpr static std::uint32_t ADDRESS_ALIGNMENT = 32;
constexpr static std::size_t DEFAULT_L1_SMALL_SIZE = 0;  //(1 << 15);  // 32KB

// Number of entries for each core/shard in dispatch command
constexpr static std::uint32_t NUM_ENTRIES_PER_SHARD = 3;    //per shard/core 3 entries (uint32_t each)
                                                             //first number of pages in core/shard
                                                             //second x coordinate
                                                             //third y coordinate
