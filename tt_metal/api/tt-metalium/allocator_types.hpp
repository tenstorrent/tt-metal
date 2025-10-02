// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>

namespace tt::tt_metal {

// Fwd declares
/*
MemoryBlockTable is a list of memory blocks in the following format:
[{"blockID": "0", "address": "0", "size": "0", "prevID": "0", "nextID": "0", "allocated": true}]
address: bytes
size: bytes
*/
using MemoryBlockTable = std::vector<std::unordered_map<std::string, std::string>>;

struct Statistics {
    size_t total_allocatable_size_bytes = 0;
    size_t total_allocated_bytes = 0;
    size_t total_free_bytes = 0;
    size_t largest_free_block_bytes = 0;
    std::vector<uint32_t>
        largest_free_block_addrs;  // addresses (relative to bank) that can hold the largest_free_block_bytes
};

}  // namespace tt::tt_metal
