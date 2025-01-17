// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "allocator.hpp"

namespace tt {

namespace tt_metal {

struct AllocatorConfig;

namespace allocator {

void init_compute_and_storage_l1_bank_manager(Allocator& allocator, const AllocatorConfig& alloc_config);

uint64_t alloc_at_addr_in_compute_and_storage(
    const AllocatorConfig& config,
    BankManager& bank_manager,
    uint64_t size,
    uint64_t page_size,
    uint64_t relative_address);

}  // namespace allocator

// For Grayskull:
// There are 108 (9x12) compute and storage cores where each core has one 1 MB bank with top 512 KB (non-exclusively)
// dedicated to L1 buffer storage. Circular buffers can grow into L1 buffer storage space but L1 buffers cannot grow
// past 512 KB. There are an additional 10 storage cores where each core has two banks of 512 KB dedicated solely to L1
// buffer storage. This gives a total of (108 + 1 bank) + (10 * 2 banks) = 128 banks of 512 KB for L1 buffers DRAM
// allocation is the same as BasicAllocator
struct L1BankingAllocator : Allocator {
    L1BankingAllocator(const AllocatorConfig& alloc_config);
    ~L1BankingAllocator();
};

}  // namespace tt_metal

}  // namespace tt
