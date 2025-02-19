// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "allocator.hpp"

namespace tt {

namespace tt_metal {

struct AllocatorConfig;

// For Grayskull:
// There are 108 (9x12) compute and storage cores where each core has one 1 MB bank with top 512 KB (non-exclusively)
// dedicated to L1 buffer storage. Circular buffers can grow into L1 buffer storage space but L1 buffers cannot grow
// past 512 KB. There are an additional 10 storage cores where each core has two banks of 512 KB dedicated solely to L1
// buffer storage. This gives a total of (108 + 1 bank) + (10 * 2 banks) = 128 banks of 512 KB for L1 buffers DRAM
// allocation is the same as BasicAllocator
class L1BankingAllocator : public Allocator {
public:
    L1BankingAllocator(const AllocatorConfig& alloc_config);
};

}  // namespace tt_metal

}  // namespace tt
