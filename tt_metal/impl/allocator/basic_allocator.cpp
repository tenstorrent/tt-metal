// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <basic_allocator.hpp>

namespace tt {

namespace tt_metal {

// Basic allocator has 1 bank per DRAM channel and 1 bank per L1 of Tensix core
BasicAllocator::BasicAllocator(const AllocatorConfig& alloc_config) :
    Allocator(
        alloc_config,
        {.dram = {.init = allocator::init_one_bank_per_channel, .alloc = allocator::base_alloc},
         .l1 = {.init = allocator::init_one_bank_per_l1, .alloc = allocator::base_alloc}}) {}

}  // namespace tt_metal

}  // namespace tt
