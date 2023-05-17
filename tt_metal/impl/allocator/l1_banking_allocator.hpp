#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <unordered_map>
#include <variant>
#include <memory>

#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/common/tt_soc_descriptor.h"
#include "tt_metal/impl/allocator/algorithms/allocator_algorithm.hpp"

namespace tt {

namespace tt_metal {

namespace allocator {

void init_compute_and_storage_l1_bank_manager(Allocator &allocator, const tt_SocDescriptor &soc_desc);

BankIdToRelativeAddress alloc_in_compute_and_storage_l1(BankManager &bank_manager, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, bool bottom_up);

BankIdToRelativeAddress alloc_at_addr_in_compute_and_storage(BankManager &bank_manager, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, uint32_t absolute_address);

}   // namespace allocator


// Currently only designed for Grayskull.
// There are 108 (9x12) compute and storage cores where each core has one 1 MB bank with top 512 KB (non-exclusively) dedicated to L1 buffer storage.
// Circular buffers can grow into L1 buffer storage space but L1 buffers cannot grow past 512 KB.
// There are an additional 10 storage cores where each core has two banks of 512 KB dedicated solely to L1 buffer storage.
// This gives a total of (108 + 1 bank) + (10 * 2 banks) = 128 banks of 512 KB for L1 buffers
// DRAM allocation is the same as BasicAllocator
struct L1BankingAllocator : Allocator {
    L1BankingAllocator(const tt_SocDescriptor &soc_desc);
};

}  // namespace tt_metal

}  // namespace tt
