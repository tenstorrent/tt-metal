#include "tt_metal/impl/allocator/basic_allocator.hpp"

namespace tt {

namespace tt_metal {

// Basic allocator has 1 bank per DRAM channel and 1 bank per L1 of Tensix core
BasicAllocator::BasicAllocator(const AllocatorConfig &alloc_config)
    : Allocator(
        alloc_config,
        {
            .dram = {
                .init=allocator::init_one_bank_per_channel,
                .alloc=allocator::base_alloc,
                .alloc_at_addr=allocator::base_alloc_at_addr
            },
            .l1 = {
                .init=allocator::init_one_bank_per_l1,
                .alloc=allocator::base_alloc,
                .alloc_at_addr=allocator::base_alloc_at_addr
            }
        }
    ) {}

}  // namespace tt_metal

}  // namespace tt
