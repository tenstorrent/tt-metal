#include "tt_metal/impl/allocator/basic_allocator.hpp"

namespace tt {

namespace tt_metal {

// Basic allocator has 1 bank per DRAM channel and 1 bank per L1 of Tensix core
BasicAllocator::BasicAllocator(const tt_SocDescriptor &soc_desc)
    : Allocator(
        soc_desc,
        {
            .dram = {
                .init=allocator::init_one_bank_per_channel,
                .alloc=allocator::alloc_one_bank_per_storage_unit,
                .alloc_at_addr=allocator::alloc_at_addr_one_bank_per_storage_unit
            },
            .l1 = {
                .init=allocator::init_one_bank_per_l1,
                .alloc=allocator::alloc_one_bank_per_storage_unit,
                .alloc_at_addr=allocator::alloc_at_addr_one_bank_per_storage_unit
            }
        }
    ) {}

}  // namespace tt_metal

}  // namespace tt
