#include "tt_metal/impl/allocator/basic_allocator.hpp"

namespace tt {

namespace tt_metal {

// Basic allocator has 1 bank per DRAM channel and 1 bank per L1 of Tensix core
BasicAllocator::BasicAllocator(const tt_SocDescriptor &soc_desc)
    : Allocator(
        soc_desc,
        {.dram=allocator::init_one_bank_per_channel_manager, .l1=allocator::init_one_bank_per_l1_manager},
        {
            .dram=allocator::allocate_buffer_one_bank_per_storage_unit,
            .dram_at_address=allocator::allocate_buffer_at_address_one_bank_per_storage_unit,
            .l1=allocator::allocate_buffer_one_bank_per_storage_unit,
            .l1_at_address=allocator::allocate_buffer_at_address_one_bank_per_storage_unit
        }
    ) {}

}  // namespace tt_metal

}  // namespace tt
