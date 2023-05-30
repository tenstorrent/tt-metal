#pragma once

#include <vector>
#include <cstdlib>
#include "common/core_coord.h"

namespace tt::tt_metal {

// Fwd declares
struct Allocator;
namespace allocator {
    class BankManager;
}

// Setup what each core-type is
enum class AllocCoreType {
    Dispatch,
    StorageOnly,
    ComputeOnly,
    ComputeAndStore,
    Invalid,
};

//! Allocator configuration -- decouples allocation from soc-desc - Up to user to populate from soc_desc
struct AllocatorConfig {
    //! DRAM specific configuration
    size_t num_dram_channels = 0;
    size_t dram_bank_size = 0;
    //! worker specific configuration
    CoreCoord worker_grid_size = {};
    size_t worker_l1_size = 0;
    size_t storage_core_l1_bank_size = 0;
    std::unordered_map<CoreCoord, AllocCoreType> core_type_from_noc_coord_table = {};
    std::unordered_map<CoreCoord, CoreCoord> logical_to_routing_coord_lookup_table = {};
};

enum class MemoryAllocator {
    BASIC = 0,
    L1_BANKING = 1,
};

struct AddressDescriptor {
    uint32_t offset_bytes;
    uint32_t relative_address;

    uint32_t absolute_address()     const {
        return offset_bytes + relative_address;
    }
};

using BankIdToRelativeAddress = std::unordered_map<uint32_t, AddressDescriptor>;

struct BankDescriptor {
    uint32_t offset_bytes;
    uint32_t size_bytes;
};

namespace allocator {

struct InitAndAllocFuncs {
    std::function<void(Allocator &, const AllocatorConfig &)> init;
    std::function<BankIdToRelativeAddress(const AllocatorConfig &, BankManager &, uint32_t, uint32_t, uint32_t, bool)> alloc;
    std::function<BankIdToRelativeAddress(const AllocatorConfig &, BankManager &, uint32_t, uint32_t, uint32_t, uint32_t)> alloc_at_addr;
};

// Holds callback functions required by allocators that specify how to initialize the bank managers and what the allocation scheme
// is for a given storage substrate
struct AllocDescriptor {
    InitAndAllocFuncs dram;
    InitAndAllocFuncs l1;
};

}

}
