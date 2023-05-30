#pragma once

#include <vector>
#include <cstdlib>
#include "common/core_coord.h"
#include "hostdevcommon/common_values.hpp"

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

using BankMapping = std::vector<u32>;

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
    BankMapping l1_bank_remap = {}; // for remapping which l1 bank points to which bank if we assume normal row-major assignment
};

enum class MemoryAllocator {
    BASIC = 0,
    L1_BANKING = 1,
};

struct AddressDescriptor {
    u32 offset_bytes = 0;
    u32 relative_address = 0;

    u32 absolute_address()     const {
        return offset_bytes + relative_address;
    }
};

using BankIdToRelativeAddress = std::unordered_map<u32, AddressDescriptor>;

struct BankDescriptor {
    u32 offset_bytes = 0;
    u32 size_bytes = 0;
    // This is to store offsets for any banks that share a core (storage core), so we can view all banks similarly on storage cores
    // set to 0 for cores with only 1 bank
    i32 bank_offset_bytes = 0;
};

namespace allocator {

struct InitAndAllocFuncs {
    std::function<void(Allocator &, const AllocatorConfig &)> init;
    std::function<BankIdToRelativeAddress(const AllocatorConfig &, BankManager &, u32, u32, u32, bool)> alloc;
    std::function<BankIdToRelativeAddress(const AllocatorConfig &, BankManager &, u32, u32, u32, u32)> alloc_at_addr;
};

// Holds callback functions required by allocators that specify how to initialize the bank managers and what the allocation scheme
// is for a given storage substrate
struct AllocDescriptor {
    InitAndAllocFuncs dram;
    InitAndAllocFuncs l1;
};

}

}
