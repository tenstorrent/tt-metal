#pragma once

#include <vector>
#include <cstdlib>
#include "common/core_coord.h"

namespace tt::tt_metal {

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
    std::unordered_map<CoreCoord, AllocCoreType> core_type_from_noc_coord_table = {};
    std::unordered_map<CoreCoord, CoreCoord> logical_to_routing_coord_lookup_table = {};
};

}
