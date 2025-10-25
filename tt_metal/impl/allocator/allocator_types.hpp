// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

// Setup what each core-type is
enum class AllocCoreType {
    Dispatch,
    StorageOnly,
    ComputeOnly,
    ComputeAndStore,
    Invalid,
};

using BankMapping = std::vector<uint32_t>;

//! Allocator configuration -- decouples allocation from soc-desc - Up to user to populate from soc_desc
struct AllocatorConfig {
    //! DRAM specific configuration
    size_t num_dram_channels = 0;
    size_t dram_bank_size = 0;
    std::vector<size_t> dram_bank_offsets;
    uint32_t dram_unreserved_base = 0;
    uint32_t dram_alignment = 0;
    //! worker specific configuration
    uint32_t l1_unreserved_base = 0;
    CoreRangeSet worker_grid;
    size_t worker_l1_size = 0;
    std::optional<uint32_t> storage_core_bank_size = 0;
    size_t l1_small_size = 0;
    size_t trace_region_size = 0;
    std::unordered_map<CoreCoord, AllocCoreType> core_type_from_noc_coord_table;
    std::unordered_map<int, int> worker_log_to_virtual_routing_x;
    std::unordered_map<int, int> worker_log_to_virtual_routing_y;
    BankMapping
        l1_bank_remap;  // for remapping which l1 bank points to which bank if we assume normal row-major assignment
    CoreRangeSet compute_grid;
    uint32_t l1_alignment = 0;
    bool disable_interleaved = false;
    void reset();
    ~AllocatorConfig() { reset(); }
};

};  // namespace tt::tt_metal
