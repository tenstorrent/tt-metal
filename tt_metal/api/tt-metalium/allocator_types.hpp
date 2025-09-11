// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdlib>
#include <functional>
#include <tt-metalium/core_coord.hpp>
#include <hostdevcommon/common_values.hpp>

namespace tt::tt_metal {

// Fwd declares
/*
MemoryBlockTable is a list of memory blocks in the following format:
[{"blockID": "0", "address": "0", "size": "0", "prevID": "0", "nextID": "0", "allocated": true}]
address: bytes
size: bytes
*/
using MemoryBlockTable = std::vector<std::unordered_map<std::string, std::string>>;
class Allocator;
class BankManager;

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
// Lightweight representation of allocator dependencies to avoid including BankManager in this header.
struct AllocatorDependenciesConfig {
    // Adjacency list indexed by allocator state id; each entry lists neighbor ids
    std::vector<std::vector<uint32_t>> dependencies{{}};  // Default single allocator (0) with no deps
};

struct AllocatorConfig {
    //! DRAM specific configuration
    size_t num_dram_channels = 0;
    size_t dram_bank_size = 0;
    std::vector<size_t> dram_bank_offsets = {};
    uint32_t dram_unreserved_base = 0;
    uint32_t dram_alignment = 0;
    //! worker specific configuration
    uint32_t l1_unreserved_base = 0;
    CoreRangeSet worker_grid = {};
    size_t worker_l1_size = 0;
    std::optional<uint32_t> storage_core_bank_size = 0;
    size_t l1_small_size = 0;
    size_t trace_region_size = 0;
    std::unordered_map<CoreCoord, AllocCoreType> core_type_from_noc_coord_table = {};
    std::unordered_map<int, int> worker_log_to_virtual_routing_x = {};
    std::unordered_map<int, int> worker_log_to_virtual_routing_y = {};
    BankMapping l1_bank_remap =
        {};  // for remapping which l1 bank points to which bank if we assume normal row-major assignment
    CoreRangeSet compute_grid = {};
    uint32_t l1_alignment = 0;
    bool disable_interleaved = false;
    // Dependencies between allocator states for overlapped submeshes
    AllocatorDependenciesConfig allocator_dependencies{};
    void reset();
    ~AllocatorConfig() { reset(); }
};

enum class MemoryAllocator {
    BASIC = 0,
    L1_BANKING = 1,
};

struct Statistics {
    size_t total_allocatable_size_bytes = 0;
    size_t total_allocated_bytes = 0;
    size_t total_free_bytes = 0;
    size_t largest_free_block_bytes = 0;
    std::vector<uint32_t>
        largest_free_block_addrs;  // addresses (relative to bank) that can hold the largest_free_block_bytes
};

}  // namespace tt::tt_metal
