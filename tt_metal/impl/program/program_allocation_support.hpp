// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/hal_types.hpp"  // DeviceAddr
#include "impl/allocator/persistent_l1_arena.hpp"

namespace tt::tt_metal::detail {

// Program-local (ephemeral) L1 allocation metadata, used with the functions below.
struct EphemeralAllocationMetadata {
    // Represents allocation metadata where a unit of CoreRange needs to be allocated lockstep.
    struct LockstepAllocationRegion {
        LockstepAllocationRegion(const CoreRange& core_range_) : core_range(core_range_) {}

        // The bank coordinates this Allocator represents
        //
        // Note that other allocators could be co-owning part of the cores.
        CoreRange core_range;

        // Holds vector of addresses where program-local resources are allocated [start, end)
        // There are multiple ranges because per core L1 regions are not in lockstep but program-local resources
        // spanning multiple cores must share the same address. To enable this, program-local resource address is the
        // maximum address amongst all of its target cores This vector is sorted from lower to higher address spaces
        std::vector<std::pair<uint64_t, uint64_t>> l1_regions;

        // Base address of the program-local allocation across the core_range.
        // This needs to be set after all persistent L1 allocations are accounted for.
        //
        // See persistent_l1_arena.hpp for more information about L1 address space allocation
        uint64_t base_address = 0;

        // If address is the end of the last L1 region, the last region is extended by size bytes,
        //  otherwise address must be higher than existing regions and a new L1 region [address, size) is added
        void mark_address(uint64_t address, uint64_t size);

        // Reset when associated allocation is invalidated
        void reset_available_addresses() { *this = LockstepAllocationRegion(core_range); }
    };

    std::vector<LockstepAllocationRegion> regions;
};

// Registers regions of allocation sharing the same allocation.
//
// side-effect: mutates metadata
void register_ephemeral_allocation(EphemeralAllocationMetadata& metadata, const CoreRangeSet& cores);

// This is to be called before program-local allocation begins.
//
// It prevents further persistent-l1 space allocations when they could intersect with program-local allocations.
// Also sets the base address of all the regions.
//
// side-effect:
//  - updates regions' base address.
//  - "seal" persistent_l1 to prevent collision between address spaces during `allocate_ephemeral_memory`.
void prepare_ephemeral_allocation(
    EphemeralAllocationMetadata& metadata,
    PersistentL1Arena& persistent_l1,
    std::unordered_map<CoreCoord, PersistentL1Arena::Seal>& seals);

// Performs lockstep allocation of a program-local resource across `cores`.
//
// Returns the start of the allocation, aligned to `alignment`; [address, address + size) is reserved on every core
// of `cores`.
//
// pre-condition:
//  - every range of `cores` has been registered in `metadata` with register_ephemeral_allocation().
//  - prepare_ephemeral_allocation() has been called on `metadata`, and no region has been added since.
//
// side-effect:
//  - mutates `metadata` regions that have core_range intersecting with `cores`.
DeviceAddr allocate_ephemeral_memory(
    EphemeralAllocationMetadata& metadata, const CoreRangeSet& cores, uint64_t size, uint32_t alignment);

// Clears all allocations recorded in `metadata`; the registered core ranges are kept.
void invalidate_ephemeral_allocation_metadata(EphemeralAllocationMetadata& metadata);

}  // namespace tt::tt_metal::detail
