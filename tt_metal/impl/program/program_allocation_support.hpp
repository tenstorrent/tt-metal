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
        //
        // When allocating, l1_regions is always non-empty as it is initialized by `prepare_ephemeral_allocation`.
        std::vector<std::pair<uint64_t, uint64_t>> l1_regions;

        // If address is the end of the last L1 region, the last region is extended by size bytes,
        //  otherwise address must be higher than existing regions and a new L1 region [address, size) is added
        void mark_address(uint64_t address, uint64_t size);

        // Reset when associated allocation is invalidated
        void reset_available_addresses() { *this = LockstepAllocationRegion(core_range); }
    };

    // Regions may have overlapping core_ranges
    std::vector<LockstepAllocationRegion> regions;

    // Per-core L1 is laid out as [persistent L1 arena, program-scope resources, ...], growing upward.
    // (See persistent_l1_arena.hpp for more.)
    //
    // Once `prepare_ephemeral_allocation` is called, further persistent L1 allocations must be forbidden on the
    // regions' cores, as they would overlap the program-scope resources placed above the arena.
    //
    // These seals stop the arena from giving out space on those cores, per arena and per core. The seals are released
    // via RAII once this metadata is destructed.
    std::unordered_map<PersistentL1Arena*, std::unordered_map<CoreCoord, PersistentL1Arena::Seal>> seals;

    // `PersistentL1Arena::high_water_mark` of an empty CoreRangeSet, to be returned when allocation on an empty core
    // seet is requested.
    DeviceAddr empty_cores_high_water_mark = 0;
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
//  - starts each region that has no allocations at its base address.
//  - "seal" persistent_l1 to prevent collision between address spaces during allocation.
void prepare_ephemeral_allocation(EphemeralAllocationMetadata& metadata, PersistentL1Arena& persistent_l1);

// TODO(#62451):
// The allocation logic should be unified in the future but we unfortunately have different allocation routine for now.
// The dfb allocation maybe erroneous and thus should be fixed while we unify the two.

// Performs lockstep allocation of a circular buffer or dataflow buffer across `cores`.
//
// Returns the start of the allocation, aligned to `alignment`; [address, address + size) is reserved on every core
// of `cores`. The address is read from the regions whose core_range exactly matches a range of `cores`.
//
// When an empty `cores` is supplied, no allocation will be performed. The return value forwards
// `PersistentL1Arena::high_water_mark` of an empty CoreRangeSet, when recorded by the last
// `prepare_ephemeral_allocation()` call.
//
// pre-condition:
//  - the cores are empty or
//      - every range of `cores` has been registered in `metadata` with register_ephemeral_allocation().
//  - prepare_ephemeral_allocation() has been called on `metadata`, and no region has been added since.
//
// post-condition:
//  - the returned address is at or above the start of the address range for user-facing memory objects (outside
//    firmware-reserved L1).
//  - when `cores` is not empty, [address, address + size) is reserved exclusively for the caller on every core of
//    `cores`: it overlaps neither persistent L1 nor any other allocation made through `metadata`.
//    TODO(#62451): can currently be violated when allocations are placed on overlapping core ranges.
//
// side-effect:
//  - mutates `metadata` regions that have core_range intersecting with `cores`.
DeviceAddr allocate_dfb(
    EphemeralAllocationMetadata& metadata, const CoreRangeSet& cores, uint64_t size, uint32_t alignment);

// Performs lockstep allocation of a scratchpad across `cores`.
//
// Returns the start of the allocation, aligned to `alignment`; [address, address + size) is reserved on every core
// of `cores`. The address is above every region whose core_range intersects `cores`, so it never overlaps an earlier
// allocation, but may reserve more than needed.
//
// When an empty `cores` is supplied, no allocation will be performed. The return value forwards
// `PersistentL1Arena::high_water_mark` of an empty CoreRangeSet, when recorded by the last
// `prepare_ephemeral_allocation()` call.
//
// pre-condition:
//  - the cores are empty or
//      - every range of `cores` has been registered in `metadata` with register_ephemeral_allocation().
//  - prepare_ephemeral_allocation() has been called on `metadata`, and no region has been added since.
//
// post-condition:
//  - the returned address is at or above the start of the address range for user-facing memory objects (outside
//    firmware-reserved L1).
//  - when `cores` is not empty, [address, address + size) is reserved exclusively for the caller on every core of
//    `cores`: it overlaps neither persistent L1 nor any other allocation made through `metadata`.
//
// side-effect:
//  - mutates `metadata` regions that have core_range intersecting with `cores`.
DeviceAddr allocate_scratchpad(
    EphemeralAllocationMetadata& metadata, const CoreRangeSet& cores, uint64_t size, uint32_t alignment);

// Clears all allocations recorded in `metadata`; the registered core ranges are kept.
void invalidate_ephemeral_allocation_metadata(EphemeralAllocationMetadata& metadata);

}  // namespace tt::tt_metal::detail
