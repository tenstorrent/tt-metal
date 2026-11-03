// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "program_allocation_support.hpp"
#include "tt-metalium/core_coord.hpp"

namespace tt::tt_metal {

class IDevice;

namespace detail {

class ProgramImpl;

// Places the circular buffers of a Metal 1.0 program in L1 and checks the placement against the device.
// invariants:
//  - An allocator belongs to exactly one ProgramImpl and lives as long as it.
//  - The program is registered with every device in `cb_devices_` while it has circular buffers.
class Metal1CircularBufferAllocator {
public:
    explicit Metal1CircularBufferAllocator(ProgramImpl& owner) : owner_(owner) {}

    // Makes the circular buffers on `cores` take part in the allocation.
    // precondition: the circular buffers were not allocated yet.
    void register_cores(const CoreRangeSet& cores);

    // The allocation metadata, for the L1 setup that is shared with dataflow buffers.
    EphemeralAllocationMetadata& allocation_metadata() { return cb_allocation_metadata_; }
    const EphemeralAllocationMetadata& allocation_metadata() const { return cb_allocation_metadata_; }

    // Marks the circular buffer layout as out of date.
    // side-effect: the program is compiled and allocated again on next use.
    void invalidate();

    // Gives each locally allocated circular buffer an L1 address on `device`.
    // Always used in conjunction with validate_region and compile
    // precondition: the allocation metadata is prepared for `device`.
    // postcondition: the program is registered with every physical device of `device`.
    // side-effect: reports the allocation to the graph tracker.
    void allocate(const IDevice* device);

    // Deallocate circular buffers and unregister from devices.
    // Releases the allocation on every device that holds it.
    // postcondition: the program is not registered with any device.
    void deallocate();

    // Releases the persistent L1 seals that the allocation holds.
    void release_seals();

    // Ensures that circular buffer core ranges are within the device compute grid.
    // Throws if a circular buffer core range is outside the compute grid of `device` and not on a service core.
    void validate_core_ranges(const IDevice* device);

    // Ensures that statically allocated circular buffers do not grow into L1 buffer space.
    // Throws if the static circular buffer region of any core range reaches live L1 buffers or service core memory.
    void validate_region(const IDevice* device);

    // CB tracking for SHM memory reporting
    // Merge exact core-range unions before expanding to individual cores.
    // Adds the L1 regions of the circular buffers to `regions_per_range`, merged per core range.
    // side-effect: mutates `regions_per_range`.
    void merge_l1_regions_by_core_range(
        std::map<CoreRange, std::vector<std::pair<uint64_t, uint64_t>>>& regions_per_range) const;

    // Splits the regions of each core range to the cores of that range, merged per core.
    static std::map<CoreCoord, std::vector<std::pair<uint64_t, uint64_t>>> expand_l1_regions_per_core(
        const std::map<CoreRange, std::vector<std::pair<uint64_t, uint64_t>>>& regions_per_range);

private:
    ProgramImpl& owner_;

    EphemeralAllocationMetadata cb_allocation_metadata_;
    // Tracks which devices this program has CBs allocated on (for CB memory reporting)
    std::unordered_set<const IDevice*> cb_devices_;
    bool local_circular_buffer_allocation_needed_{false};
};

}  // namespace detail
}  // namespace tt::tt_metal
