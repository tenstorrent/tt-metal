// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {

// Device-global, per-core L1 arena for objects whose lifetime spans programs.
// Allocations are lockstep only across their participating cores. Consequently,
// allocations on disjoint cores can reuse the same L1 address.
//
// Per-core worker L1 (addresses grow upward):
//
//   worker_l1_size
//   ┌─────────────────────────────────┐
//   │ L1_SMALL (global, top of bank)  │
//   ├─────────────────────────────────┤  worker_l1_size - l1_small_size
//   │                                 │  ← PersistentL1Arena limit
//   │ Global L1 Buffers (top-down)    │
//   │                                 │
//   │          (free)                 │
//   │                                 │
//   ├─────────────────────────────────┤  high_water_mark(this core)
//   │ Program-local / subdevice L1    │  ← DFB, CB, scratchpad  OR
//   │ (ephemeral stack, bottom-up)    │    subdevice local_l1_size slab
//   ├─────────────────────────────────┤
//   │ PrefetcherPipe config / ring    │  ← PersistentL1Arena
//   ├─────────────────────────────────┤  l1_unreserved_base
//   │ Firmware / reserved             │    (arena base)
//   └─────────────────────────────────┘
//   0
//
// Seal a core after program-local or subdevice L1 is placed on it so a later
// persistent allocation cannot overlap that snapshot. Unseal when that holder
// is destroyed.
class PersistentL1Arena {
public:
    struct Allocation {
        uint64_t id = 0;
        DeviceAddr address = 0;
        DeviceAddr size = 0;
    };

    PersistentL1Arena(DeviceAddr base, DeviceAddr limit, const CoreRangeSet& worker_grid);

    Allocation allocate(const CoreRangeSet& cores, DeviceAddr size, DeviceAddr alignment);
    void deallocate(uint64_t allocation_id);

    DeviceAddr high_water_mark(const CoreRangeSet& cores) const;
    std::vector<std::pair<DeviceAddr, DeviceAddr>> occupied_ranges() const;

    // Persistent allocations must be established before program-local L1 is
    // placed. Once a core is sealed, no new persistent allocation may use it.
    void seal(const CoreRangeSet& cores);
    void unseal(const CoreRangeSet& cores);

private:
    struct Region {
        DeviceAddr begin;
        DeviceAddr end;
        uint64_t allocation_id;
    };

    struct AllocationRecord {
        CoreRangeSet cores;
        DeviceAddr address;
        DeviceAddr size;
    };

    size_t seal_index(const CoreCoord& core) const;

    DeviceAddr base_;
    DeviceAddr limit_;
    uint32_t grid_width_ = 0;
    uint32_t grid_height_ = 0;
    uint64_t next_allocation_id_ = 1;
    mutable std::mutex mutex_;
    std::unordered_map<CoreCoord, std::vector<Region>> regions_by_core_;
    std::unordered_map<uint64_t, AllocationRecord> allocations_;
    std::vector<uint32_t> seal_refcounts_;
};

}  // namespace tt::tt_metal
