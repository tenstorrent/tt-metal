// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
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
// persistent allocation cannot silently overlap that snapshot. Hold the returned
// Seal until that placement is gone; its destructor unseals.
//
// A seal is either fixed or relocatable. A fixed seal (subdevice local L1 slab)
// makes allocate() fail on that core. A relocatable seal names a Relocatable
// holder (a Program) whose placement can move: allocate() places the new region
// and then has every holder sealed on its cores give up its seals and place its
// program-local L1 again, above the new high-water mark, before it next launches.
class PersistentL1Arena {
public:
    struct Allocation {
        uint64_t id = 0;
        DeviceAddr address = 0;
        DeviceAddr size = 0;
    };

    // A holder whose program-local L1 can be placed again after a persistent
    // allocation lands under it.
    class Relocatable {
    public:
        // Called by allocate() without the arena lock held, at most once per
        // allocation. The holder must release every Seal it holds on `arena`
        // and place its program-local L1 again before its next launch. The
        // holder's previous placement may already overlap the new allocation,
        // so nothing may still be running from it: the caller of allocate()
        // drains the device first.
        virtual void relocate_persistent_l1(PersistentL1Arena& arena) noexcept = 0;

    protected:
        // Holders are never destroyed through this interface.
        ~Relocatable() = default;
    };

    class Seal {
    public:
        Seal() = default;
        Seal(Seal&& other) noexcept;
        Seal& operator=(Seal&& other) noexcept;
        ~Seal();

        Seal(const Seal&) = delete;
        Seal& operator=(const Seal&) = delete;

    private:
        friend class PersistentL1Arena;
        Seal(const std::shared_ptr<PersistentL1Arena*>& arena, CoreRangeSet cores, const Relocatable* holder);

        std::weak_ptr<PersistentL1Arena*> arena_;
        CoreRangeSet cores_;
        // Identifies the relocatable holder in the arena's registry; null for a fixed seal.
        // Never dereferenced: the holder may already be mid-destruction when this Seal is.
        const Relocatable* holder_ = nullptr;
    };

    PersistentL1Arena(DeviceAddr base, DeviceAddr limit, const CoreRangeSet& worker_grid);
    ~PersistentL1Arena();

    PersistentL1Arena(const PersistentL1Arena&) = delete;
    PersistentL1Arena& operator=(const PersistentL1Arena&) = delete;

    // Places `size` bytes at the same address on every core of `cores`. Fails if any of
    // those cores holds a fixed seal. Holders with relocatable seals on those cores are
    // relocated (each once) after the region is placed; see Relocatable.
    Allocation allocate(const CoreRangeSet& cores, DeviceAddr size, DeviceAddr alignment);
    void deallocate(uint64_t allocation_id);

    // True when allocate(cores, ...) would relocate holders: some core is sealed by a
    // relocatable holder and none holds a fixed seal (on which allocate fails instead).
    bool would_relocate(const CoreRangeSet& cores) const;

    DeviceAddr high_water_mark(const CoreRangeSet& cores) const;
    std::vector<std::pair<DeviceAddr, DeviceAddr>> occupied_ranges() const;
    std::vector<std::pair<DeviceAddr, DeviceAddr>> occupied_ranges(const CoreCoord& core) const;

    // Fixed seal: no persistent allocation may use these cores while it is held.
    [[nodiscard]] Seal seal(const CoreRangeSet& cores);
    // Relocatable seal: a persistent allocation on these cores relocates `holder`. The
    // holder must be owned by a shared_ptr; the arena keeps only a weak reference, and a
    // holder already being destroyed when an allocation lands is skipped.
    [[nodiscard]] Seal seal(const CoreRangeSet& cores, const std::weak_ptr<Relocatable>& holder);

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

    struct RelocatableSeals {
        // Weak, so a make_shared holder's storage (and thus its key) is not reused while this
        // entry exists.
        std::weak_ptr<Relocatable> holder;
        // Relocatable Seals of this holder that cover the core.
        uint32_t count = 0;
    };

    struct CoreSeals {
        uint32_t num_fixed = 0;
        std::unordered_map<const Relocatable*, RelocatableSeals> relocatable;
    };

    struct SealScan {
        // The first scanned core that holds a fixed seal; the scan stops there.
        std::optional<CoreCoord> fixed_core;
        // Each relocatable holder sealed on the scanned cores, once, including expired ones.
        std::vector<std::weak_ptr<Relocatable>> holders;
    };

    size_t seal_index(const CoreCoord& core) const;
    // Caller holds mutex_.
    SealScan scan_seals_locked(const std::vector<CoreCoord>& core_list) const;
    // Places the region; caller holds mutex_.
    Allocation place_locked(
        const CoreRangeSet& cores, const std::vector<CoreCoord>& core_list, DeviceAddr size, DeviceAddr alignment);
    // Caller holds mutex_. A null holder adds a fixed seal.
    void add_seal_locked(const CoreRangeSet& cores, const std::shared_ptr<Relocatable>& holder);
    void release_seal(const CoreRangeSet& cores, const Relocatable* holder);

    // Liveness token handed to every Seal. Nulled and released in the destructor so a
    // Seal destroyed after the arena unseals into nothing instead of freed memory.
    std::shared_ptr<PersistentL1Arena*> liveness_;

    DeviceAddr base_;
    DeviceAddr limit_;
    uint32_t grid_width_ = 0;
    uint32_t grid_height_ = 0;
    uint64_t next_allocation_id_ = 1;
    mutable std::mutex mutex_;
    std::unordered_map<CoreCoord, std::vector<Region>> regions_by_core_;
    std::unordered_map<uint64_t, AllocationRecord> allocations_;
    // Indexed by seal_index().
    std::vector<CoreSeals> seals_;
};

}  // namespace tt::tt_metal
