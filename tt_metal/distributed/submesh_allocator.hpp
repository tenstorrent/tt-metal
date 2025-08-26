// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::distributed {

// SubmeshAllocator coordinates allocation across N independent pools with a dependency graph.
// If parent -> child is declared, allocations in parent make the same address range unavailable in child
// (child's unavailability is the max of all parents' ranges at a given start address). Freeing a parent
// reduces the child's unavailability accordingly. Allocating in a child requires that all its parents are
// free at the chosen address.
class SubmeshAllocator {
public:
    using PoolId = unsigned int;

    struct AllocationHandle {
        PoolId pool;
        unsigned long long start;
        unsigned long long size;
    };

    // Create N pools with uniform capacity and alignment (in bytes).
    SubmeshAllocator(unsigned int num_pools, unsigned long long capacity_bytes, unsigned long long alignment_bytes);
    ~SubmeshAllocator();
    SubmeshAllocator(const SubmeshAllocator&) = delete;
    SubmeshAllocator& operator=(const SubmeshAllocator&) = delete;
    SubmeshAllocator(SubmeshAllocator&&) noexcept;
    SubmeshAllocator& operator=(SubmeshAllocator&&) noexcept;

    // Declare that 'child' depends on 'parent'. Parent allocations constrain child's free space.
    void add_dependency(PoolId parent, PoolId child);

    // Allocate 'size' in the specified pool. Returns true and fills 'out' on success.
    bool allocate(PoolId pool, unsigned long long size, AllocationHandle& out);

    // Allocate exactly at 'start' if possible (observes alignment and dependencies). Returns true on success.
    bool allocate_at(PoolId pool, unsigned long long start, unsigned long long size, AllocationHandle& out);

    // Free a prior allocation. No-op if the handle is invalid or already freed.
    void free(const AllocationHandle& handle);

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace tt::tt_metal::distributed
