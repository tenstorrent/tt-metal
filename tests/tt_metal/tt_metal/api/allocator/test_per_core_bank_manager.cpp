// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests for per-core allocation using BankManager's AllocatorDependencies.
//
// The per-core allocation model uses N+1 allocators inside one BankManager:
//   Allocator 0       = lockstep  (same address on all banks/cores)
//   Allocator 1..N    = per-bank  (independent address per bank/core)
//
// Dependency graph:
//   Allocator 0 depends on {1, 2, ..., N}  — lockstep must avoid all per-bank regions
//   Allocator k depends on {0}              — per-bank must avoid lockstep regions
//   Per-bank allocators are independent of each other (separate physical L1s)

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/impl/allocator/bank_manager.hpp"

namespace per_core_bank_manager_tests {

using namespace tt::tt_metal;
using AllocatorID = BankManager::AllocatorDependencies::AllocatorID;

// Helper: build the per-core dependency graph for N banks.
BankManager::AllocatorDependencies make_per_core_dependencies(uint32_t num_banks) {
    std::unordered_map<AllocatorID, ttsl::SmallVector<AllocatorID>> deps_map;
    ttsl::SmallVector<AllocatorID> lockstep_deps;
    for (uint32_t i = 1; i <= num_banks; i++) {
        lockstep_deps.push_back(AllocatorID{i});
        deps_map[AllocatorID{i}] = {AllocatorID{0}};
    }
    deps_map[AllocatorID{0}] = lockstep_deps;
    return BankManager::AllocatorDependencies{deps_map};
}

// Helper: create a BankManager with per-core dependencies.
BankManager make_per_core_bank_manager(uint64_t bank_size, uint32_t alignment, uint32_t num_banks) {
    std::vector<int64_t> bank_offsets(num_banks, 0);
    auto deps = make_per_core_dependencies(num_banks);
    return BankManager(
        BufferType::DRAM,
        bank_offsets,
        bank_size,
        alignment,
        /*dram_alignment_bytes=*/alignment,
        /*alloc_offset=*/0,
        /*disable_interleaved=*/false,
        deps);
}

DeviceAddr alloc(BankManager& bm, uint32_t size, AllocatorID id, bool bottom_up = true) {
    return bm.allocate_buffer(size, size, bottom_up, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, id);
}

void dealloc(BankManager& bm, DeviceAddr addr, AllocatorID id) { bm.deallocate_buffer(addr, id); }

constexpr AllocatorID LOCKSTEP{0};
constexpr AllocatorID BANK0{1};
constexpr AllocatorID BANK1{2};

}  // namespace per_core_bank_manager_tests

using namespace per_core_bank_manager_tests;

// Per-bank allocators are independent: different sizes yield different second-alloc offsets.
TEST(PerCoreAllocation, BanksAllocateDifferentSizes) {
    auto bm = make_per_core_bank_manager(1024 * 1024, 1024, 2);

    auto addr_b0 = alloc(bm, 1024, BANK0);
    auto addr_b1 = alloc(bm, 8192, BANK1);
    EXPECT_EQ(addr_b0, 0u);
    EXPECT_EQ(addr_b1, 0u);

    auto addr_b0_2 = alloc(bm, 1024, BANK0);
    auto addr_b1_2 = alloc(bm, 1024, BANK1);
    EXPECT_EQ(addr_b0_2, 1024u);
    EXPECT_EQ(addr_b1_2, 8192u);
}

// Lockstep allocation must skip regions occupied by any per-bank allocator.
TEST(PerCoreAllocation, LockstepAvoidsPerBankRegions) {
    auto bm = make_per_core_bank_manager(1024 * 1024, 1024, 2);

    alloc(bm, 4096, BANK0);  // [0, 4096)
    alloc(bm, 2048, BANK1);  // [0, 2048)

    // Lockstep must start after the max per-bank extent
    auto lockstep_addr = alloc(bm, 1024, LOCKSTEP);
    EXPECT_EQ(lockstep_addr, 4096u);
}

// Per-bank allocation must skip regions occupied by the lockstep allocator.
TEST(PerCoreAllocation, PerBankAvoidsLockstepRegion) {
    auto bm = make_per_core_bank_manager(1024 * 1024, 1024, 2);

    alloc(bm, 4096, LOCKSTEP);  // [0, 4096) on all banks

    auto addr_b0 = alloc(bm, 1024, BANK0);
    auto addr_b1 = alloc(bm, 1024, BANK1);
    EXPECT_EQ(addr_b0, 4096u);
    EXPECT_EQ(addr_b1, 4096u);
}

// Deallocating per-bank regions lets lockstep reuse that space.
TEST(PerCoreAllocation, DeallocatePerBankFreesForLockstep) {
    auto bm = make_per_core_bank_manager(1024 * 1024, 1024, 2);

    auto b0 = alloc(bm, 4096, BANK0);
    auto b1 = alloc(bm, 4096, BANK1);
    EXPECT_EQ(alloc(bm, 1024, LOCKSTEP), 4096u);

    dealloc(bm, b0, BANK0);
    dealloc(bm, b1, BANK1);

    // Lockstep can now reuse [0, 4096)
    EXPECT_EQ(alloc(bm, 1024, LOCKSTEP), 0u);
}

// Scale test: 110 banks (realistic Blackhole core count).
TEST(PerCoreAllocation, HundredTenBanksScaling) {
    constexpr uint32_t NUM_BANKS = 110;
    auto deps = make_per_core_dependencies(NUM_BANKS);
    EXPECT_EQ(deps.num_allocators(), NUM_BANKS + 1);

    std::vector<int64_t> bank_offsets(NUM_BANKS, 0);
    BankManager bm(BufferType::DRAM, bank_offsets, 1024 * 1024, 1024, 1024, 0, false, deps);
    CoreRangeSet grid(CoreRange(CoreCoord(0, 0), CoreCoord(NUM_BANKS - 1, 0)));

    // Allocate different sizes on each bank
    for (uint32_t i = 0; i < NUM_BANKS; i++) {
        uint32_t size = (i + 1) * 1024;
        auto addr = bm.allocate_buffer(size, size, true, grid, 1, AllocatorID{i + 1});
        EXPECT_EQ(addr, 0u);  // All start at 0 (independent)
    }

    // Lockstep must start after the largest per-bank allocation (110KB)
    auto ls = bm.allocate_buffer(1024, 1024, true, grid, std::nullopt, AllocatorID{0});
    EXPECT_EQ(ls, 110u * 1024);
}
