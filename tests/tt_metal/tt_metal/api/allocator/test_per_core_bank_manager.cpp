// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

DeviceAddr range_alloc(
    BankManager& bank_manager,
    uint32_t size,
    std::vector<AllocatorID> allocator_ids,
    bool bottom_up = true,
    const std::vector<std::pair<DeviceAddr, DeviceAddr>>& additional_occupied_ranges = {}) {
    return bank_manager.allocate_buffer_across_allocators(
        size, bottom_up, allocator_ids, additional_occupied_ranges);
}

DeviceAddr range_alloc_extents(
    BankManager& bank_manager,
    std::vector<std::pair<AllocatorID, DeviceAddr>> allocator_extents,
    bool bottom_up = true,
    const std::vector<std::pair<DeviceAddr, DeviceAddr>>& additional_occupied_ranges = {}) {
    return bank_manager.allocate_buffer_across_allocators(
        allocator_extents, bottom_up, additional_occupied_ranges);
}

constexpr AllocatorID LOCKSTEP{0};
constexpr AllocatorID BANK0{1};
constexpr AllocatorID BANK1{2};

}  // namespace per_core_bank_manager_tests

using namespace per_core_bank_manager_tests;

// Per-bank allocators are independent: different sizes yield different second-alloc offsets.
TEST(PerCoreAllocation, CPU_BanksAllocateDifferentSizes) {
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
TEST(PerCoreAllocation, CPU_LockstepAvoidsPerBankRegions) {
    auto bm = make_per_core_bank_manager(1024 * 1024, 1024, 2);

    alloc(bm, 4096, BANK0);  // [0, 4096)
    alloc(bm, 2048, BANK1);  // [0, 2048)

    // Lockstep must start after the max per-bank extent
    auto lockstep_addr = alloc(bm, 1024, LOCKSTEP);
    EXPECT_EQ(lockstep_addr, 4096u);
}

// Per-bank allocation must skip regions occupied by the lockstep allocator.
TEST(PerCoreAllocation, CPU_PerBankAvoidsLockstepRegion) {
    auto bm = make_per_core_bank_manager(1024 * 1024, 1024, 2);

    alloc(bm, 4096, LOCKSTEP);  // [0, 4096) on all banks

    auto addr_b0 = alloc(bm, 1024, BANK0);
    auto addr_b1 = alloc(bm, 1024, BANK1);
    EXPECT_EQ(addr_b0, 4096u);
    EXPECT_EQ(addr_b1, 4096u);
}

// Deallocating per-bank regions lets lockstep reuse that space.
TEST(PerCoreAllocation, CPU_DeallocatePerBankFreesForLockstep) {
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
TEST(PerCoreAllocation, CPU_HundredTenBanksScaling) {
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

TEST(PerCoreAllocation, CPU_RangeLockstepReusesAddressAcrossDisjointBanks) {
    auto bank_manager = make_per_core_bank_manager(1024 * 1024, 1024, 4);
    constexpr AllocatorID bank2{3};

    EXPECT_EQ(range_alloc(bank_manager, 4096, {BANK0, BANK1}), 0u);
    EXPECT_EQ(range_alloc(bank_manager, 4096, {bank2}), 0u);
    EXPECT_EQ(range_alloc(bank_manager, 1024, {BANK1, bank2}), 4096u);
}

TEST(PerCoreAllocation, CPU_RangeLockstepAvoidsLocalAndDefaultLockstepAllocations) {
    auto bank_manager = make_per_core_bank_manager(1024 * 1024, 1024, 2);

    EXPECT_EQ(alloc(bank_manager, 2048, LOCKSTEP), 0u);
    EXPECT_EQ(alloc(bank_manager, 3072, BANK1), 2048u);
    EXPECT_EQ(range_alloc(bank_manager, 1024, {BANK0, BANK1}), 5120u);
}

TEST(PerCoreAllocation, CPU_RangeLockstepHonorsAlignmentAndAdditionalRanges) {
    auto bank_manager = make_per_core_bank_manager(1024 * 1024, 1024, 2);

    const DeviceAddr address = range_alloc(
        bank_manager,
        2048,
        {BANK0, BANK1},
        /*bottom_up=*/true,
        {{0, 3072}});
    EXPECT_EQ(address, 3072u);
    EXPECT_EQ(address % 1024, 0u);
}

TEST(PerCoreAllocation, CPU_RangeLockstepPlacementIsIndependentOfBankOrder) {
    auto forward_manager = make_per_core_bank_manager(1024 * 1024, 1024, 2);
    auto reverse_manager = make_per_core_bank_manager(1024 * 1024, 1024, 2);
    alloc(forward_manager, 3072, BANK0);
    alloc(reverse_manager, 3072, BANK0);

    EXPECT_EQ(
        range_alloc(forward_manager, 2048, {BANK0, BANK1}),
        range_alloc(reverse_manager, 2048, {BANK1, BANK0}));
}

TEST(PerCoreAllocation, CPU_RangeLockstepDeallocationRestoresEverySelectedBank) {
    auto bank_manager = make_per_core_bank_manager(1024 * 1024, 1024, 2);
    const DeviceAddr address = range_alloc(bank_manager, 4096, {BANK0, BANK1});
    dealloc(bank_manager, address, BANK0);
    dealloc(bank_manager, address, BANK1);

    EXPECT_EQ(range_alloc(bank_manager, 4096, {BANK0, BANK1}), address);
}

TEST(PerCoreAllocation, CPU_VariableExtentRangeLockstepReservesOnlyEachBanksRequiredBytes) {
    constexpr DeviceAddr bank_size = 16 * 1024;
    auto uniform_manager = make_per_core_bank_manager(bank_size, 1024, 2);
    auto variable_manager = make_per_core_bank_manager(bank_size, 1024, 2);

    auto create_fragmented_ranges = [](BankManager& bank_manager) {
        EXPECT_EQ(alloc(bank_manager, 8 * 1024, BANK0), 0u);
        EXPECT_EQ(alloc(bank_manager, 4 * 1024, BANK0, /*bottom_up=*/false), 12 * 1024u);
        EXPECT_EQ(alloc(bank_manager, 8 * 1024, BANK1), 0u);
    };
    create_fragmented_ranges(uniform_manager);
    create_fragmented_ranges(variable_manager);

    EXPECT_ANY_THROW(range_alloc(uniform_manager, 8 * 1024, {BANK0, BANK1}));
    EXPECT_EQ(
        range_alloc_extents(variable_manager, {{BANK0, 4 * 1024}, {BANK1, 8 * 1024}}),
        8 * 1024u);
}

TEST(PerCoreAllocation, CPU_VariableExtentRangeLockstepTopDownUsesLargestValidCommonAddress) {
    auto bank_manager = make_per_core_bank_manager(16 * 1024, 1024, 2);

    EXPECT_EQ(
        range_alloc_extents(
            bank_manager,
            {{BANK0, 4 * 1024}, {BANK1, 8 * 1024}},
            /*bottom_up=*/false),
        8 * 1024u);
}

TEST(PerCoreAllocation, CPU_VariableExtentRangeLockstepKeepsAdditionalRangesPerBank) {
    auto bank_manager = make_per_core_bank_manager(16 * 1024, 1024, 2);
    const std::unordered_map<uint32_t, std::vector<std::pair<DeviceAddr, DeviceAddr>>> occupied_ranges_by_bank = {
        {BANK0.get(), {{0, 8 * 1024}, {12 * 1024, 16 * 1024}}},
        {BANK1.get(), {{0, 8 * 1024}}},
    };

    EXPECT_EQ(
        bank_manager.allocate_buffer_across_allocators(
            {{BANK0, 4 * 1024}, {BANK1, 8 * 1024}},
            /*bottom_up=*/true,
            occupied_ranges_by_bank),
        8 * 1024u);
}
