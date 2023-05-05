#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <variant>
#include <memory>

#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/common/tt_soc_descriptor.h"
#include "tt_metal/impl/allocator/algorithms/allocator_algorithm.hpp"

namespace tt {

namespace tt_metal {

// Currently only designed for Grayskull.
// There are 108 (9x12) compute and storage cores where each core has one 1 MB bank with top 512 KB (non-exclusively) dedicated to L1 buffer storage.
// Circular buffers can grow into L1 buffer storage space but L1 buffers cannot grow past 512 KB.
// There are an additional 10 storage cores where each core has two banks of 512 KB dedicated solely to L1 buffer storage.
// This gives a total of (108 + 1 bank) + (10 * 2 banks) = 128 banks of 512 KB for L1 buffers
// DRAM allocation is the same as BasicAllocator
class L1BankingAllocator : public Allocator {
   public:
    L1BankingAllocator(const tt_SocDescriptor &soc_desc);

    ~L1BankingAllocator() {}

    // TODO: Update copy/move semantics
    L1BankingAllocator(const L1BankingAllocator &other) { }
    L1BankingAllocator& operator=(const L1BankingAllocator &other) { return *this; }

    L1BankingAllocator(L1BankingAllocator &&other) { }
    L1BankingAllocator& operator=(L1BankingAllocator &&other) { return *this; }

    uint32_t allocate_dram_buffer(int dram_channel, uint32_t size_bytes);

    uint32_t allocate_dram_buffer(int dram_channel, uint32_t start_address, uint32_t size_bytes);

    std::vector<DramBankAddrPair> allocate_interleaved_dram_buffer(int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry);

    void deallocate_dram_buffer(int dram_channel, uint32_t address);

    uint32_t allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes);

    uint32_t allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes);

    uint32_t allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes);

    uint32_t allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes);

    std::vector<L1BankAddrPair> allocate_interleaved_l1_buffer(int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry);

    uint32_t get_address_for_circular_buffers_across_core_range(const std::pair<tt_xy_pair, tt_xy_pair> &logical_core_range, uint32_t size_in_bytes) const;

    void deallocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t address);

    void clear_dram();

    void clear_l1();

    void clear();

   private:
    constexpr static uint32_t storage_core_bank_size_bytes_ = 512 * 1024;
    constexpr static int num_banks_per_storage_core_ = 2;
    constexpr static uint32_t min_allocation_size_bytes_ = 32;
    // DRAM -> L1 and L1 -> DRAM transfers need to have 32B alignment, which means:
    // DRAM_buffer_addr % 32 == L1_buffer_addr % 32, or
    // DRAM_buffer_addr % 32 == L1_buffer_addr % 32 == 0
    constexpr static uint32_t alignment_ = 32;

    struct Bank {
        std::unique_ptr<allocator::Algorithm> allocator_algo;
        uint32_t offset_bytes;
        Bank(std::unique_ptr<allocator::Algorithm> allocator, uint32_t offset) : allocator_algo(std::move(allocator)), offset_bytes(offset) {}
    };

    using UniqueBank = std::unique_ptr<Bank>;
    using UniqueBanks = std::vector<UniqueBank>;

    void init_dram_manager(const tt_SocDescriptor &soc_desc);

    void init_compute_and_storage_cores_l1_manager(const tt_SocDescriptor &soc_desc);

    void init_storage_cores_l1_manager(const tt_SocDescriptor &soc_desc);

    allocator::Algorithm &allocator_for_dram_channel(int dram_channel) const;

    bool is_compute_and_storage_core(const tt_xy_pair &logical_core) const;

    bool is_storage_only_core(const tt_xy_pair &logical_core) const;

    Bank &bank_for_logical_compute_and_storage_core(const tt_xy_pair &logical_core) const;

    UniqueBanks &banks_for_storage_only_cores(const tt_xy_pair &logical_core);

    Bank &bank_for_logical_core(const tt_xy_pair &logical_core, uint32_t absolute_address) const;

    std::map<int, std::unique_ptr<allocator::Algorithm>> dram_manager_;

    tt_xy_pair logical_grid_size_;
    std::map<tt_xy_pair, UniqueBank> compute_and_storage_cores_l1_manager_;
    std::map<tt_xy_pair, UniqueBanks> storage_cores_l1_manager_;
};

}  // namespace tt_metal

}  // namespace tt
