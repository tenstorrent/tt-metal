#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <unordered_map>
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

    BankIdToRelativeAddress allocate_buffer(uint32_t starting_bank_id, uint32_t size, uint32_t page_size, const BufferType &buffer_type);

    BankIdToRelativeAddress allocate_buffer(uint32_t starting_bank_id, uint32_t size, uint32_t page_size, uint32_t address, const BufferType &buffer_type);

    void deallocate_buffer(uint32_t bank_id, uint32_t address, const BufferType &buffer_type);

    uint32_t num_banks(const BufferType &buffer_type) const;

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;

    tt_xy_pair logical_core_from_bank_id(uint32_t bank_id) const;

    std::vector<uint32_t> bank_ids_from_dram_channel(uint32_t dram_channel) const;

    std::vector<uint32_t> bank_ids_from_logical_core(const tt_xy_pair &logical_core) const;

    uint32_t allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes);

    uint32_t allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes);

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

    struct L1Bank {
        std::unique_ptr<allocator::Algorithm> allocator_algo;
        uint32_t offset_bytes;
        L1Bank(std::unique_ptr<allocator::Algorithm> allocator, uint32_t offset) : allocator_algo(std::move(allocator)), offset_bytes(offset) {}
    };

    using UniqueL1Bank = std::unique_ptr<L1Bank>;
    using UniqueL1Banks = std::vector<UniqueL1Bank>;

    void init_dram_manager(const tt_SocDescriptor &soc_desc);

    void init_compute_and_storage_cores_l1_manager(const tt_SocDescriptor &soc_desc);

    void init_storage_cores_l1_manager(const tt_SocDescriptor &soc_desc);

    void init_l1_bank_id_to_logical_core_mapping();

    uint32_t num_l1_banks() const;

    allocator::Algorithm &allocator_for_dram_channel(uint32_t bank_id) const;

    bool is_compute_and_storage_core(const tt_xy_pair &logical_core) const;

    bool is_storage_only_core(const tt_xy_pair &logical_core) const;

    L1Bank &bank_for_logical_compute_and_storage_core(uint32_t bank_id) const;

    UniqueL1Banks &banks_for_storage_only_cores(uint32_t bank_id);

    L1Bank &bank_for_logical_core(uint32_t bank_id, uint32_t absolute_address) const;

    BankIdToRelativeAddress allocate_dram_buffer(uint32_t bank_id, uint32_t size_bytes);

    BankIdToRelativeAddress allocate_l1_buffer(uint32_t bank_id, uint32_t size_bytes);

    BankIdToRelativeAddress allocate_contiguous_buffer(uint32_t bank_id, uint32_t size_bytes, const BufferType &buffer_type);

    BankIdToRelativeAddress allocate_dram_buffer(uint32_t bank_id, uint32_t start_address, uint32_t size_bytes);

    BankIdToRelativeAddress allocate_l1_buffer(uint32_t bank_id, uint32_t start_address, uint32_t size_bytes);

    BankIdToRelativeAddress allocate_contiguous_buffer(uint32_t bank_id, uint32_t start_address, uint32_t size_bytes, const BufferType &buffer_type);

    void deallocate_dram_buffer(uint32_t bank_id, uint32_t address);

    void deallocate_l1_buffer(uint32_t bank_id, uint32_t address);

    BankIdToRelativeAddress allocate_interleaved_dram_buffer(uint32_t num_banks, uint32_t starting_bank_id, uint32_t num_pages, uint32_t page_size);

    BankIdToRelativeAddress allocate_interleaved_l1_buffer(uint32_t num_banks, uint32_t starting_bank_id, uint32_t num_pages, uint32_t page_size);

    BankIdToRelativeAddress allocate_interleaved_buffer(uint32_t starting_bank_id, uint32_t num_pages, uint32_t page_size, const BufferType &buffer_type);

    BankIdToRelativeAddress allocate_interleaved_dram_buffer(uint32_t num_banks, uint32_t starting_bank_id, uint32_t num_pages, uint32_t page_size, uint32_t address);

    BankIdToRelativeAddress allocate_interleaved_l1_buffer(uint32_t num_banks, uint32_t starting_bank_id, uint32_t num_pages, uint32_t page_size, uint32_t address);

    BankIdToRelativeAddress allocate_interleaved_buffer(uint32_t starting_bank_id, uint32_t num_pages, uint32_t page_size, uint32_t address, const BufferType &buffer_type);

    std::map<int, std::unique_ptr<allocator::Algorithm>> dram_manager_;

    std::unordered_map<uint32_t, tt_xy_pair> bank_id_to_logical_core_;
    std::unordered_map<tt_xy_pair, std::vector<uint32_t>> logical_core_to_bank_ids_;
    tt_xy_pair logical_grid_size_;
    std::map<tt_xy_pair, UniqueL1Bank> compute_and_storage_cores_l1_manager_;
    std::map<tt_xy_pair, UniqueL1Banks> storage_cores_l1_manager_;
};

}  // namespace tt_metal

}  // namespace tt
