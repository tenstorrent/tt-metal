#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>

#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/common/tt_soc_descriptor.h"
#include "tt_metal/impl/allocator/algorithms/allocator_algorithm.hpp"

namespace tt {

namespace tt_metal {

class BasicAllocator : public Allocator {
   public:
    BasicAllocator(const tt_SocDescriptor &soc_desc);

    ~BasicAllocator() {}

    // TODO: Update copy/move semantics
    BasicAllocator(const BasicAllocator &other) { }
    BasicAllocator& operator=(const BasicAllocator &other) { return *this; }

    BasicAllocator(BasicAllocator &&other) { }
    BasicAllocator& operator=(BasicAllocator &&other) { return *this; }

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
    static constexpr bool allocate_bottom_up_ = true;
    allocator::Algorithm &allocator_for_dram_channel(int dram_channel) const;

    allocator::Algorithm &allocator_for_logical_core(const tt_xy_pair &logical_core) const;

    tt_xy_pair logical_grid_size_;
    std::map<int, std::unique_ptr<allocator::Algorithm>> dram_manager_;
    std::map<tt_xy_pair, std::unique_ptr<allocator::Algorithm>> l1_manager_;
};

}  // namespace tt_metal

}  // namespace tt
