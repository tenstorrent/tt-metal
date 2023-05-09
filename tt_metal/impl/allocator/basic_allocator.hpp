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

    void clear_dram();

    void clear_l1();

    void clear();

   private:
    allocator::Algorithm &allocator_for_dram_channel(uint32_t bank_id) const;

    allocator::Algorithm &allocator_for_logical_core(uint32_t bank_id) const;

    allocator::Algorithm &get_allocator(uint32_t bank_id, const BufferType &buffer_type) const;

    std::string generate_bank_identifier_str(uint32_t bank_id, uint32_t size_bytes, const BufferType &buffer_type) const;

    BankIdToRelativeAddress allocate_contiguous_buffer(uint32_t bank_id, uint32_t size_bytes, const BufferType &buffer_type);

    BankIdToRelativeAddress allocate_contiguous_buffer(uint32_t bank_id, uint32_t address, uint32_t size_bytes, const BufferType &buffer_type);

    static constexpr bool allocate_bottom_up_ = true;
    tt_xy_pair logical_grid_size_;
    std::map<int, std::unique_ptr<allocator::Algorithm>> dram_manager_;
    std::map<tt_xy_pair, std::unique_ptr<allocator::Algorithm>> l1_manager_;
};

}  // namespace tt_metal

}  // namespace tt
