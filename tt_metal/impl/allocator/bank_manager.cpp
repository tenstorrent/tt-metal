// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bank_manager.hpp"

#include <enchantum/enchantum.hpp>
#include <util.hpp>
#include <limits>
#include <string_view>
#include <utility>

#include "allocator/algorithms/allocator_algorithm.hpp"
#include "allocator_types.hpp"
#include "assert.hpp"
#include "buffer_types.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/impl/allocator/algorithms/free_list_opt.hpp"

namespace tt {

namespace tt_metal {

void BankManager::init_allocator(DeviceAddr size_bytes, uint32_t alignment_bytes, DeviceAddr offset) {
    allocator_ = std::make_unique<allocator::FreeListOpt>(
        size_bytes, offset, alignment_bytes, alignment_bytes, allocator::FreeListOpt::SearchPolicy::FIRST);
}

void validate_num_banks(uint32_t num_banks, const BufferType& buffer_type, bool disable_interleaved) {
    bool doesnt_support_interleaved = buffer_type == BufferType::L1_SMALL or disable_interleaved;
    bool is_pow2_num_banks = num_banks && (!(num_banks & (num_banks - 1)));
    // Dataflow API does not have a working implementation of generic modulo to determine bank_id for interleaved
    // address gen For non pow2 num banks, special cases need to be added to avoid falling back to generic
    // implementation. See https://github.com/tenstorrent/tt-metal/issues/3321
    std::unordered_set<uint32_t> acceptable_num_non_pow2_mem_banks = {
        7, 12, 56, 63, 70, 80, 94, 110, 120, 124, 130, 140};
    bool custom_mod_bank_id_calculation_exists = acceptable_num_non_pow2_mem_banks.count(num_banks) > 0;
    bool valid_num_banks = (is_pow2_num_banks or custom_mod_bank_id_calculation_exists or doesnt_support_interleaved);
    if (not valid_num_banks) {
        TT_THROW(
            "Invalid number of memory banks for {}. Num banks must be power of 2 or have a dedicated modulo "
            "implementation",
            enchantum::to_string(buffer_type),
            num_banks);
    }
}

BankManager::BankManager(
    const BufferType& buffer_type,
    const std::vector<int64_t>& bank_offsets,
    DeviceAddr size_bytes,
    uint32_t alignment_bytes,
    DeviceAddr alloc_offset,
    bool disable_interleaved) :
    buffer_type_(buffer_type), alignment_bytes_(alignment_bytes) {
    unsigned int bank_id = 0;
    for (const auto bank_offset : bank_offsets) {
        bank_id_to_bank_offset_.insert({bank_id, bank_offset});
        bank_id++;
    }
    interleaved_address_limit_ = 0;
    validate_num_banks(bank_id_to_bank_offset_.size(), buffer_type_, disable_interleaved);
    this->init_allocator(size_bytes, MetalContext::instance().hal().get_alignment(HalMemType::DRAM), alloc_offset);
}

BankManager::BankManager(
    const BufferType& buffer_type,
    const std::unordered_map<uint32_t, int64_t>& bank_id_to_bank_offset,
    DeviceAddr size_bytes,
    DeviceAddr interleaved_address_limit,
    uint32_t alignment_bytes,
    DeviceAddr alloc_offset,
    bool disable_interleaved) :
    buffer_type_(buffer_type),
    bank_id_to_bank_offset_(bank_id_to_bank_offset),
    interleaved_address_limit_(interleaved_address_limit),
    alignment_bytes_(alignment_bytes) {
    validate_num_banks(bank_id_to_bank_offset_.size(), buffer_type_, disable_interleaved);
    this->init_allocator(size_bytes, MetalContext::instance().hal().get_alignment(HalMemType::DRAM), alloc_offset);
}

uint32_t BankManager::num_banks() const { return bank_id_to_bank_offset_.size(); }

DeviceAddr BankManager::bank_size() const {
    TT_ASSERT(bool(allocator_), "Allocator not initialized!");
    DeviceAddr max_size_bytes_u64 = allocator_->max_size_bytes();
    if (max_size_bytes_u64 > std::numeric_limits<DeviceAddr>::max()) {
        TT_THROW("Bank size {} overflows DeviceAddr", max_size_bytes_u64);
    }
    DeviceAddr max_size_bytes = (DeviceAddr)max_size_bytes_u64;
    return max_size_bytes;
}

int64_t BankManager::bank_offset(uint32_t bank_id) const {
    this->validate_bank_id(bank_id);
    return bank_id_to_bank_offset_.at(bank_id);
}

void BankManager::validate_bank_id(uint32_t bank_id) const {
    TT_FATAL(
        bank_id_to_bank_offset_.find(bank_id) != bank_id_to_bank_offset_.end(),
        "Expected bank {} to be tracked!",
        bank_id,
        bank_id_to_bank_offset_.size());
}

uint64_t BankManager::allocate_buffer(
    DeviceAddr size,
    DeviceAddr page_size,
    bool bottom_up,
    const CoreRangeSet& compute_grid,
    std::optional<uint32_t> num_shards) {
    uint32_t num_banks = this->num_banks();
    bool is_sharded = false;
    if (num_shards.has_value()) {
        auto num_compute_banks = compute_grid.num_cores();
        is_sharded = true;
        TT_FATAL(
            num_shards.value() <= num_compute_banks,
            "Expected number of shards {} to be less than or equal to total number of L1 banks {} in compute cores",
            num_shards.value(),
            num_compute_banks);
        num_banks = num_shards.value();
    }
    DeviceAddr size_per_bank = tt::tt_metal::detail::SizeBytesPerBank(size, page_size, num_banks, alignment_bytes_);
    DeviceAddr address_limit = 0;
    if (!is_sharded and buffer_type_ == BufferType::L1) {
        address_limit = interleaved_address_limit_;
        TT_FATAL(address_limit > 0, "Address limit {} needs to be larger than zero.", address_limit);
    }
    TT_ASSERT(bool(allocator_), "Allocator not initialized!");
    auto address = allocator_->allocate(size_per_bank, bottom_up, address_limit);
    if (not address.has_value()) {
        TT_THROW(
            "Out of Memory: Not enough space to allocate {} B {} buffer across {} banks, where each bank needs to "
            "store {} B",
            size,
            enchantum::to_string(buffer_type_),
            num_banks,
            size_per_bank);
    }
    allocated_buffers_.insert(address.value());
    return address.value();
}

void BankManager::deallocate_buffer(DeviceAddr address) { allocator_->deallocate(address); }

void BankManager::deallocate_all() {
    for (DeviceAddr addr : allocated_buffers_) {
        allocator_->deallocate(addr);
    }
}

void BankManager::clear() {
    if (allocator_) {
        allocator_->clear();
    }
}

BankManager::~BankManager() {
    deallocate_all();
    allocated_buffers_.clear();
    bank_id_to_bank_offset_.clear();
    allocator_.reset(nullptr);
}

BankManager&& BankManager::operator=(BankManager&& that) noexcept {
    buffer_type_ = that.buffer_type_;
    allocated_buffers_ = that.allocated_buffers_;
    bank_id_to_bank_offset_ = that.bank_id_to_bank_offset_;
    allocator_.reset(that.allocator_.release());
    interleaved_address_limit_ = that.interleaved_address_limit_;
    alignment_bytes_ = that.alignment_bytes_;
    return std::move(*this);
}

std::optional<DeviceAddr> BankManager::lowest_occupied_address(uint32_t bank_id) const {
    if (not allocator_) {
        return std::nullopt;
    }
    auto lowest_address = allocator_->lowest_occupied_address();
    if (not lowest_address.has_value()) {
        return lowest_address;
    }
    DeviceAddr adjusted_abs_addr = lowest_address.value() + this->bank_offset(bank_id);
    return adjusted_abs_addr;
}

Statistics BankManager::get_statistics() const { return allocator_ ? allocator_->get_statistics() : Statistics(); }

void BankManager::dump_blocks(std::ofstream& out) const {
    if (allocator_) {
        allocator_->dump_blocks(out);
    }
}

MemoryBlockTable BankManager::get_memory_block_table() const {
    if (allocator_) {
        return allocator_->get_memory_block_table();
    }

    log_warning(tt::LogAlways, "allocator is not initialized, cannot get block table for memory");
    return {};
}

void BankManager::shrink_size(DeviceAddr shrink_size, bool bottom_up) {
    if (allocator_) {
        allocator_->shrink_size(shrink_size, bottom_up);
    }
}

void BankManager::reset_size() {
    if (allocator_) {
        allocator_->reset_size();
    }
}

}  // namespace tt_metal

}  // namespace tt
