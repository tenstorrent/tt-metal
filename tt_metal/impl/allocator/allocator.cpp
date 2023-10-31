// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/allocator/algorithms/free_list.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "third_party/magic_enum/magic_enum.hpp"

namespace tt {

namespace tt_metal {

namespace allocator {

void BankManager::init_allocator(u64 size_bytes, u64 offset) {
    this->allocator_ = std::make_unique<FreeList>(
        size_bytes,
        offset,
        this->min_allocation_size_bytes_,
        ADDRESS_ALIGNMENT,
        FreeList::SearchPolicy::FIRST
    );
}

BankManager::BankManager(const BufferStorage &buffer_storage, const std::vector<i64> &bank_offsets, u64 size_bytes, u64 alloc_offset) : buffer_storage_(buffer_storage) {
    unsigned int bank_id = 0;
    for (const auto bank_offset : bank_offsets) {
        this->bank_id_to_bank_offset_.insert({bank_id, bank_offset});
        bank_id++;
    }
    this->init_allocator(size_bytes, alloc_offset);
}

BankManager::BankManager(const BufferStorage &buffer_storage, const std::unordered_map<u32, i64> &bank_id_to_bank_offset, u64 size_bytes, u64 alloc_offset) : buffer_storage_(buffer_storage), bank_id_to_bank_offset_(bank_id_to_bank_offset) {
    this->init_allocator(size_bytes, alloc_offset);
}

u32 BankManager::num_banks() const {
    return this->bank_id_to_bank_offset_.size();
}

u32 BankManager::bank_size() const {
    uint64_t max_size_bytes_u64 = this->allocator_->max_size_bytes();
    if (max_size_bytes_u64 > std::numeric_limits<uint32_t>::max()) {
        tt::log_fatal(tt::LogMetal, "Bank size {} overflows uint32_t", max_size_bytes_u64);
    }
    uint32_t max_size_bytes = (uint32_t)max_size_bytes_u64;
    return max_size_bytes;
}

i64 BankManager::bank_offset(u32 bank_id) const {
    this->validate_bank_id(bank_id);
    return this->bank_id_to_bank_offset_.at(bank_id);
}

void BankManager::validate_bank_id(u32 bank_id) const {
    log_assert(this->bank_id_to_bank_offset_.find(bank_id) != this->bank_id_to_bank_offset_.end(), "Expected bank {} to be tracked!", bank_id);
}

u64 BankManager::allocate_buffer(u32 size, u32 page_size, bool bottom_up) {
    uint32_t num_banks = this->num_banks();
    // Each page needs to be at a 32B aligned address
    uint32_t size_per_bank = tt::tt_metal::detail::SizeBytesPerBank(size, page_size, num_banks);

    auto address = this->allocator_->allocate(size_per_bank, bottom_up);
    if (not address.has_value()) {
        log_fatal(tt::LogMetal, "Out of Memory: Not enough space to allocate {} B {} buffer across {} banks, where each bank needs to store {} B", size, magic_enum::enum_name(this->buffer_storage_), num_banks, size_per_bank);
    }
    allocated_buffers_.insert(address.value());
    return address.value();
}

void BankManager::deallocate_buffer(u64 address) {
    this->allocator_->deallocate(address);
}

void BankManager::deallocate_all(){
    for (u64 addr : this->allocated_buffers_)
    {
        this->allocator_->deallocate(addr);
    }
}


void BankManager::clear() {
    this->allocator_->clear();
}

std::optional<u64> BankManager::lowest_occupied_address(u32 bank_id) const {
    auto lowest_address = this->allocator_->lowest_occupied_address();
    if (not lowest_address.has_value()) {
        return lowest_address;
    }
    auto adjusted_abs_addr = lowest_address.value() + this->bank_offset(bank_id);
    return adjusted_abs_addr;
}

Statistics BankManager::get_statistics() const {
    return this->allocator_->get_statistics();
}

void BankManager::dump_blocks(std::ofstream &out) const {
    this->allocator_->dump_blocks(out);
}

void init_one_bank_per_channel(Allocator &allocator, const AllocatorConfig &alloc_config) {
    // Space up to DRAM_UNRESERVED_BASE is reserved for DRAM write barrier
    u64 offset_bytes = static_cast<u64>(DRAM_UNRESERVED_BASE);
    u32 dram_bank_size = alloc_config.dram_bank_size - DRAM_UNRESERVED_BASE;
    std::vector<i64> bank_offsets (alloc_config.num_dram_channels);
    for (u32 channel_id = 0; channel_id < alloc_config.num_dram_channels; channel_id++) {
        bank_offsets.at(channel_id) = static_cast<i32>(alloc_config.dram_bank_offsets.at(channel_id));
    }
    allocator.dram_manager = BankManager(BufferStorage::DRAM, bank_offsets, dram_bank_size, offset_bytes);
    for (u32 bank_id = 0; bank_id < alloc_config.num_dram_channels; bank_id++) {
        allocator.bank_id_to_dram_channel.insert({bank_id, bank_id});
        allocator.dram_channel_to_bank_ids.insert({bank_id, {bank_id}});
    }
}

void init_one_bank_per_l1(Allocator &allocator, const AllocatorConfig &alloc_config) {
    u32 num_l1_banks = alloc_config.worker_grid_size.y * alloc_config.worker_grid_size.x;
    // Space up to L1_UNRESERVED_BASE is reserved for risc binaries, kernel args, debug and perf monitoring tools
    u64 offset_bytes = static_cast<u64>(L1_UNRESERVED_BASE);
    u32 l1_bank_size = alloc_config.worker_l1_size - L1_UNRESERVED_BASE;
    std::vector<i64> bank_offsets (num_l1_banks, 0);
    allocator.l1_manager = BankManager(BufferStorage::L1, bank_offsets, l1_bank_size, offset_bytes);

    u32 bank_id = 0;
    for (u32 y = 0; y < alloc_config.worker_grid_size.y; y++) {
        for (u32 x = 0; x < alloc_config.worker_grid_size.x; x++) {
            CoreCoord logical_core = CoreCoord{x, y};
            allocator.bank_id_to_logical_core.insert({bank_id, logical_core});
            allocator.logical_core_to_bank_ids.insert({logical_core, {bank_id}});
            bank_id++;
        }
    }
}

u32 num_banks(const Allocator &allocator, const BufferStorage &buffer_storage) {
    switch (buffer_storage) {
        case BufferStorage::DRAM: return allocator.dram_manager.num_banks();
        case BufferStorage::L1: return allocator.l1_manager.num_banks();
        default: {
            TT_ASSERT(false && "Unsupported buffer type!");
        }
    }
    return 0;
}

u32 bank_size(const Allocator &allocator, const BufferStorage &buffer_storage) {
    switch (buffer_storage) {
        case BufferStorage::DRAM: return allocator.dram_manager.bank_size();
        case BufferStorage::L1: return allocator.l1_manager.bank_size();
        default: {
            log_fatal(tt::LogMetal, "Unsupported buffer type!");
        }
    }
    return 0;
}

u32 dram_channel_from_bank_id(const Allocator &allocator, u32 bank_id) {
    TT_ASSERT(allocator.bank_id_to_dram_channel.find(bank_id) != allocator.bank_id_to_dram_channel.end());
    return allocator.bank_id_to_dram_channel.at(bank_id);
}

CoreCoord logical_core_from_bank_id(const Allocator &allocator, u32 bank_id) {
    TT_ASSERT(allocator.bank_id_to_logical_core.find(bank_id) != allocator.bank_id_to_logical_core.end());
    return allocator.bank_id_to_logical_core.at(bank_id);
}

i32 l1_bank_offset_from_bank_id(const Allocator &allocator, u32 bank_id) {
    return allocator.l1_manager.bank_offset(bank_id);
}

i32 dram_bank_offset_from_bank_id(const Allocator &allocator, u32 bank_id) {
    return allocator.dram_manager.bank_offset(bank_id);
}

std::vector<u32> bank_ids_from_dram_channel(const Allocator &allocator, u32 dram_channel) {
    TT_ASSERT(allocator.dram_channel_to_bank_ids.find(dram_channel) != allocator.dram_channel_to_bank_ids.end());
    return allocator.dram_channel_to_bank_ids.at(dram_channel);
}

std::vector<u32> bank_ids_from_logical_core(const Allocator &allocator, const CoreCoord &logical_core) {
    TT_ASSERT(allocator.logical_core_to_bank_ids.find(logical_core) != allocator.logical_core_to_bank_ids.end());
    return allocator.logical_core_to_bank_ids.at(logical_core);
}

Statistics get_statistics(const Allocator &allocator, const BufferStorage &buffer_storage) {
    Statistics stats;
    switch (buffer_storage) {
        case BufferStorage::DRAM: return allocator.dram_manager.get_statistics();
        case BufferStorage::L1: return allocator.l1_manager.get_statistics();
        default: {
            log_assert(false, "Unsupported buffer storage!");
        }
    }
    return stats;
}

void dump_memory_blocks(const Allocator &allocator, const BufferStorage &buffer_storage, std::ofstream &out) {
    switch (buffer_storage) {
        case BufferStorage::DRAM: allocator.dram_manager.dump_blocks(out);
        break;
        case BufferStorage::L1: allocator.l1_manager.dump_blocks(out);
        break;
        default: {
            log_assert(false, "Unsupported buffer storage!");
        }
    }
}

std::optional<u64> lowest_occupied_l1_address(const Allocator &allocator, u32 bank_id) {
    return allocator.l1_manager.lowest_occupied_address(bank_id);
}

u64 base_alloc(const AllocatorConfig &config, BankManager &bank_manager, u64 size, u64 page_size, bool bottom_up) {
    return bank_manager.allocate_buffer(size, page_size, bottom_up);
}

u64 allocate_buffer(Allocator &allocator, u32 size, u32 page_size, const BufferStorage &buffer_storage, bool bottom_up) {
    u64 address = 0;
    switch (buffer_storage) {
        case BufferStorage::DRAM: return allocator.descriptor.dram.alloc(allocator.config, allocator.dram_manager, size, page_size, bottom_up);
        case BufferStorage::L1: return allocator.descriptor.l1.alloc(allocator.config, allocator.l1_manager, size, page_size, bottom_up);
        default: {
            TT_ASSERT(false && "Unsupported buffer type!");
        }
    }
    return address;
}

void deallocate_buffer(Allocator &allocator, u64 address, const BufferStorage &buffer_storage) {
    switch (buffer_storage) {
        case BufferStorage::DRAM:
            allocator.dram_manager.deallocate_buffer(address);
        break;
        case BufferStorage::L1:
            allocator.l1_manager.deallocate_buffer(address);
        break;
        default: {
            TT_ASSERT(false && "Unsupported buffer type!");
        }
    }
}

void deallocate_buffers(Allocator &allocator) {
    allocator.dram_manager.deallocate_all();
    allocator.l1_manager.deallocate_all();
}

void clear(Allocator &allocator) {
    allocator.dram_manager.clear();
    allocator.l1_manager.clear();
}

}  // namespace allocator

Allocator::Allocator(const AllocatorConfig &alloc_config, const allocator::AllocDescriptor &alloc_descriptor) : config(alloc_config), descriptor(alloc_descriptor) {
    // TODO: add validation for allocator_descriptor?
    this->descriptor.dram.init(*this, alloc_config);
    this->descriptor.l1.init(*this, alloc_config);
    // assert that bank managers have been initialized?
    TT_ASSERT(not bank_id_to_dram_channel.empty() and not dram_channel_to_bank_ids.empty());
    TT_ASSERT(not bank_id_to_logical_core.empty() and not bank_id_to_logical_core.empty());
}

}  // namespace tt_metal

}  // namespace tt
