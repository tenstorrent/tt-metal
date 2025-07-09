// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <allocator.hpp>
#include <buffer.hpp>
#include <magic_enum/magic_enum.hpp>
#include <functional>
#include <string>
#include <string_view>

#include "assert.hpp"
#include "buffer_types.hpp"
#include "impl/allocator/bank_manager.hpp"
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/xy_pair.h>

namespace tt {

namespace tt_metal {

Allocator::Allocator(const AllocatorConfig& alloc_config) : config_(alloc_config) {}

void Allocator::validate_bank_assignments() const {
    TT_ASSERT(not bank_id_to_dram_channel_.empty() and not dram_channel_to_bank_ids_.empty());
    TT_ASSERT(dram_channel_to_bank_ids_.size() == config_.num_dram_channels);
    TT_ASSERT(not bank_id_to_logical_core_.empty() and not logical_core_to_bank_ids_.empty());
}

void Allocator::init_one_bank_per_channel() {
    // DRAM bank is between unreserved start and trace_region start: UNRESERVED | DRAM BANK | TRACE REGION
    DeviceAddr dram_bank_size = config_.dram_bank_size - config_.dram_unreserved_base - config_.trace_region_size;
    std::vector<int64_t> bank_offsets(config_.num_dram_channels);
    for (uint32_t channel_id = 0; channel_id < config_.num_dram_channels; channel_id++) {
        bank_offsets.at(channel_id) = static_cast<int32_t>(config_.dram_bank_offsets.at(channel_id));
    }
    dram_manager_ = std::make_unique<BankManager>(
        BufferType::DRAM,
        bank_offsets,
        dram_bank_size,
        config_.dram_alignment,
        config_.dram_unreserved_base,
        config_.disable_interleaved);
    for (uint32_t bank_id = 0; bank_id < config_.num_dram_channels; bank_id++) {
        CoreCoord logical_core = CoreCoord{bank_id, 0};
        bank_id_to_dram_channel_.insert({bank_id, bank_id});
        dram_channel_to_bank_ids_.insert({bank_id, {bank_id}});
        logical_core_to_bank_ids_[BufferType::DRAM].insert({logical_core, {bank_id}});
    }
    // Trace buffers are allocated in this region (top-down). Trace region is offset at dram_bank_size + UNRESERVED
    // offset
    trace_buffer_manager_ = std::make_unique<BankManager>(
        BufferType::TRACE,
        bank_offsets,
        config_.trace_region_size,
        config_.dram_alignment,
        dram_bank_size + config_.dram_unreserved_base,
        config_.disable_interleaved);
    for (uint32_t bank_id = 0; bank_id < config_.num_dram_channels; bank_id++) {
        CoreCoord logical_core = CoreCoord{bank_id, 0};
        bank_id_to_dram_channel_.insert({bank_id, bank_id});
        dram_channel_to_bank_ids_.insert({bank_id, {bank_id}});
        logical_core_to_bank_ids_[BufferType::TRACE].insert({logical_core, {bank_id}});
    }
}

void Allocator::init_one_bank_per_l1() {
    TT_ASSERT(config_.l1_small_size == 0);
    uint32_t num_l1_banks = config_.worker_grid.num_cores();
    // Space up to L1 unreserved base is reserved for risc binaries, kernel args, debug and perf monitoring tools
    DeviceAddr l1_bank_size = config_.worker_l1_size - config_.l1_unreserved_base;
    std::vector<int64_t> bank_offsets(num_l1_banks, 0);
    l1_manager_ = std::make_unique<BankManager>(
        BufferType::L1,
        bank_offsets,
        l1_bank_size,
        config_.l1_alignment,
        config_.l1_unreserved_base,
        config_.disable_interleaved);

    uint32_t bank_id = 0;
    const auto& cores = corerange_to_cores(config_.worker_grid, std::nullopt, true);
    for (const auto& logical_core : cores) {
        bank_id_to_logical_core_.insert({bank_id, logical_core});
        logical_core_to_bank_ids_[BufferType::L1].insert({logical_core, {bank_id}});
        bank_id++;
    }
}

void Allocator::verify_safe_allocation() const {
    // Inform the user that its unsafe to allocate buffers when a trace is live on device.
    // If the user does this, they are meant to ensure that buffers allocated when a trace is active,
    // have a lifetime that ends before the trace is executed.
    // Print the warning once per device, to ensure that user output is not clobbered.
    thread_local static bool warning_generated = false;
    if (allocations_unsafe_ and not warning_generated) {
        log_warning(
            tt::LogMetal,
            "Allocating device buffers is unsafe due to the existence of an active trace. These buffers may be "
            "corrupted once a trace is executed.");
        warning_generated = true;
    }
}

DeviceAddr Allocator::allocate_buffer(Buffer* buffer) {
    DeviceAddr address = 0;
    auto size = buffer->aligned_size();
    auto page_size = buffer->aligned_page_size();
    auto buffer_type = buffer->buffer_type();
    auto bottom_up = buffer->bottom_up();
    auto num_cores = buffer->num_cores();
    this->verify_safe_allocation();
    if (config_.disable_interleaved) {
        TT_FATAL(num_cores.has_value(), "Interleaved allocation is disabled, see validate_num_banks");
    }
    switch (buffer_type) {
        case BufferType::DRAM:
            address = dram_manager_->allocate_buffer(size, page_size, bottom_up, config_.compute_grid, num_cores);
            break;
        case BufferType::L1:
            address = l1_manager_->allocate_buffer(size, page_size, bottom_up, config_.compute_grid, num_cores);
            break;
        case BufferType::L1_SMALL: {
            TT_FATAL(num_cores.has_value(), "L1_SMALL only supports sharded allocations, see validate_num_banks");
            address = l1_small_manager_->allocate_buffer(size, page_size, bottom_up, config_.compute_grid, num_cores);
            break;
        }
        case BufferType::TRACE:
            address =
                trace_buffer_manager_->allocate_buffer(size, page_size, bottom_up, config_.compute_grid, num_cores);
            break;
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
    allocated_buffers_.insert(buffer);
    return address;
}

void Allocator::deallocate_buffer(Buffer* buffer) {
    auto address = buffer->address();
    auto buffer_type = buffer->buffer_type();
    switch (buffer_type) {
        case BufferType::DRAM: dram_manager_->deallocate_buffer(address); break;
        case BufferType::L1: l1_manager_->deallocate_buffer(address); break;
        case BufferType::L1_SMALL: l1_small_manager_->deallocate_buffer(address); break;
        case BufferType::TRACE: trace_buffer_manager_->deallocate_buffer(address); break;
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
    allocated_buffers_.erase(buffer);
}

void Allocator::deallocate_buffers() {
    dram_manager_->deallocate_all();
    l1_manager_->deallocate_all();
    l1_small_manager_->deallocate_all();
    trace_buffer_manager_->deallocate_all();
}

const std::unordered_set<Buffer*>& Allocator::get_allocated_buffers() const { return allocated_buffers_; }

uint32_t Allocator::get_num_banks(const BufferType& buffer_type) const {
    switch (buffer_type) {
        case BufferType::DRAM: return dram_manager_->num_banks();
        case BufferType::L1: return l1_manager_->num_banks();
        case BufferType::L1_SMALL: return l1_small_manager_->num_banks();
        case BufferType::TRACE: return trace_buffer_manager_->num_banks();
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
    return 0;
}

DeviceAddr Allocator::get_bank_size(const BufferType& buffer_type) const {
    switch (buffer_type) {
        case BufferType::DRAM: return dram_manager_->bank_size();
        case BufferType::L1: return l1_manager_->bank_size();
        case BufferType::L1_SMALL: return l1_small_manager_->bank_size();
        case BufferType::TRACE: return trace_buffer_manager_->bank_size();
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
    return 0;
}

uint32_t Allocator::get_dram_channel_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(bank_id_to_dram_channel_.find(bank_id) != bank_id_to_dram_channel_.end());
    return bank_id_to_dram_channel_.at(bank_id);
}

CoreCoord Allocator::get_logical_core_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(bank_id_to_logical_core_.find(bank_id) != bank_id_to_logical_core_.end());
    return bank_id_to_logical_core_.at(bank_id);
}

int32_t Allocator::get_bank_offset(BufferType buffer_type, uint32_t bank_id) const {
    switch (buffer_type) {
        case BufferType::DRAM: return dram_manager_->bank_offset(bank_id);
        case BufferType::L1: return l1_manager_->bank_offset(bank_id);
        case BufferType::L1_SMALL: return l1_small_manager_->bank_offset(bank_id);
        case BufferType::TRACE: return trace_buffer_manager_->bank_offset(bank_id);
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
}

const std::vector<uint32_t>& Allocator::get_bank_ids_from_dram_channel(uint32_t dram_channel) const {
    if (dram_channel_to_bank_ids_.find(dram_channel) == dram_channel_to_bank_ids_.end()) {
        TT_THROW("No DRAM bank exists for DRAM channel {}", dram_channel);
    }
    return dram_channel_to_bank_ids_.at(dram_channel);
}

const std::vector<uint32_t>& Allocator::get_bank_ids_from_logical_core(
    BufferType buffer_type, const CoreCoord& logical_core) const {
    if (logical_core_to_bank_ids_.at(buffer_type).find(logical_core) ==
        logical_core_to_bank_ids_.at(buffer_type).end()) {
        TT_THROW("No {} bank exists for core {}", magic_enum::enum_name(buffer_type), logical_core.str());
    }
    return logical_core_to_bank_ids_.at(buffer_type).at(logical_core);
}

const AllocatorConfig& Allocator::get_config() const { return config_; }

uint32_t Allocator::get_alignment(BufferType buffer_type) const {
    switch (buffer_type) {
        case BufferType::DRAM:
        case BufferType::TRACE: return config_.dram_alignment;
        case BufferType::L1:
        case BufferType::L1_SMALL: return config_.l1_alignment;
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
}

DeviceAddr Allocator::get_base_allocator_addr(const HalMemType& mem_type) const {
    switch (mem_type) {
        case HalMemType::DRAM: return config_.dram_unreserved_base;
        case HalMemType::L1: return config_.l1_unreserved_base;
        default: {
            TT_THROW("Allocator does not support allocating in {}", magic_enum::enum_name(mem_type));
        }
    }
    return 0;
}

Statistics Allocator::get_statistics(const BufferType& buffer_type) const {
    Statistics stats;
    switch (buffer_type) {
        case BufferType::DRAM: return dram_manager_->get_statistics();
        case BufferType::L1: return l1_manager_->get_statistics();
        case BufferType::L1_SMALL: return l1_small_manager_->get_statistics();
        case BufferType::TRACE: return trace_buffer_manager_->get_statistics();
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
    return stats;
}

MemoryBlockTable Allocator::get_memory_block_table(const BufferType& buffer_type) const {
    switch (buffer_type) {
        case BufferType::DRAM: return dram_manager_->get_memory_block_table();
        case BufferType::L1: return l1_manager_->get_memory_block_table();
        case BufferType::L1_SMALL: return l1_small_manager_->get_memory_block_table();
        case BufferType::TRACE: return trace_buffer_manager_->get_memory_block_table();
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
}

void Allocator::dump_memory_blocks(const BufferType& buffer_type, std::ofstream& out) const {
    switch (buffer_type) {
        case BufferType::DRAM: dram_manager_->dump_blocks(out); break;
        case BufferType::L1: l1_manager_->dump_blocks(out); break;
        case BufferType::L1_SMALL: l1_small_manager_->dump_blocks(out); break;
        case BufferType::TRACE: trace_buffer_manager_->dump_blocks(out); break;
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
}

std::optional<DeviceAddr> Allocator::get_lowest_occupied_l1_address(uint32_t bank_id) const {
    // l1_manager always sits below l1_small_manager in the address space, so there is no need to check l1_small_manager
    return l1_manager_->lowest_occupied_address(bank_id);
}

void Allocator::shrink_allocator_size(const BufferType& buffer_type, DeviceAddr shrink_size, bool bottom_up) {
    switch (buffer_type) {
        case BufferType::DRAM: dram_manager_->shrink_size(shrink_size, bottom_up); break;
        case BufferType::L1: l1_manager_->shrink_size(shrink_size, bottom_up); break;
        case BufferType::L1_SMALL: l1_small_manager_->shrink_size(shrink_size, bottom_up); break;
        case BufferType::TRACE: trace_buffer_manager_->shrink_size(shrink_size, bottom_up); break;
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
}

void Allocator::reset_allocator_size(const BufferType& buffer_type) {
    switch (buffer_type) {
        case BufferType::DRAM: dram_manager_->reset_size(); break;
        case BufferType::L1: l1_manager_->reset_size(); break;
        case BufferType::L1_SMALL: l1_small_manager_->reset_size(); break;
        case BufferType::TRACE: trace_buffer_manager_->reset_size(); break;
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
}

void Allocator::mark_allocations_unsafe() { allocations_unsafe_ = true; }

void Allocator::mark_allocations_safe() { allocations_unsafe_ = false; }

void Allocator::clear() {
    dram_manager_->clear();
    l1_manager_->clear();
    l1_small_manager_->clear();
    trace_buffer_manager_->clear();
}

void AllocatorConfig::reset() {
    dram_bank_offsets.clear();
    core_type_from_noc_coord_table.clear();
    worker_log_to_virtual_routing_x.clear();
    worker_log_to_virtual_routing_y.clear();
    l1_bank_remap.clear();
}

Allocator::~Allocator() {
    bank_id_to_dram_channel_.clear();
    dram_channel_to_bank_ids_.clear();
    bank_id_to_logical_core_.clear();
    for (auto& [buffer_type, submap] : logical_core_to_bank_ids_) {
        submap.clear();
    }

    dram_manager_->clear();
    l1_manager_->clear();
    l1_small_manager_->clear();
    trace_buffer_manager_->clear();
    allocated_buffers_.clear();
    config_.reset();
}

}  // namespace tt_metal

}  // namespace tt
