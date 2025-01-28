// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <allocator.hpp>

#include <magic_enum/magic_enum.hpp>
#include <math.hpp>
#include <util.hpp>
#include "tt_metal/impl/allocator/algorithms/free_list_opt.hpp"
#include <buffer.hpp>

namespace tt {

namespace tt_metal {

namespace allocator {
#if defined(TRACY_ENABLE)
static char const* get_memory_pool_name(BufferType buffer_type) {
    switch (buffer_type) {
        case BufferType::DRAM: return "DRAM";
        case BufferType::L1: return "L1";
        case BufferType::SYSTEM_MEMORY: return "SYSTEM_MEMORY";
        default: return "UNKNOWN";
    }
}
#endif

void BankManager::init_allocator(DeviceAddr size_bytes, uint32_t alignment_bytes, DeviceAddr offset) {
    this->allocator_ = std::make_unique<FreeListOpt>(
        size_bytes, offset, alignment_bytes, alignment_bytes, FreeListOpt::SearchPolicy::FIRST);
}

void validate_num_banks(uint32_t num_banks, const BufferType& buffer_type, bool disable_interleaved) {
    bool doesnt_support_interleaved = buffer_type == BufferType::L1_SMALL or disable_interleaved;
    bool is_pow2_num_banks = num_banks && (!(num_banks & (num_banks - 1)));
    // Dataflow API does not have a working implementation of generic modulo to determine bank_id for interleaved
    // address gen For non pow2 num banks, special cases need to be added to avoid falling back to generic
    // implementation. See https://github.com/tenstorrent/tt-metal/issues/3321
    std::unordered_set<uint32_t> acceptable_num_non_pow2_mem_banks = {12, 56, 63, 70, 80, 94, 124, 130, 140};
    bool custom_mod_bank_id_calculation_exists = acceptable_num_non_pow2_mem_banks.count(num_banks) > 0;
    bool valid_num_banks = (is_pow2_num_banks or custom_mod_bank_id_calculation_exists or doesnt_support_interleaved);
    if (not valid_num_banks) {
        TT_THROW(
            "Invalid number of memory banks for {}. Num banks must be power of 2 or have a dedicated modulo "
            "implementation",
            magic_enum::enum_name(buffer_type),
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
        this->bank_id_to_bank_offset_.insert({bank_id, bank_offset});
        bank_id++;
    }
    this->interleaved_address_limit_ = 0;
    validate_num_banks(this->bank_id_to_bank_offset_.size(), this->buffer_type_, disable_interleaved);
    this->init_allocator(size_bytes, alignment_bytes, alloc_offset);
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
    validate_num_banks(this->bank_id_to_bank_offset_.size(), this->buffer_type_, disable_interleaved);
    this->init_allocator(size_bytes, alignment_bytes, alloc_offset);
}

uint32_t BankManager::num_banks() const { return this->bank_id_to_bank_offset_.size(); }

DeviceAddr BankManager::bank_size() const {
    TT_ASSERT(bool(this->allocator_), "Allocator not initialized!");
    DeviceAddr max_size_bytes_u64 = this->allocator_->max_size_bytes();
    if (max_size_bytes_u64 > std::numeric_limits<DeviceAddr>::max()) {
        TT_THROW("Bank size {} overflows DeviceAddr", max_size_bytes_u64);
    }
    DeviceAddr max_size_bytes = (DeviceAddr)max_size_bytes_u64;
    return max_size_bytes;
}

int64_t BankManager::bank_offset(uint32_t bank_id) const {
    this->validate_bank_id(bank_id);
    return this->bank_id_to_bank_offset_.at(bank_id);
}

void BankManager::validate_bank_id(uint32_t bank_id) const {
    TT_FATAL(
        this->bank_id_to_bank_offset_.find(bank_id) != this->bank_id_to_bank_offset_.end(),
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
    DeviceAddr size_per_bank =
        tt::tt_metal::detail::SizeBytesPerBank(size, page_size, num_banks, this->alignment_bytes_);
    DeviceAddr address_limit = 0;
    if (!is_sharded and this->buffer_type_ == BufferType::L1) {
        address_limit = this->interleaved_address_limit_;
        TT_FATAL(address_limit > 0, "Address limit {} needs to be larger than zero.", address_limit);
    }
    TT_ASSERT(bool(this->allocator_), "Allocator not initialized!");
    auto address = this->allocator_->allocate(size_per_bank, bottom_up, address_limit);
    if (not address.has_value()) {
        TT_THROW(
            "Out of Memory: Not enough space to allocate {} B {} buffer across {} banks, where each bank needs to "
            "store {} B",
            size,
            magic_enum::enum_name(this->buffer_type_),
            num_banks,
            size_per_bank);
    }
    allocated_buffers_.insert(address.value());
    return address.value();
}

void BankManager::deallocate_buffer(DeviceAddr address) { this->allocator_->deallocate(address); }

void BankManager::deallocate_all() {
    for (DeviceAddr addr : this->allocated_buffers_) {
        this->allocator_->deallocate(addr);
    }
}

void BankManager::clear() {
    if (this->allocator_) {
        this->allocator_->clear();
    }
}

BankManager::~BankManager() {
    deallocate_all();
    allocated_buffers_.clear();
    bank_id_to_bank_offset_.clear();
    this->allocator_.reset(nullptr);
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
    if (not this->allocator_) {
        return std::nullopt;
    }
    auto lowest_address = this->allocator_->lowest_occupied_address();
    if (not lowest_address.has_value()) {
        return lowest_address;
    }
    DeviceAddr adjusted_abs_addr = lowest_address.value() + this->bank_offset(bank_id);
    return adjusted_abs_addr;
}

Statistics BankManager::get_statistics() const {
    return this->allocator_ ? this->allocator_->get_statistics() : Statistics();
}

void BankManager::dump_blocks(std::ofstream& out) const {
    if (this->allocator_) {
        this->allocator_->dump_blocks(out);
    }
}

MemoryBlockTable BankManager::get_memory_block_table() const {
    if (this->allocator_) {
        return this->allocator_->get_memory_block_table();
    }

    log_warning("allocator is not initialized, cannot get block table for memory");
    return {};
}

void BankManager::shrink_size(DeviceAddr shrink_size, bool bottom_up) {
    if (this->allocator_) {
        this->allocator_->shrink_size(shrink_size, bottom_up);
    }
}

void BankManager::reset_size() {
    if (this->allocator_) {
        this->allocator_->reset_size();
    }
}

}  // namespace allocator

Allocator::Allocator(const AllocatorConfig& alloc_config) : config_(alloc_config) {
    TT_ASSERT(not bank_id_to_dram_channel_.empty() and not dram_channel_to_bank_ids_.empty());
    TT_ASSERT(not bank_id_to_logical_core_.empty() and not bank_id_to_logical_core_.empty());
}

void Allocator::init_one_bank_per_channel() {
    // DRAM bank is between unreserved start and trace_region start: UNRESERVED | DRAM BANK | TRACE REGION
    DeviceAddr dram_bank_size = config_.dram_bank_size - config_.dram_unreserved_base - config_.trace_region_size;
    std::vector<int64_t> bank_offsets(config_.num_dram_channels);
    for (uint32_t channel_id = 0; channel_id < config_.num_dram_channels; channel_id++) {
        bank_offsets.at(channel_id) = static_cast<int32_t>(config_.dram_bank_offsets.at(channel_id));
    }
    dram_manager_ = allocator::BankManager(
        BufferType::DRAM,
        bank_offsets,
        dram_bank_size,
        config_.alignment,
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
    trace_buffer_manager_ = allocator::BankManager(
        BufferType::TRACE,
        bank_offsets,
        config_.trace_region_size,
        config_.alignment,
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
    l1_manager_ = allocator::BankManager(
        BufferType::L1,
        bank_offsets,
        l1_bank_size,
        config_.alignment,
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
    auto num_shards = buffer->num_cores();
    this->verify_safe_allocation();
    if (config_.disable_interleaved) {
        TT_FATAL(num_shards.has_value(), "Interleaved allocation is disabled, see validate_num_banks");
    }
    switch (buffer_type) {
        case BufferType::DRAM:
            address = dram_manager_.allocate_buffer(size, page_size, bottom_up, config_.compute_grid, num_shards);
            break;
        case BufferType::L1:
            address = l1_manager_.allocate_buffer(size, page_size, bottom_up, config_.compute_grid, num_shards);
            break;
        case BufferType::L1_SMALL: {
            TT_FATAL(num_shards.has_value(), "L1_SMALL only supports sharded allocations, see validate_num_banks");
            address = l1_small_manager_.allocate_buffer(size, page_size, bottom_up, config_.compute_grid, num_shards);
            break;
        }
        case BufferType::TRACE:
            address =
                trace_buffer_manager_.allocate_buffer(size, page_size, bottom_up, config_.compute_grid, num_shards);
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
        case BufferType::DRAM: dram_manager_.deallocate_buffer(address); break;
        case BufferType::L1: l1_manager_.deallocate_buffer(address); break;
        case BufferType::L1_SMALL: l1_small_manager_.deallocate_buffer(address); break;
        case BufferType::TRACE: trace_buffer_manager_.deallocate_buffer(address); break;
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
    allocated_buffers_.erase(buffer);
}

void Allocator::deallocate_buffers() {
    dram_manager_.deallocate_all();
    l1_manager_.deallocate_all();
    l1_small_manager_.deallocate_all();
    trace_buffer_manager_.deallocate_all();
}

const std::unordered_set<Buffer*>& Allocator::get_allocated_buffers() const { return allocated_buffers_; }

uint32_t Allocator::get_num_banks(const BufferType& buffer_type) const {
    switch (buffer_type) {
        case BufferType::DRAM: return dram_manager_.num_banks();
        case BufferType::L1: return l1_manager_.num_banks();
        case BufferType::L1_SMALL: return l1_small_manager_.num_banks();
        case BufferType::TRACE: return trace_buffer_manager_.num_banks();
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
    return 0;
}

DeviceAddr Allocator::get_bank_size(const BufferType& buffer_type) const {
    switch (buffer_type) {
        case BufferType::DRAM: return dram_manager_.bank_size();
        case BufferType::L1: return l1_manager_.bank_size();
        case BufferType::L1_SMALL: return l1_small_manager_.bank_size();
        case BufferType::TRACE: return trace_buffer_manager_.bank_size();
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
        case BufferType::DRAM: return dram_manager_.bank_offset(bank_id);
        case BufferType::L1: return l1_manager_.bank_offset(bank_id);
        case BufferType::L1_SMALL: return l1_small_manager_.bank_offset(bank_id);
        case BufferType::TRACE: return trace_buffer_manager_.bank_offset(bank_id);
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

DeviceAddr Allocator::get_unreserved_base_address(const HalMemType& mem_type) const {
    switch (mem_type) {
        case HalMemType::DRAM: return config_.dram_unreserved_base;
        case HalMemType::L1: return config_.l1_unreserved_base;
        default: {
            TT_THROW("Allocator does not support allocating in {}", magic_enum::enum_name(mem_type));
        }
    }
    return 0;
}

allocator::Statistics Allocator::get_statistics(const BufferType& buffer_type) const {
    allocator::Statistics stats;
    switch (buffer_type) {
        case BufferType::DRAM: return dram_manager_.get_statistics();
        case BufferType::L1: return l1_manager_.get_statistics();
        case BufferType::L1_SMALL: return l1_small_manager_.get_statistics();
        case BufferType::TRACE: return trace_buffer_manager_.get_statistics();
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
    return stats;
}

MemoryBlockTable Allocator::get_memory_block_table(const BufferType& buffer_type) const {
    switch (buffer_type) {
        case BufferType::DRAM: return dram_manager_.get_memory_block_table();
        case BufferType::L1: return l1_manager_.get_memory_block_table();
        case BufferType::L1_SMALL: return l1_small_manager_.get_memory_block_table();
        case BufferType::TRACE: return trace_buffer_manager_.get_memory_block_table();
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
}

void Allocator::dump_memory_blocks(const BufferType& buffer_type, std::ofstream& out) const {
    switch (buffer_type) {
        case BufferType::DRAM: dram_manager_.dump_blocks(out); break;
        case BufferType::L1: l1_manager_.dump_blocks(out); break;
        case BufferType::L1_SMALL: l1_small_manager_.dump_blocks(out); break;
        case BufferType::TRACE: trace_buffer_manager_.dump_blocks(out); break;
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
}

std::optional<DeviceAddr> Allocator::get_lowest_occupied_l1_address(uint32_t bank_id) const {
    // l1_manager always sits below l1_small_manager in the address space, so there is no need to check l1_small_manager
    return l1_manager_.lowest_occupied_address(bank_id);
}

void Allocator::shrink_allocator_size(const BufferType& buffer_type, DeviceAddr shrink_size, bool bottom_up) {
    switch (buffer_type) {
        case BufferType::DRAM: dram_manager_.shrink_size(shrink_size, bottom_up); break;
        case BufferType::L1: l1_manager_.shrink_size(shrink_size, bottom_up); break;
        case BufferType::L1_SMALL: l1_small_manager_.shrink_size(shrink_size, bottom_up); break;
        case BufferType::TRACE: trace_buffer_manager_.shrink_size(shrink_size, bottom_up); break;
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
}

void Allocator::reset_allocator_size(const BufferType& buffer_type) {
    switch (buffer_type) {
        case BufferType::DRAM: dram_manager_.reset_size(); break;
        case BufferType::L1: l1_manager_.reset_size(); break;
        case BufferType::L1_SMALL: l1_small_manager_.reset_size(); break;
        case BufferType::TRACE: trace_buffer_manager_.reset_size(); break;
        default: {
            TT_THROW("Unsupported buffer type!");
        }
    }
}

void Allocator::mark_allocations_unsafe() { allocations_unsafe_ = true; }

void Allocator::mark_allocations_safe() { allocations_unsafe_ = false; }

void Allocator::clear() {
    dram_manager_.clear();
    l1_manager_.clear();
    l1_small_manager_.clear();
    trace_buffer_manager_.clear();
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

    dram_manager_.clear();
    l1_manager_.clear();
    l1_small_manager_.clear();
    trace_buffer_manager_.clear();
    allocated_buffers_.clear();
    config_.reset();
}

}  // namespace tt_metal

}  // namespace tt
