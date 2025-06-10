// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <allocator.hpp>
#include <assert.hpp>
#include <buffer.hpp>
#include <buffer_types.hpp>
#include <device.hpp>
#include <graph_tracking.hpp>
#include <magic_enum/magic_enum.hpp>
#include <math.hpp>
#include <nlohmann/json.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/overloaded.hpp>
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include "fmt/base.h"
#include "lightmetal/host_api_capture_helpers.hpp"
#include <tt_stl/strong_type.hpp>
#include "impl/context/metal_context.hpp"
#include "tracy/Tracy.hpp"
#include "tt_align.hpp"
#include "util.hpp"

namespace tt::tt_metal {
namespace {

#if defined(TRACY_ENABLE)

std::unordered_map<int, std::string> global_mempool_names;
std::mutex global_mempool_names_mutex;

static const char* get_buffer_location_name(BufferType buffer_type, int device_id) {
    std::scoped_lock<std::mutex> lock(global_mempool_names_mutex);
    int name_combo = (int)buffer_type * 1000 + device_id;
    if (global_mempool_names.find(name_combo) == global_mempool_names.end()) {
        std::string global_mempool_name = fmt::format("Device {} {}", device_id, magic_enum::enum_name(buffer_type));
        global_mempool_names.emplace(name_combo, global_mempool_name);
    }
    return global_mempool_names[name_combo].c_str();
}
#endif

bool is_l1_impl(BufferType buffer_type) { return buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL; }

void validate_buffer_parameters(
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType& /*buffer_type*/,
    const TensorMemoryLayout& buffer_layout,
    const std::optional<ShardSpecBuffer>& shard_parameters,
    const std::optional<BufferDistributionSpec>& buffer_distribution_spec) {
    // Validate shard parameters are correct; only one of shard_parameters or buffer_distribution_spec can be set
    if (is_sharded(buffer_layout)) {
        if (buffer_distribution_spec.has_value()) {
            TT_FATAL(
                buffer_layout == TensorMemoryLayout::BLOCK_SHARDED,
                "Buffer with BufferDistributionSpec must be BLOCK_SHARDED layout!");
            TT_FATAL(
                shard_parameters == std::nullopt,
                "Buffer must only have either BufferDistributionSpec or ShardSpecBuffer!");
        } else {
            TT_FATAL(
                shard_parameters != std::nullopt,
                "Buffer was specified as sharded but does not have shard_parameters specified");
        }
    } else {
        TT_FATAL(
            shard_parameters == std::nullopt, "Buffer was specified as not sharded but has shard_parameters specified");
        TT_FATAL(
            shard_parameters == std::nullopt, "Buffer was specified as not sharded but has shard_parameters specified");
    }

    if (size == 0) {
        return;
    }

    if (buffer_layout == TensorMemoryLayout::SINGLE_BANK) {
        TT_FATAL(page_size == size, "Contiguous buffer must be one contiguous page");
    } else {
        TT_FATAL(
            size % page_size == 0,
            "For valid non-interleaved buffers page size {} must equal buffer size {}. For interleaved-buffers, "
            "buffer size should be divisble by the page size",
            page_size,
            size);
    }
}

std::tuple<std::vector<std::vector<uint32_t>>, std::vector<std::array<uint32_t, 2>>> core_to_host_pages(
    const uint32_t /*total_pages*/,
    const uint32_t pages_per_shard,
    const uint32_t num_shards,
    const TensorMemoryLayout layout,
    const std::array<uint32_t, 2>& page_shape,
    const std::array<uint32_t, 2>& shard_shape,
    const std::array<uint32_t, 2>& tensor2d_size) {
    std::array<uint32_t, 2> shard_in_pages = {
        page_shape[0] == 0 ? 0 : shard_shape[0] / page_shape[0],
        page_shape[1] == 0 ? 0 : shard_shape[1] / page_shape[1]};
    std::vector<std::vector<uint32_t>> ret_vec(num_shards);
    std::vector<std::array<uint32_t, 2>> ret_shard_shape(num_shards, shard_in_pages);

    if (layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t rem_pages = tensor2d_size[0] * tensor2d_size[1];
        uint32_t page_id = 0;
        for (uint32_t i = 0; i < num_shards; i++) {
            if (rem_pages == 0) {
                ret_shard_shape[i] = {0, 0};
            } else {
                uint32_t num_cols = std::min(pages_per_shard, rem_pages);
                if (pages_per_shard > rem_pages) {
                    ret_shard_shape[i] = {rem_pages / ret_shard_shape[i][1], ret_shard_shape[i][1]};
                }
                ret_vec[i] = std::vector<uint32_t>(num_cols);
                for (uint32_t j = 0; j < num_cols; j++) {
                    ret_vec[i][j] = page_id++;
                }
                rem_pages -= num_cols;
            }
        }
    } else if (layout == TensorMemoryLayout::WIDTH_SHARDED or layout == TensorMemoryLayout::BLOCK_SHARDED) {
        uint32_t i_offset = 0;
        uint32_t j_offset = 0;
        uint32_t num_shard_columns = shard_in_pages[1] == 0 ? 0 : div_up(tensor2d_size[1], shard_in_pages[1]);
        uint32_t shard_in_row = 0;

        for (uint32_t shard_idx = 0; shard_idx < num_shards; shard_idx++) {
            ret_vec[shard_idx].reserve(pages_per_shard);

            uint32_t i = 0;
            uint32_t j = 0;
            for (i = i_offset; i < (shard_in_pages[0] + i_offset); i++) {
                if (i >= tensor2d_size[0]) {
                    break;
                }
                for (j = j_offset; j < (shard_in_pages[1] + j_offset) and (j < (tensor2d_size[1])); j++) {
                    uint32_t host_page = i * tensor2d_size[1] + j;
                    ret_vec[shard_idx].push_back(host_page);
                }
            }
            ret_shard_shape[shard_idx] = {i - i_offset, j - j_offset};
            if (((shard_in_row + 1) == (num_shard_columns))) {
                shard_in_row = 0;
                j_offset = 0;
                i_offset += shard_in_pages[0];
            } else {
                shard_in_row++;
                j_offset += shard_in_pages[1];
            }
        }
    }
    return {ret_vec, ret_shard_shape};
}

void validate_sub_device_id(
    std::optional<SubDeviceId> sub_device_id,
    IDevice* device,
    BufferType buffer_type,
    const std::optional<ShardSpecBuffer>& shard_parameters) {
    // No need to validate if we're using the global allocator or not sharding
    if (!sub_device_id.has_value()) {
        return;
    }
    TT_FATAL(shard_parameters.has_value(), "Specifying sub-device for buffer requires buffer to be sharded");
    TT_FATAL(is_l1_impl(buffer_type), "Specifying sub-device for buffer requires buffer to be L1");
    const auto& sub_device_cores = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id.value());
    const auto& shard_cores = shard_parameters->grid();
    TT_FATAL(
        sub_device_cores.contains(shard_cores),
        "Shard cores specified {} do not match sub-device cores {}",
        shard_cores,
        sub_device_cores);
}

void validate_sub_device_manager_id(std::optional<SubDeviceManagerId> sub_device_manager_id, IDevice* device) {
    if (sub_device_manager_id.has_value()) {
        TT_FATAL(
            sub_device_manager_id.value() == device->get_active_sub_device_manager_id(),
            "Sub-device manager id mismatch. Buffer sub-device manager id: {}, Device active sub-device manager id: {}",
            sub_device_manager_id.value(),
            device->get_active_sub_device_manager_id());
    }
}

}  // namespace

std::atomic<size_t> Buffer::next_unique_id = 0;

std::ostream& operator<<(std::ostream& os, const ShardSpec& spec) {
    tt::stl::reflection::operator<<(os, spec);
    return os;
}

bool is_sharded(const TensorMemoryLayout& layout) {
    return (
        layout == TensorMemoryLayout::HEIGHT_SHARDED || layout == TensorMemoryLayout::WIDTH_SHARDED ||
        layout == TensorMemoryLayout::BLOCK_SHARDED);
}

BufferPageMapping generate_buffer_page_mapping(const Buffer& buffer) {
    BufferPageMapping buffer_page_mapping;

    if (buffer.size() == 0) {
        return buffer_page_mapping;
    }

    if (buffer.is_nd_sharded()) {
        return buffer.buffer_distribution_spec()->compute_page_mapping();
    }

    uint32_t num_cores = buffer.num_cores().value();

    auto shard_spec = buffer.shard_spec();
    bool row_major = shard_spec.orientation() == ShardOrientation::ROW_MAJOR;
    buffer_page_mapping.all_cores = corerange_to_cores(shard_spec.grid(), num_cores, row_major);
    TT_FATAL(
        num_cores == buffer_page_mapping.all_cores.size(),
        "Buffer has {} cores, but page mapping expects {} cores",
        num_cores,
        buffer_page_mapping.all_cores.size());
    uint32_t core_id = 0;
    for (const auto& core : buffer_page_mapping.all_cores) {
        buffer_page_mapping.core_to_core_id.insert({core, core_id});
        core_id++;
    }

    uint32_t num_dev_pages = buffer.num_dev_pages();
    auto [core_host_page_indices, shard_shape] = core_to_host_pages(
        num_dev_pages,
        shard_spec.num_pages(),
        num_cores,
        buffer.buffer_layout(),
        shard_spec.page_shape,
        shard_spec.shape(),
        shard_spec.tensor2d_shape_in_pages);

    buffer_page_mapping.core_host_page_indices = std::vector<std::vector<uint32_t>>(num_cores);

    buffer_page_mapping.dev_page_to_host_page_mapping =
        std::vector<std::optional<uint32_t>>(num_dev_pages, std::nullopt);
    buffer_page_mapping.dev_page_to_core_mapping = std::vector<uint32_t>(num_dev_pages);

    buffer_page_mapping.host_page_to_local_shard_page_mapping = std::vector<uint32_t>(buffer.num_pages());
    buffer_page_mapping.host_page_to_dev_page_mapping = std::vector<uint32_t>(buffer.num_pages());
    buffer_page_mapping.core_shard_shape = std::move(shard_shape);
    uint32_t dev_page_index = 0;

    auto shape_in_pages = shard_spec.shape_in_pages();
    for (uint32_t core_index = 0; core_index < core_host_page_indices.size(); core_index++) {
        uint32_t valid_shard_page = 0;
        buffer_page_mapping.core_host_page_indices[core_index].reserve(shard_spec.num_pages());
        uint32_t shard_page_id = 0;
        for (uint32_t shard_page_x = 0; shard_page_x < shape_in_pages[0]; shard_page_x++) {
            for (uint32_t shard_page_y = 0; shard_page_y < shape_in_pages[1]; shard_page_y++) {
                buffer_page_mapping.dev_page_to_core_mapping[dev_page_index] = core_index;
                if (shard_page_x < buffer_page_mapping.core_shard_shape[core_index][0] and
                    shard_page_y < buffer_page_mapping.core_shard_shape[core_index][1]) {
                    uint32_t host_page = core_host_page_indices[core_index][valid_shard_page];
                    buffer_page_mapping.dev_page_to_host_page_mapping[dev_page_index] = host_page;
                    buffer_page_mapping.core_host_page_indices[core_index].push_back(host_page);
                    buffer_page_mapping.host_page_to_local_shard_page_mapping[host_page] = shard_page_id;
                    buffer_page_mapping.host_page_to_dev_page_mapping[host_page] = dev_page_index;
                    valid_shard_page++;
                }
                dev_page_index++;
                shard_page_id++;
            }
        }
    }

    return buffer_page_mapping;
}

Buffer::Buffer(
    IDevice* device,
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const TensorMemoryLayout buffer_layout,
    const std::optional<std::variant<ShardSpecBuffer, BufferDistributionSpec>>& shard_parameters,
    const std::optional<bool> bottom_up,
    const std::optional<SubDeviceId> sub_device_id,
    const bool owns_data,
    Private) :
    device_(device),
    size_(size),
    page_size_(page_size),
    buffer_type_(buffer_type),
    buffer_layout_(buffer_layout),
    bottom_up_(bottom_up.value_or(this->is_dram())),
    sub_device_id_(sub_device_id),
    owns_data_(owns_data),
    buffer_page_mapping_(nullptr) {
    if (shard_parameters) {
        std::visit(
            tt::stl::overloaded{
                [this](const ShardSpecBuffer& shard_spec_buffer) { this->shard_parameters_ = shard_spec_buffer; },
                [this](const BufferDistributionSpec& buffer_distribution_spec) {
                    this->buffer_distribution_spec_ = buffer_distribution_spec;
                }},
            shard_parameters.value());
    }
    TT_FATAL(this->device_ != nullptr, "Device needs to not be null.");
    if (this->sub_device_id_.has_value()) {
        validate_sub_device_id(this->sub_device_id_, this->device_, buffer_type, shard_parameters_);
        this->sub_device_manager_id_ = this->device_->get_active_sub_device_manager_id();
        this->allocator_ = device->allocator(*this->sub_device_id_).get();
    } else {
        this->allocator_ = device->allocator().get();
    }
    validate_buffer_parameters(
        size, page_size, buffer_type, buffer_layout, shard_parameters_, buffer_distribution_spec_);
    unique_id_ = next_unique_id.fetch_add(1);
}

std::shared_ptr<Buffer> Buffer::create(
    IDevice* device,
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const TensorMemoryLayout buffer_layout,
    const std::optional<std::variant<ShardSpecBuffer, BufferDistributionSpec>>& shard_parameters,
    const std::optional<bool> bottom_up,
    const std::optional<SubDeviceId> sub_device_id) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();

    auto buffer = std::make_shared<Buffer>(
        device,
        size,
        page_size,
        buffer_type,
        buffer_layout,
        shard_parameters,
        bottom_up,
        sub_device_id,
        true /* owns data */,
        Private());

    if (buffer->size_ == 0) {
        buffer->allocation_status_ = AllocationStatus::ALLOCATED;
        return buffer;
    }

    buffer->allocate_impl();

    LIGHT_METAL_TRACE_FUNCTION_CALL(
        CaptureBufferCreate,
        buffer,
        device,
        std::nullopt,
        size,
        page_size,
        buffer_type,
        buffer_layout,
        shard_parameters,
        bottom_up,
        sub_device_id);

    return buffer;
}

std::shared_ptr<Buffer> Buffer::create(
    IDevice* device,
    DeviceAddr address,
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const TensorMemoryLayout buffer_layout,
    const std::optional<std::variant<ShardSpecBuffer, BufferDistributionSpec>>& shard_parameters,
    const std::optional<bool> bottom_up,
    const std::optional<SubDeviceId> sub_device_id) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    auto buffer = std::make_shared<Buffer>(
        device,
        size,
        page_size,
        buffer_type,
        buffer_layout,
        shard_parameters,
        bottom_up,
        sub_device_id,
        false /* owns data */,
        Private());

    buffer->address_ = address;
    buffer->allocation_status_ = AllocationStatus::ALLOCATED;

    LIGHT_METAL_TRACE_FUNCTION_CALL(
        CaptureBufferCreate,
        buffer,
        device,
        address,
        size,
        page_size,
        buffer_type,
        buffer_layout,
        shard_parameters,
        bottom_up,
        sub_device_id);

    return buffer;
}

void Buffer::allocate_impl() {
    if (GraphTracker::instance().hook_allocate(this)) {
        address_ = 0;
        hooked_allocation_ = true;
    } else {
        validate_sub_device_manager_id(sub_device_manager_id_, device_);

        address_ = allocator_->allocate_buffer(this);

        // Assertion here because buffer class returns a u32 when address is queried
        // Requires updating all use cases of buffer address to accept a u64 to remove
        TT_ASSERT(address_ <= std::numeric_limits<uint32_t>::max());

#if defined(TRACY_ENABLE)
        if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_buffer_usage_enabled()) {
            TracyAllocN(
                reinterpret_cast<const void*>(address_), size_, get_buffer_location_name(buffer_type_, device_->id()));
        }
#endif
    }

    // Important! Graph tracker must called after the allocation status is updated.
    allocation_status_ = AllocationStatus::ALLOCATED;

    GraphTracker::instance().track_allocate(this);
}

void Buffer::deallocate() {
    if (!owns_data_) {
        return;
    }
    this->deallocate_impl();
}

void Buffer::mark_as_deallocated() { allocation_status_ = AllocationStatus::DEALLOCATED; }

void Buffer::deallocate_impl() {
    if (allocation_status_ != AllocationStatus::ALLOCATED) {
        return;
    }

    if (device_->is_initialized() && size_ != 0) {
        // address_ is only modified from this thread, no sync required
        GraphTracker::instance().track_deallocate(this);
        if (!GraphTracker::instance().hook_deallocate(this) && !hooked_allocation_) {
#if defined(TRACY_ENABLE)
            if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_buffer_usage_enabled()) {
                TracyFreeN(
                    reinterpret_cast<const void*>(address()), get_buffer_location_name(buffer_type_, device_->id()));
            }
#endif
            validate_sub_device_manager_id(sub_device_manager_id_, device_);
            allocator_->deallocate_buffer(this);
        }

        // Capture deallocates here instead of higher levels.
        LIGHT_METAL_TRACE_FUNCTION_ENTRY();
        LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureBufferDeallocate, *this);
    }

    allocation_status_ = AllocationStatus::DEALLOCATED;
}

Buffer::~Buffer() {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureBufferDelete, *this);
    if (this->allocation_status_ != AllocationStatus::DEALLOCATED) {
        this->deallocate();
    }
}

bool Buffer::is_allocated() const { return allocation_status_ == AllocationStatus::ALLOCATED; }

uint32_t Buffer::address() const {
    TT_FATAL(
        allocation_status_ != AllocationStatus::ALLOCATION_REQUESTED,
        "Can only query the address of a buffer that has been allocated");
    return address_;
}

DeviceAddr Buffer::page_size() const { return page_size_; }

void Buffer::set_page_size(DeviceAddr page_size) {
    TT_FATAL(
        buffer_distribution_spec_ == std::nullopt, "Buffer::set_page_size is unsupported for BufferDistributionSpec!");
    TT_FATAL(page_size == 0 ? size_ == 0 : size_ % page_size == 0, "buffer size must be divisible by new page size");
    page_size_ = page_size;
    this->buffer_page_mapping_ = nullptr;
}

uint32_t Buffer::num_pages() const { return page_size() == 0 ? 0 : size() / page_size(); }

uint32_t Buffer::num_dev_pages() const {
    if (!is_sharded(this->buffer_layout_)) {
        return this->num_pages();
    }
    if (buffer_distribution_spec_.has_value()) {
        return buffer_distribution_spec_.value().num_dev_pages_per_core() * num_cores().value();
    }
    // TODO: This logic seems wrong since num_cores() doesn't store the actual number of cores used for shards
    return this->shard_spec().num_pages() * this->num_cores().value();
}

HalMemType Buffer::memory_type() const {
    if (this->is_dram()) {
        return HalMemType::DRAM;
    } else if (this->is_l1()) {
        return HalMemType::L1;
    } else {
        TT_THROW("Unknown HAL memory type for {} buffer type", this->buffer_type());
    }
}

CoreType Buffer::core_type() const {
    switch (this->buffer_type_) {
        case BufferType::DRAM: return CoreType::DRAM;
        case BufferType::L1:
        case BufferType::L1_SMALL: return CoreType::WORKER;
        default: TT_THROW("Unknown CoreType {} for buffer", this->buffer_type_);
    }
}

bool Buffer::is_l1() const { return is_l1_impl(buffer_type()); }
bool Buffer::is_dram() const { return buffer_type() == BufferType::DRAM || buffer_type() == BufferType::TRACE; }
bool Buffer::is_trace() const { return buffer_type() == BufferType::TRACE; }

bool Buffer::is_valid_region(const BufferRegion& region) const { return region.offset + region.size <= this->size(); }

bool Buffer::is_valid_partial_region(const BufferRegion& region) const {
    return this->is_valid_region(region) && (region.offset > 0 || region.size != this->size());
}

DeviceAddr Buffer::page_address(uint32_t bank_id, uint32_t page_index) const {
    uint32_t num_banks = allocator_->get_num_banks(buffer_type_);
    TT_FATAL(bank_id < num_banks, "Invalid Bank ID: {} exceeds total numbers of banks ({})!", bank_id, num_banks);
    int pages_offset_within_bank = (int)page_index / num_banks;
    auto offset = (round_up(this->page_size(), this->alignment()) * pages_offset_within_bank);
    return translate_page_address(offset, bank_id);
}

DeviceAddr Buffer::bank_local_page_address(uint32_t bank_id, uint32_t page_index) const {
    uint32_t num_banks = allocator_->get_num_banks(buffer_type_);
    TT_FATAL(bank_id < num_banks, "Invalid Bank ID: {} exceeds total numbers of banks ({})!", bank_id, num_banks);
    uint32_t offset;
    if (is_sharded(this->buffer_layout())) {
        size_t num_pages_per_shard = 0;
        if (is_nd_sharded()) {
            const auto& distribution_spec = *buffer_distribution_spec_;
            num_pages_per_shard = distribution_spec.num_dev_pages_per_core();
        } else {
            auto shard_spec = this->shard_spec();
            num_pages_per_shard = shard_spec.num_pages();
        }
        uint32_t pages_offset_within_bank = page_index % num_pages_per_shard;
        offset = (round_up(this->page_size(), this->alignment()) * pages_offset_within_bank);
    } else {
        uint32_t pages_offset_within_bank = page_index / num_banks;
        offset = (round_up(this->page_size(), this->alignment()) * pages_offset_within_bank);
    }
    return this->address() + offset;
}

uint32_t Buffer::alignment() const { return allocator_->get_alignment(this->buffer_type()); }

DeviceAddr Buffer::aligned_page_size() const { return align(page_size(), this->alignment()); }
DeviceAddr Buffer::aligned_size() const { return this->num_dev_pages() * this->aligned_page_size(); }

DeviceAddr Buffer::aligned_size_per_bank() const {
    // TODO: Revist for ND sharding (it looks okay since num_cores() handles ND sharding)
    uint32_t num_banks =
        is_sharded(this->buffer_layout_) ? this->num_cores().value() : allocator_->get_num_banks(this->buffer_type());
    return tt::tt_metal::detail::SizeBytesPerBank(
        this->aligned_size(), this->aligned_page_size(), num_banks, this->alignment());
}

DeviceAddr Buffer::sharded_page_address(uint32_t bank_id, uint32_t page_index) const {
    TT_FATAL(is_sharded(this->buffer_layout()), "Buffer not sharded");
    size_t num_pages_per_shard = 0;
    if (is_nd_sharded()) {
        const auto& distribution_spec = *buffer_distribution_spec_;
        num_pages_per_shard = distribution_spec.num_dev_pages_per_core();
    } else {
        auto shard_spec = this->shard_spec();
        num_pages_per_shard = shard_spec.num_pages();
    }
    uint32_t pages_offset_within_bank = page_index % num_pages_per_shard;
    auto offset = (round_up(this->page_size(), this->alignment()) * pages_offset_within_bank);
    return translate_page_address(offset, bank_id);
}

ShardSpecBuffer Buffer::shard_spec() const {
    TT_FATAL(is_sharded(this->buffer_layout_), "Buffer not sharded");
    TT_FATAL(shard_parameters_.has_value(), "Buffer is sharded, but no shard parameters specified");
    return this->shard_parameters_.value();
}

void Buffer::set_shard_spec(const ShardSpecBuffer& shard_spec) {
    this->shard_parameters_ = shard_spec;
    this->buffer_page_mapping_ = nullptr;
}

std::optional<uint32_t> Buffer::num_cores() const {
    if (!is_sharded(this->buffer_layout_)) {
        return std::nullopt;
    }
    if (buffer_distribution_spec_.has_value()) {
        return buffer_distribution_spec_.value().num_cores();
    }
    return this->shard_spec().tensor_shard_spec.grid.num_cores();
}

DeviceAddr Buffer::translate_page_address(uint64_t offset, uint32_t bank_id) const {
    DeviceAddr base_page_address = this->address() + allocator_->get_bank_offset(buffer_type_, bank_id);
    return base_page_address + offset;
}

const std::shared_ptr<const BufferPageMapping>& Buffer::get_buffer_page_mapping() {
    TT_FATAL(is_sharded(this->buffer_layout_), "Buffer not sharded");
    if (!this->buffer_page_mapping_) {
        this->buffer_page_mapping_ = std::make_shared<const BufferPageMapping>(generate_buffer_page_mapping(*this));
    }
    return this->buffer_page_mapping_;
}

bool Buffer::is_nd_sharded() const { return this->buffer_distribution_spec_.has_value(); }

const std::optional<BufferDistributionSpec>& Buffer::buffer_distribution_spec() const {
    return this->buffer_distribution_spec_;
}

bool ShardSpec::operator==(const ShardSpec&) const = default;
bool ShardSpec::operator!=(const ShardSpec&) const = default;

std::array<uint32_t, 2> ShardSpecBuffer::shape_in_pages() const {
    auto height_in_pages = page_shape[0] == 0 ? 0 : tensor_shard_spec.shape[0] / page_shape[0];
    auto width_in_pages = page_shape[1] == 0 ? 0 : tensor_shard_spec.shape[1] / page_shape[1];
    return {height_in_pages, width_in_pages};
}

DeviceAddr ShardSpecBuffer::num_pages() const {
    auto shape_in_pages_ = this->shape_in_pages();
    return shape_in_pages_[0] * shape_in_pages_[1];
}

}  // namespace tt::tt_metal

namespace tt::stl::json {
tt_metal::ShardSpec from_json_t<tt_metal::ShardSpec>::operator()(const nlohmann::json& json_object) const {
    const auto& shard_mode = from_json<tt_metal::ShardMode>(json_object.at("mode"));
    const auto& physical_shard_shape =
        from_json<std::optional<std::array<uint32_t, 2>>>(json_object.at("physical_shard_shape"));
    if (physical_shard_shape.has_value()) {
        TT_FATAL(
            shard_mode == tt::tt_metal::ShardMode::LOGICAL,
            "Physical shard shape can only be provided in logical sharding mode!");
        return tt_metal::ShardSpec{
            from_json<CoreRangeSet>(json_object.at("grid")),
            from_json<std::array<uint32_t, 2>>(json_object.at("shape")),
            physical_shard_shape.value(),
            from_json<tt_metal::ShardOrientation>(json_object.at("orientation"))};
    }
    return tt_metal::ShardSpec{
        from_json<CoreRangeSet>(json_object.at("grid")),
        from_json<std::array<uint32_t, 2>>(json_object.at("shape")),
        from_json<tt_metal::ShardOrientation>(json_object.at("orientation")),
        shard_mode};
}
}  // namespace tt::stl::json
