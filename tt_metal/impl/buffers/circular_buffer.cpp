// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <buffer.hpp>
#include <circular_buffer.hpp>
#include <global_circular_buffer.hpp>
#include <array>
#include <string>

#include <tt_stl/assert.hpp>
#include "impl/buffers/circular_buffer.hpp"
#include "circular_buffer_config.hpp"
#include "circular_buffer_constants.h"
#include "tile.hpp"

namespace tt::tt_metal {
// We use 16 bits to store tiles_received and tiles_acked.
static constexpr uint32_t cb_page_count_bits = 16;
static constexpr uint32_t max_num_cb_pages = (1 << cb_page_count_bits) - 1;

// Dynamic CBs will be created with address_ initialized to globally allocated address
// Static CBs will not have address set until their owning Program allocates them
CircularBufferImpl::CircularBufferImpl(const CoreRangeSet& core_range_set, const CircularBufferConfig& config) :
    id_(reinterpret_cast<uintptr_t>(this)),
    core_ranges_(core_range_set),
    config_(config),
    locally_allocated_address_(std::nullopt) {
    this->validate_set_config_attributes();
    TT_FATAL(
        this->config_.remote_buffer_indices().empty(),
        "Remote buffer indices are not supported without a GlobalCircularBuffer");
    if (globally_allocated()) {
        globally_allocated_address_ = config.globally_allocated_address().value();
    }
}

CircularBufferImpl::CircularBufferImpl(
    const CoreRangeSet& core_ranges,
    const CircularBufferConfig& config,
    const experimental::GlobalCircularBuffer& global_circular_buffer) :
    id_(reinterpret_cast<uintptr_t>(this)),
    core_ranges_(core_ranges),
    config_(config),
    locally_allocated_address_(std::nullopt) {
    this->validate_set_config_attributes();
    TT_FATAL(
        !config.globally_allocated_address().has_value(),
        "Connot create CircularBuffer with specified GlobalCircularBuffer when config already linked to a buffer");
    TT_FATAL(
        !this->config_.remote_buffer_indices().empty(),
        "Remote buffer indices should be specified when using a GlobalCircularBuffer");
    this->set_global_circular_buffer(global_circular_buffer);
}

CircularBufferImpl::CircularBufferImpl(const CBDescriptor& descriptor) :
    id_(reinterpret_cast<uintptr_t>(this)),
    core_ranges_(descriptor.core_ranges),
    config_(descriptor),
    locally_allocated_address_(std::nullopt) {
    this->validate_set_config_attributes();
    if (descriptor.global_circular_buffer) {
        TT_FATAL(
            !config_.globally_allocated_address().has_value(),
            "Connot create CircularBuffer with specified GlobalCircularBuffer when config already linked to a buffer");
        TT_FATAL(
            !this->config_.remote_buffer_indices().empty(),
            "Remote buffer indices should be specified when using a GlobalCircularBuffer");
        this->set_global_circular_buffer(*descriptor.global_circular_buffer);
    } else {
        if (globally_allocated()) {
            globally_allocated_address_ = config_.globally_allocated_address().value();
        }
    }
}

void CircularBufferImpl::validate_set_config_attributes() {
    for (uint8_t buffer_index = 0; buffer_index < NUM_CIRCULAR_BUFFERS; buffer_index++) {
        std::optional<DataFormat> data_format_spec = this->config_.data_formats().at(buffer_index);
        std::optional<uint32_t> page_size_spec = this->config_.page_sizes().at(buffer_index);

        bool df_set = data_format_spec.has_value();
        bool ps_set = page_size_spec.has_value();
        if (df_set != ps_set) {
            std::string df_set_str = df_set ? "Data format is set" : "Data format is not set";
            std::string ps_set_str = ps_set ? "Page size is set" : "Page size is not set";
            TT_THROW(
                "Expected both data format and page size to be set for buffer index {}. {}. {}.",
                buffer_index,
                df_set_str,
                ps_set_str);
        }
        if (ps_set) {
            // Validate number of pages is not too large.
            this->num_pages(buffer_index);
        }
    }
}

bool CircularBufferImpl::is_on_logical_corerange(const CoreRange& logical_cr) const {
    return this->core_ranges_.intersects(logical_cr);
}

bool CircularBufferImpl::is_on_logical_core(const CoreCoord& logical_core) const {
    return this->core_ranges_.contains(logical_core);
}

bool CircularBufferImpl::uses_buffer_index(uint32_t buffer_index) const {
    return this->buffer_indices().contains(buffer_index);
}

uint32_t CircularBufferImpl::page_size(uint32_t buffer_index) const {
    if (not this->uses_buffer_index(buffer_index)) {
        TT_THROW(
            "Cannot access page size for buffer index {} because circular buffer is not configured on that index",
            buffer_index);
    }
    uint32_t page_size = this->config_.page_sizes().at(buffer_index).value();
    if (this->size() % page_size != 0) {
        TT_THROW("Total circular buffer size {} B must be divisible by page size {} B", this->size(), page_size);
    }
    return page_size;
}

uint32_t CircularBufferImpl::num_pages(uint32_t buffer_index) const {
    uint32_t num_pages = this->size() / this->page_size(buffer_index);
    // LocalCBInterface.tiles_acked and LocalCBInterface.tiles_received are 16 bits, so we need to check if the
    // number of pages would cause overflow.
    TT_FATAL(
        num_pages <= max_num_cb_pages,
        "For buffer index {}, number of CB pages {} is greater than {}. Would cause wraparound in the CB "
        "calculations.",
        buffer_index,
        num_pages,
        max_num_cb_pages);
    return num_pages;
}

DataFormat CircularBufferImpl::data_format(uint32_t buffer_index) const {
    if (not this->uses_buffer_index(buffer_index)) {
        TT_THROW(
            "Cannot access data format for buffer index {} because circular buffer is not configured on that index",
            buffer_index);
    }
    return this->config_.data_formats().at(buffer_index).value();
}

const std::optional<Tile>& CircularBufferImpl::tile(uint32_t buffer_index) const {
    if (not this->uses_buffer_index(buffer_index)) {
        TT_THROW(
            "Cannot access tile dims for buffer index {} because circular buffer is not configured on that index",
            buffer_index);
    }
    return this->config_.tiles().at(buffer_index);
}

uint32_t CircularBufferImpl::address() const {
    if (not locally_allocated_address_.has_value() and not this->globally_allocated()) {
        TT_THROW("Circular buffer has not been allocated, cannot request address at this time!");
    }

    return this->globally_allocated() ? globally_allocated_address_ : locally_allocated_address_.value();
}

void CircularBufferImpl::assign_global_address() {
    globally_allocated_address_ = config_.shadow_global_buffer->address();
}

void CircularBufferImpl::set_global_circular_buffer(const experimental::GlobalCircularBuffer& global_circular_buffer) {
    TT_FATAL(
        global_circular_buffer.all_cores().contains(this->core_ranges_),
        "Specified cores are not contained in associated GlobalCircularBuffer");
    this->config().set_globally_allocated_address(global_circular_buffer.cb_buffer());
    this->shadow_global_circular_buffer_ = &global_circular_buffer;
    this->globally_allocated_address_ = global_circular_buffer.buffer_address();
    this->global_circular_buffer_config_address_ = global_circular_buffer.config_address();
}

DeviceAddr CircularBufferImpl::config_address() const { return this->global_circular_buffer_config_address_; }

// CircularBuffer wrapper implementations
CircularBuffer::CircularBuffer(const CircularBufferImpl* impl) : impl_(impl) {}

CBHandle CircularBuffer::id() const { return impl_->id(); }

const CoreRangeSet& CircularBuffer::core_ranges() const { return impl_->core_ranges(); }

std::size_t CircularBuffer::size() const { return impl_->size(); }

bool CircularBuffer::globally_allocated() const { return impl_->globally_allocated(); }

const std::unordered_set<uint8_t>& CircularBuffer::buffer_indices() const { return impl_->buffer_indices(); }

uint32_t CircularBuffer::page_size(uint32_t buffer_index) const { return impl_->page_size(buffer_index); }

uint32_t CircularBuffer::num_pages(uint32_t buffer_index) const { return impl_->num_pages(buffer_index); }

DataFormat CircularBuffer::data_format(uint32_t buffer_index) const { return impl_->data_format(buffer_index); }

}  // namespace tt::tt_metal
