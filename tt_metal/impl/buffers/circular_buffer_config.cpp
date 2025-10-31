// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "circular_buffer_config.hpp"

#include <unordered_map>

#include <tt_stl/assert.hpp>
#include "buffer.hpp"
#include <tt-logger/tt-logger.hpp>

namespace tt {
enum class DataFormat : uint8_t;
}  // namespace tt

namespace tt::tt_metal {

// Static circular buffer spec
CircularBufferConfigImpl::CircularBufferConfigImpl(
    uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec) :
    total_size_(total_size), globally_allocated_address_(std::nullopt), dynamic_cb_(false) {
    this->set_config(data_format_spec);
}

// User is expected to use the builder here.
CircularBufferConfigImpl::CircularBufferConfigImpl(uint32_t total_size) :
    total_size_(total_size), globally_allocated_address_(std::nullopt), dynamic_cb_(false) {}

// Dynamic circular buffer spec
CircularBufferConfigImpl::CircularBufferConfigImpl(
    uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec, const Buffer& buffer) :
    total_size_(total_size) {
    this->set_globally_allocated_address(buffer);
    this->set_config(data_format_spec);
}

CircularBufferConfigImpl::CircularBufferConfigImpl(const CBDescriptor& descriptor) :
    total_size_(descriptor.total_size) {
    if (descriptor.buffer) {
        this->set_globally_allocated_address(*descriptor.buffer);
    }

    auto process_format_descriptor = [this](const CBFormatDescriptor& format_descriptor) {
        if (format_descriptor.buffer_index > NUM_CIRCULAR_BUFFERS - 1) {
            TT_THROW(
                "Buffer index ({}) exceeds max number of circular buffers per core ({})",
                format_descriptor.buffer_index,
                NUM_CIRCULAR_BUFFERS);
        }
        this->data_formats_[format_descriptor.buffer_index] = format_descriptor.data_format;
        if (this->total_size_ % format_descriptor.page_size != 0) {
            TT_THROW(
                "Total circular buffer size {} B must be divisible by page size {} B",
                this->total_size_,
                format_descriptor.page_size);
        }
        this->page_sizes_[format_descriptor.buffer_index] = format_descriptor.page_size;
        // if (format_descriptor.tile) {
        //     this->tiles_[format_descriptor.buffer_index] = Tile(
        //         {format_descriptor.tile->height, format_descriptor.tile->width}, format_descriptor.tile->transpose);
        // }
    };
    this->buffer_indices_.reserve(descriptor.format_descriptors.size() + descriptor.remote_format_descriptors.size());
    this->local_buffer_indices_.reserve(descriptor.format_descriptors.size());
    this->remote_buffer_indices_.reserve(descriptor.remote_format_descriptors.size());
    for (const auto& format_descriptor : descriptor.format_descriptors) {
        process_format_descriptor(format_descriptor);
        this->buffer_indices_.insert(format_descriptor.buffer_index);
        this->local_buffer_indices_.insert(format_descriptor.buffer_index);
    }
    for (const auto& format_descriptor : descriptor.remote_format_descriptors) {
        process_format_descriptor(format_descriptor);
        this->buffer_indices_.insert(format_descriptor.buffer_index);
        this->remote_buffer_indices_.insert(format_descriptor.buffer_index);
    }
}

// For flatbuffer deserialization, set all private members.
CircularBufferConfigImpl::CircularBufferConfigImpl(
    uint32_t total_size,
    std::optional<uint32_t> globally_allocated_address,
    const std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS>& data_formats,
    const std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS>& page_sizes,
    const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& tiles,
    const std::unordered_set<uint8_t>& buffer_indices,
    const std::unordered_set<uint8_t>& local_buffer_indices,
    const std::unordered_set<uint8_t>& remote_buffer_indices,
    bool dynamic_cb,
    uint32_t max_size,
    uint32_t buffer_size) :
    total_size_(total_size),
    globally_allocated_address_(globally_allocated_address),
    data_formats_(data_formats),
    page_sizes_(page_sizes),
    tiles_(tiles),
    buffer_indices_(buffer_indices),
    local_buffer_indices_(local_buffer_indices),
    remote_buffer_indices_(remote_buffer_indices),
    dynamic_cb_(dynamic_cb),
    max_size_(max_size),
    buffer_size_(buffer_size) {}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_page_size(uint8_t buffer_index, uint32_t page_size) {
    if (buffer_index > NUM_CIRCULAR_BUFFERS - 1) {
        TT_THROW(
            "Buffer index ({}) exceeds max number of circular buffers per core ({})",
            buffer_index,
            NUM_CIRCULAR_BUFFERS);
    }
    if (this->buffer_indices_.find(buffer_index) == this->buffer_indices_.end()) {
        TT_THROW(
            "Illegal circular buffer index {}. Page size can only be specified for buffer indices configured "
            "during config creation",
            buffer_index);
    }
    if (this->total_size_ % page_size != 0) {
        TT_THROW(
            "Failed allocation attempt on buffer index {}. Total circular buffer size {} B must be divisible by page "
            "size {} B",
            buffer_index,
            this->total_size_,
            page_size);
    }

    this->page_sizes_[buffer_index] = page_size;
    return *this;
}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_total_size(uint32_t total_size) {
    if (dynamic_cb_) {
        TT_FATAL(
            total_size <= this->max_size_,
            "Cannot set circular buffer size to {}. This is larger than the associated dynamically allocated "
            "L1 buffer bank size of {} B",
            total_size,
            this->max_size_);
    }
    this->total_size_ = total_size;
    return *this;
}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_globally_allocated_address(const Buffer& buffer) {
    return this->set_globally_allocated_address_and_total_size(buffer, this->total_size_);
}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_globally_allocated_address_and_total_size(
    const Buffer& buffer, uint32_t total_size) {
    if (not buffer.is_l1()) {
        TT_THROW("Only L1 buffers can have an associated circular buffer!");
    }
    this->globally_allocated_address_ = buffer.address();
    this->dynamic_cb_ = true;
    this->max_size_ = buffer.aligned_size_per_bank();
    this->buffer_size_ = buffer.aligned_size();
    this->shadow_global_buffer = &buffer;
    this->set_total_size(total_size);
    return *this;
}

CircularBufferConfigImpl& CircularBufferConfigImpl::set_tile_dims(uint8_t buffer_index, const Tile& tile) {
    this->tiles_[buffer_index] = tile;
    return *this;
}

const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& CircularBufferConfigImpl::tiles() const {
    return this->tiles_;
}

uint32_t CircularBufferConfigImpl::total_size() const { return this->total_size_; }

std::optional<uint32_t> CircularBufferConfigImpl::globally_allocated_address() const {
    return this->globally_allocated_address_;
}

const std::unordered_set<uint8_t>& CircularBufferConfigImpl::buffer_indices() const { return this->buffer_indices_; }
const std::unordered_set<uint8_t>& CircularBufferConfigImpl::local_buffer_indices() const {
    return this->local_buffer_indices_;
}
const std::unordered_set<uint8_t>& CircularBufferConfigImpl::remote_buffer_indices() const {
    return this->remote_buffer_indices_;
}

const std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS>& CircularBufferConfigImpl::data_formats() const {
    return this->data_formats_;
}

const std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS>& CircularBufferConfigImpl::page_sizes() const {
    return this->page_sizes_;
}

bool CircularBufferConfigImpl::dynamic_cb() const { return this->dynamic_cb_; }

uint32_t CircularBufferConfigImpl::max_size() const { return this->max_size_; }

uint32_t CircularBufferConfigImpl::buffer_size() const { return this->buffer_size_; }

void CircularBufferConfigImpl::set_config(const std::map<uint8_t, tt::DataFormat>& data_format_spec) {
    if (data_format_spec.size() > NUM_CIRCULAR_BUFFERS) {
        TT_THROW(
            "Only {} circular buffer slots are available but data formats are specified for {} indices",
            NUM_CIRCULAR_BUFFERS,
            data_format_spec.size());
    }

    for (const auto& [buffer_index, data_format] : data_format_spec) {
        if (buffer_index > NUM_CIRCULAR_BUFFERS - 1) {
            TT_THROW(
                "Buffer index ({}) exceeds max number of circular buffers per core ({})",
                buffer_index,
                NUM_CIRCULAR_BUFFERS);
        }
        this->data_formats_[buffer_index] = data_format;
        this->buffer_indices_.insert(buffer_index);
        this->local_buffer_indices_.insert(buffer_index);
    }
}

bool operator==(const CircularBufferConfigImpl& lhs, const CircularBufferConfigImpl& rhs) {
    return lhs.total_size() == rhs.total_size() &&
           lhs.globally_allocated_address() == rhs.globally_allocated_address() &&
           lhs.data_formats() == rhs.data_formats() && lhs.page_sizes() == rhs.page_sizes() &&
           lhs.tiles() == rhs.tiles() && lhs.shadow_global_buffer == rhs.shadow_global_buffer;
}

bool operator!=(const CircularBufferConfigImpl& lhs, const CircularBufferConfigImpl& rhs) { return !(lhs == rhs); }

// CircularBufferConfig pimpl implementations

// Static circular buffer spec
CircularBufferConfig::CircularBufferConfig(
    uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec) :
    impl_(std::make_unique<CircularBufferConfigImpl>(total_size, data_format_spec)) {}

// User is expected to use the builder here.
CircularBufferConfig::CircularBufferConfig(uint32_t total_size) :
    impl_(std::make_unique<CircularBufferConfigImpl>(total_size)) {}

// Dynamic circular buffer spec
CircularBufferConfig::CircularBufferConfig(
    uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec, const Buffer& buffer) :
    impl_(std::make_unique<CircularBufferConfigImpl>(total_size, data_format_spec, buffer)) {}

CircularBufferConfig::CircularBufferConfig(const CBDescriptor& descriptor) :
    impl_(std::make_unique<CircularBufferConfigImpl>(descriptor)) {}

// Copy constructor
CircularBufferConfig::CircularBufferConfig(const CircularBufferConfig& other) :
    impl_(std::make_unique<CircularBufferConfigImpl>(*other.impl_)) {}

// Copy assignment operator
CircularBufferConfig& CircularBufferConfig::operator=(const CircularBufferConfig& other) {
    if (this != &other) {
        impl_ = std::make_unique<CircularBufferConfigImpl>(*other.impl_);
    }
    return *this;
}

// For flatbuffer deserialization, set all private members.
CircularBufferConfig::CircularBufferConfig(
    uint32_t total_size,
    std::optional<uint32_t> globally_allocated_address,
    const std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS>& data_formats,
    const std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS>& page_sizes,
    const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& tiles,
    const std::unordered_set<uint8_t>& buffer_indices,
    const std::unordered_set<uint8_t>& local_buffer_indices,
    const std::unordered_set<uint8_t>& remote_buffer_indices,
    bool dynamic_cb,
    uint32_t max_size,
    uint32_t buffer_size) :
    impl_(std::make_unique<CircularBufferConfigImpl>(
        total_size,
        globally_allocated_address,
        data_formats,
        page_sizes,
        tiles,
        buffer_indices,
        local_buffer_indices,
        remote_buffer_indices,
        dynamic_cb,
        max_size,
        buffer_size)) {}

CircularBufferConfig& CircularBufferConfig::set_page_size(uint8_t buffer_index, uint32_t page_size) {
    impl_->set_page_size(buffer_index, page_size);
    return *this;
}

CircularBufferConfig& CircularBufferConfig::set_globally_allocated_address(const Buffer& buffer) {
    impl_->set_globally_allocated_address(buffer);
    return *this;
}

CircularBufferConfig& CircularBufferConfig::set_tile_dims(uint8_t buffer_index, const Tile& tile) {
    impl_->set_tile_dims(buffer_index, tile);
    return *this;
}

CircularBufferConfig::Builder CircularBufferConfig::Builder::LocalBuilder(
    CircularBufferConfig& parent, uint8_t buffer_index) {
    return Builder(parent, buffer_index);
}

CircularBufferConfig::Builder CircularBufferConfig::Builder::RemoteBuilder(
    CircularBufferConfig& parent, uint8_t buffer_index) {
    return Builder(parent, buffer_index);
}

CircularBufferConfig::Builder::Builder(CircularBufferConfig& parent, uint8_t buffer_index) :
    parent_(parent), buffer_index_(buffer_index) {}

const CircularBufferConfig::Builder& CircularBufferConfig::Builder::set_data_format(tt::DataFormat data_format) const {
    auto* impl = parent_.impl();
    impl->data_formats_[buffer_index_] = data_format;
    impl->buffer_indices_.insert(buffer_index_);
    impl->local_buffer_indices_.insert(buffer_index_);
    return *this;
}

const CircularBufferConfig::Builder& CircularBufferConfig::Builder::set_total_size(uint32_t total_size) const {
    parent_.impl_->set_total_size(total_size);
    return *this;
}

const CircularBufferConfig::Builder& CircularBufferConfig::Builder::set_page_size(uint32_t page_size) const {
    parent_.impl_->set_page_size(buffer_index_, page_size);
    return *this;
}

const CircularBufferConfig::Builder& CircularBufferConfig::Builder::set_tile_dims(const Tile& tile) const {
    parent_.impl_->set_tile_dims(buffer_index_, tile);
    return *this;
}

CircularBufferConfig::Builder CircularBufferConfig::index(uint8_t buffer_index) {
    return Builder::LocalBuilder(*this, buffer_index);
}

CircularBufferConfig::Builder CircularBufferConfig::remote_index(uint8_t buffer_index) {
    return Builder::RemoteBuilder(*this, buffer_index);
}

bool operator==(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs) { return *lhs.impl_ == *rhs.impl_; }

bool operator!=(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs) { return !(lhs == rhs); }

}  // namespace tt::tt_metal
