// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "circular_buffer_types.hpp"

namespace tt::tt_metal {
inline namespace v0 {

// Static circular buffer spec
CircularBufferConfig::CircularBufferConfig(
    uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec) :
    total_size_(total_size), globally_allocated_address_(std::nullopt), dynamic_cb_(false) {
    this->set_config(data_format_spec);
}

// User is expected to use the builder here.
CircularBufferConfig::CircularBufferConfig(uint32_t total_size) :
    total_size_(total_size), globally_allocated_address_(std::nullopt), dynamic_cb_(false) {}

// Dynamic circular buffer spec
CircularBufferConfig::CircularBufferConfig(
    uint32_t total_size, const std::map<uint8_t, tt::DataFormat>& data_format_spec, const Buffer& buffer) :
    total_size_(total_size) {
    this->set_globally_allocated_address(buffer);
    this->set_config(data_format_spec);
}

CircularBufferConfig& CircularBufferConfig::set_page_size(uint8_t buffer_index, uint32_t page_size) {
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
        TT_THROW("Total circular buffer size {} B must be divisible by page size {} B", this->total_size_, page_size);
    }
    // TODO: Should use CIRCULAR_BUFFER_WORD_SIZE_BYTES here
    if (page_size % sizeof(uint32_t) != 0) {
        TT_THROW("Page size must be divisible by sizeof(uint32_t) because buffers holds uint32_t values");
    }

    this->page_sizes_[buffer_index] = page_size;
    return *this;
}

CircularBufferConfig& CircularBufferConfig::set_total_size(uint32_t total_size) {
    if (dynamic_cb_) {
        if (total_size > this->max_size_) {
            TT_ASSERT(
                false,
                "Cannot set circular buffer size to {}. This is larger than the associated dynamically allocated "
                "L1 buffer bank size of {} B",
                total_size,
                this->max_size_);
#ifndef DEBUG
            log_warning(
                "Cannot set circular buffer size to {}. This is larger than the associated dynamically allocated "
                "L1 buffer bank size of {} B and may allow this circular buffer to write outside the allocated "
                "buffer space.",
                total_size,
                this->max_size_);
            if (total_size > this->buffer_size_) {
                TT_THROW(
                    "Cannot set circular buffer size to {}. This is larger than the associated dynamically "
                    "allocated L1 buffer size"
                    "of {} B",
                    total_size,
                    this->buffer_size_);
            }
#endif
        }
    }
    if (total_size == 0) {
        TT_THROW("Total size for circular buffer must be non-zero!");
    }
    this->total_size_ = total_size;
    return *this;
}

CircularBufferConfig& CircularBufferConfig::set_globally_allocated_address(const Buffer& buffer) {
    if (not buffer.is_l1()) {
        TT_THROW("Only L1 buffers can have an associated circular buffer!");
    }
    this->globally_allocated_address_ = buffer.address();
    this->dynamic_cb_ = true;
    this->max_size_ = buffer.aligned_size_per_bank();
    this->buffer_size_ = buffer.aligned_size();
    this->shadow_global_buffer = &buffer;
    if (this->total_size_ > this->max_size_) {
        TT_ASSERT(
            false,
            "Cannot set to globally allocated buffer. Circular buffer size {} B exceeds allocated L1 buffer bank "
            "size of {} B",
            this->total_size_,
            this->max_size_);
#ifndef DEBUG
        log_warning(
            "Circular buffer size {} B exceeds allocated L1 buffer bank size of {} B. This may allow this circular "
            "buffer to write outside the allocated buffer space.",
            this->total_size_,
            this->max_size_);
        if (this->total_size_ > this->buffer_size_) {
            TT_THROW(
                "Cannot set to globally allocated buffer. Circular buffer size {} B exceeds allocated L1 buffer "
                "size of {} B",
                this->total_size_,
                this->buffer_size_);
        }
#endif
    }
    return *this;
}

CircularBufferConfig& CircularBufferConfig::set_tile_dims(uint8_t buffer_index, const Tile& tile) {
    this->tiles_[buffer_index] = tile;
    return *this;
}

const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& CircularBufferConfig::tiles() const {
    return this->tiles_;
}

uint32_t CircularBufferConfig::total_size() const { return this->total_size_; }

std::optional<uint32_t> CircularBufferConfig::globally_allocated_address() const {
    return this->globally_allocated_address_;
}

const std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS>& CircularBufferConfig::data_formats() const {
    return this->data_formats_;
}

const std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS>& CircularBufferConfig::page_sizes() const {
    return this->page_sizes_;
}

CircularBufferConfig::Builder::Builder(CircularBufferConfig& parent, uint8_t buffer_index) :
    parent_(parent), buffer_index_(buffer_index) {
    if (buffer_index > NUM_CIRCULAR_BUFFERS - 1) {
        TT_THROW(
            "Buffer index ({}) exceeds max number of circular buffers per core ({})",
            buffer_index,
            NUM_CIRCULAR_BUFFERS);
    }
    parent_.buffer_indices_.insert(buffer_index_);
}

const CircularBufferConfig::Builder& CircularBufferConfig::Builder::set_data_format(tt::DataFormat data_format) const {
    parent_.data_formats_[buffer_index_] = data_format;
    return *this;
}

const CircularBufferConfig::Builder& CircularBufferConfig::Builder::add_size(uint32_t size) const {
    parent_.total_size_ += size;
    return *this;
}

const CircularBufferConfig::Builder& CircularBufferConfig::Builder::set_page_size(uint32_t page_size) const {
    if (parent_.total_size_ % page_size != 0) {
        TT_THROW("Total circular buffer size {} B must be divisible by page size {} B", parent_.total_size_, page_size);
    }
    if (page_size % sizeof(uint32_t) != 0) {
        TT_THROW("Page size must be divisible by sizeof(uint32_t) because buffers hold uint32_t values");
    }
    parent_.page_sizes_[buffer_index_] = page_size;
    return *this;
}

const CircularBufferConfig::Builder& CircularBufferConfig::Builder::set_tile_dims(const Tile& tile) const {
    parent_.tiles_[buffer_index_] = tile;
    return *this;
}

CircularBufferConfig::Builder CircularBufferConfig::index(uint8_t buffer_index) { return Builder(*this, buffer_index); }

void CircularBufferConfig::set_config(const std::map<uint8_t, tt::DataFormat>& data_format_spec) {
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
    }
}

bool operator==(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs) {
    return lhs.total_size() == rhs.total_size() &&
           lhs.globally_allocated_address() == rhs.globally_allocated_address() &&
           lhs.data_formats() == rhs.data_formats() && lhs.page_sizes() == rhs.page_sizes() &&
           lhs.tiles() == rhs.tiles() && lhs.shadow_global_buffer == rhs.shadow_global_buffer;
}

bool operator!=(const CircularBufferConfig& lhs, const CircularBufferConfig& rhs) { return !(lhs == rhs); }

}  // namespace v0
}  // namespace tt::tt_metal
