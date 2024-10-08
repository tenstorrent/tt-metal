// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <unordered_set>

#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/tile/tile.hpp"

namespace tt::tt_metal {
inline namespace v0 {

using CBHandle = uintptr_t;

class CircularBufferConfig {
   public:
    // Static circular buffer spec
    CircularBufferConfig(uint32_t total_size, const std::map<uint8_t, tt::DataFormat> &data_format_spec) :
        total_size_(total_size), globally_allocated_address_(std::nullopt), dynamic_cb_(false) {
        this->set_config(data_format_spec);
    }

    // Dynamic circular buffer spec
    CircularBufferConfig(
        uint32_t total_size, const std::map<uint8_t, tt::DataFormat> &data_format_spec, const Buffer &buffer) :
        total_size_(total_size),
        dynamic_cb_(true),
        max_size_(buffer.size()) {
        if (not buffer.is_l1()) {
            TT_THROW("Only L1 buffers can have an associated circular buffer!");
        }
        if (total_size > buffer.size()) {
            TT_THROW(
                "Requested {} B but dynamic circular buffer cannot be larger than allocated L1 buffer of {} B",
                total_size,
                buffer.size());
        }
        this->set_globally_allocated_address(buffer);
        this->set_config(data_format_spec);
    }

    CircularBufferConfig set_page_size(uint8_t buffer_index, uint32_t page_size) {
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
                "Total circular buffer size {} B must be divisible by page size {} B",
                this->total_size_,
                page_size);
        }
        if (page_size % sizeof(uint32_t) != 0) {
            TT_THROW("Page size must be divisible by sizeof(uint32_t) because buffers holds uint32_t values");
        }

        this->page_sizes_[buffer_index] = page_size;
        return *this;
    }

    CircularBufferConfig set_total_size(uint32_t total_size) {
        if (dynamic_cb_ and total_size > this->max_size_.value()) {
            TT_THROW(
                "Cannot grow circular buffer to {} B. This is larger than associated dynamically allocated L1 buffer "
                "of {} B",
                total_size,
                this->max_size_.value());
        }
        if (total_size == 0) {
            TT_THROW("Total size for circular buffer must be non-zero!");
        }
        this->total_size_ = total_size;
        return *this;
    }

    CircularBufferConfig set_globally_allocated_address(const Buffer &buffer) {
        if (not buffer.is_l1()) {
            TT_THROW("Only L1 buffers can have an associated circular buffer!");
        }
        this->globally_allocated_address_ = buffer.address();
        this->dynamic_cb_ = true;
        this->max_size_ = buffer.size();
        this->shadow_global_buffer = &buffer;
        return *this;
    }

    CircularBufferConfig set_tile_dims(uint8_t buffer_index, const Tile& tile) {
        this->tiles_[buffer_index] = tile;
        return *this;
    }

    const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS> &tiles() const {
        return this->tiles_;
    }

    uint32_t total_size() const { return this->total_size_; }

    std::optional<uint32_t> globally_allocated_address() const { return this->globally_allocated_address_; }

    const std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS> &data_formats() const {
        return this->data_formats_;
    }

    const std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS> &page_sizes() const { return this->page_sizes_; }
    const Buffer* shadow_global_buffer;
   private:
    void set_config(const std::map<uint8_t, tt::DataFormat> &data_format_spec) {
        if (data_format_spec.size() > NUM_CIRCULAR_BUFFERS) {
            TT_THROW(
                "Only {} circular buffer slots are available but data formats are specified for {} indices",
                NUM_CIRCULAR_BUFFERS,
                data_format_spec.size());
        }

        for (const auto &[buffer_index, data_format] : data_format_spec) {
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

    uint32_t total_size_ = 0;
    std::optional<uint32_t> globally_allocated_address_ = std::nullopt;
    std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS> data_formats_;
    std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS> page_sizes_;
    std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS> tiles_;
    std::unordered_set<uint8_t> buffer_indices_;
    bool dynamic_cb_ = false;
    // `max_size_` is used to ensure that total size does not grow beyond associated buffer size
    std::optional<uint32_t> max_size_ = std::nullopt;
};

}  // namespace v0
}  // namespace tt::tt_metal
