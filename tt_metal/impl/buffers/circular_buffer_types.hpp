/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <map>
#include <unordered_set>

#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/buffers/buffer.hpp"

namespace tt::tt_metal {

using CircularBufferID = uintptr_t;

class CircularBufferConfig {
   public:
    CircularBufferConfig(uint32_t total_size, const std::map<uint8_t, tt::DataFormat> &data_format_spec) : total_size_(total_size), globally_allocated_address_(std::nullopt) {
        this->set_config(data_format_spec);
    }

    CircularBufferConfig(uint32_t total_size, const std::map<uint8_t, tt::DataFormat> &data_format_spec, const Buffer &buffer) : total_size_(total_size), globally_allocated_address_(buffer.address()) {
        if (buffer.buffer_type() != BufferType::L1) {
            tt::log_fatal(tt::LogMetal, "Only L1 buffers can have an associated circular buffer!");
        }
        this->set_config(data_format_spec);
    }

    CircularBufferConfig set_page_size(uint8_t buffer_index, uint32_t page_size) {
        if (buffer_index > NUM_CIRCULAR_BUFFERS - 1) {
            log_fatal(tt::LogMetal, "Buffer index ({}) exceeds max number of circular buffers per core ({})", buffer_index, NUM_CIRCULAR_BUFFERS);
        }
        if (this->buffer_indices_.find(buffer_index) == this->buffer_indices_.end()) {
            log_fatal(tt::LogMetal, "Illegal circular buffer index {}. Page size can only be specified for buffer indices configured during config creation", buffer_index);
        }
        if (this->total_size_ % page_size != 0) {
            log_fatal(tt::LogMetal, "Total circular buffer size {} B must be divisible by page size {} B", this->total_size_, page_size);
        }
        if (page_size % sizeof(uint32_t) != 0) {
            log_fatal(tt::LogMetal, "Page size must be divisible by sizeof(uint32_t) because buffers holds uint32_t values");
        }

        this->page_sizes_[buffer_index] = page_size;
        return *this;
    }

    CircularBufferConfig set_total_size(uint32_t total_size) {
        if (total_size == 0) {
            log_fatal(tt::LogMetal, "Total size for circular buffer must be non-zero!");
        }
        this->total_size_ = total_size;
        return *this;
    }

    CircularBufferConfig set_globally_allocated_address(const Buffer &buffer) {
        this->globally_allocated_address_ = buffer.address();
        return *this;
    }

    uint32_t total_size() const { return this->total_size_; }

    std::optional<uint32_t> globally_allocated_address() const { return this->globally_allocated_address_; }

    const std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS> &data_formats() const { return this->data_formats_; }

    const std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS> &page_sizes() const { return this->page_sizes_; }

   private:
    void set_config(const std::map<uint8_t, tt::DataFormat> &data_format_spec) {
        if (data_format_spec.size() > NUM_CIRCULAR_BUFFERS) {
            log_fatal(tt::LogMetal, "Only {} circular buffer slots are available but data formats are specified for {} indices", NUM_CIRCULAR_BUFFERS, data_format_spec.size());
        }

        for (const auto &[buffer_index, data_format] : data_format_spec) {
            if (buffer_index > NUM_CIRCULAR_BUFFERS - 1) {
                log_fatal(tt::LogMetal, "Buffer index ({}) exceeds max number of circular buffers per core ({})", buffer_index, NUM_CIRCULAR_BUFFERS);
            }
            this->data_formats_[buffer_index] = data_format;
            this->buffer_indices_.insert(buffer_index);
        }
    }

    uint32_t total_size_;
    std::optional<uint32_t> globally_allocated_address_;
    std::array<std::optional<tt::DataFormat>, NUM_CIRCULAR_BUFFERS> data_formats_;
    std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS> page_sizes_;
    std::unordered_set<uint8_t> buffer_indices_;
};

} // namespace tt::tt_metal
