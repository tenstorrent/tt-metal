// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/circular_buffer.hpp"

#include "host_api.hpp"
#include "llrt/llrt.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/device/device_impl.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

namespace {

inline void GetBufferAddress(const tt::tt_metal::Buffer *buffer, uint32_t *address_on_host) {
    EnqueueGetBufferAddr(buffer->device()->command_queue(), address_on_host, buffer, false);
}

}
namespace tt {

namespace tt_metal {

// Dynamic CBs will be created with address_ initialized to globally allocated address
// Static CBs will not have address set until their owning Program allocates them
CircularBuffer::CircularBuffer(const CoreRangeSet &core_ranges, const CircularBufferConfig &config) :
    id_(reinterpret_cast<uintptr_t>(this)),
    core_ranges_(core_ranges),
    config_(config),
    locally_allocated_address_(std::nullopt) {
    if (this->config_.total_size() == 0) {
        TT_THROW("Circular Buffer Config Error: Circular buffer size cannot be 0 B");
    }

    for (uint8_t buffer_index = 0; buffer_index < NUM_CIRCULAR_BUFFERS; buffer_index++) {
        std::optional<DataFormat> data_format_spec = this->config_.data_formats().at(buffer_index);
        std::optional<uint32_t> page_size_spec = this->config_.page_sizes().at(buffer_index);

        bool df_set = data_format_spec.has_value();
        bool ps_set = page_size_spec.has_value();
        if (df_set != ps_set) {
            string df_set_str = df_set ? "Data format is set" : "Data format is not set";
            string ps_set_str = ps_set ? "Page size is set" : "Page size is not set";
            TT_THROW("Expected both data format and page size to be set for buffer index {}. {}. {}.", buffer_index, df_set_str, ps_set_str);
        }

        if (df_set and ps_set) {
            this->buffer_indices_.insert(buffer_index);
        }
    }

    if (globally_allocated()) {
        globally_allocated_address_ = config.globally_allocated_address().value();
    }
}

bool CircularBuffer::is_on_logical_corerange(const CoreRange &logical_cr) const {
    return this->core_ranges_.intersects(logical_cr);
}

bool CircularBuffer::is_on_logical_core(const CoreCoord &logical_core) const {
    return this->core_ranges_.core_coord_in_core_ranges(logical_core);
}

bool CircularBuffer::uses_buffer_index(uint32_t buffer_index) const {
    return this->buffer_indices_.find(buffer_index) != this->buffer_indices_.end();
}

uint32_t CircularBuffer::page_size(uint32_t buffer_index) const {
    if (not this->uses_buffer_index(buffer_index)) {
        TT_THROW("Cannot access page size for buffer index {} because circular buffer is not configured on that index", buffer_index);
    }
    uint32_t page_size = this->config_.page_sizes().at(buffer_index).value();
    if (this->size() % page_size != 0) {
        TT_THROW("Total circular buffer size {} B must be divisible by page size {} B", this->size(), page_size);
    }
    return page_size;
}

uint32_t CircularBuffer::num_pages(uint32_t buffer_index) const {
    return this->size() / this->page_size(buffer_index);
}

DataFormat CircularBuffer::data_format(uint32_t buffer_index) const {
    if (not this->uses_buffer_index(buffer_index)) {
        TT_THROW("Cannot access data format for buffer index {} because circular buffer is not configured on that index", buffer_index);
    }
    return this->config_.data_formats().at(buffer_index).value();
}

uint32_t CircularBuffer::address() const {
    if (not locally_allocated_address_.has_value() and not this->globally_allocated()) {
        TT_THROW("Circular buffer has not been allocated, cannot request address at this time!");
    }

    return this->globally_allocated() ? globally_allocated_address_
                                      : locally_allocated_address_.value();
}

void CircularBuffer::assign_global_address() {
    GetBufferAddress(config_.shadow_global_buffer, &globally_allocated_address_);
}

}  // namespace tt_metal

}  // namespace tt
