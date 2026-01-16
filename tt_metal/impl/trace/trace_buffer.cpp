// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trace_buffer.hpp"

#include <device.hpp>
#include <fmt/ranges.h>
#include <tt_metal.hpp>
#include <utility>

#include "buffer.hpp"
#include <tt-logger/tt-logger.hpp>
#include <fmt/core.h>

namespace tt::tt_metal {

TraceBuffer::TraceBuffer(std::shared_ptr<TraceDescriptor> desc, std::shared_ptr<Buffer> buffer) :
    desc(std::move(desc)), buffer(std::move(buffer)) {}

TraceBuffer::~TraceBuffer() {
    if (this->buffer and this->buffer->device()) {
        const auto current_trace_buffers_size = this->buffer->device()->get_trace_buffers_size();
        this->buffer->device()->set_trace_buffers_size(current_trace_buffers_size - this->buffer->size());
    }
}

// there is a cost to validation, please use it judiciously
void TraceBuffer::validate() {
    std::vector<uint32_t> backdoor_data;
    detail::ReadFromBuffer(this->buffer, backdoor_data);
    if (backdoor_data != this->desc->data) {
        log_error(LogMetalTrace, "Trace buffer expected: {}", fmt::join(this->desc->data, ", "));
        log_error(LogMetalTrace, "Trace buffer observed: {}", fmt::join(backdoor_data, ", "));
    }
    // add more checks
}

}  // namespace tt::tt_metal
