// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trace_buffer.hpp"

#include <device.hpp>
#include <utility>

#include "buffer.hpp"

namespace tt::tt_metal {

TraceBuffer::TraceBuffer(std::shared_ptr<TraceDescriptor> desc, std::shared_ptr<Buffer> buffer) :
    desc(std::move(desc)), buffer(std::move(buffer)) {}

TraceBuffer::~TraceBuffer() {
    if (this->buffer and this->buffer->device()) {
        const auto current_trace_buffers_size = this->buffer->device()->get_trace_buffers_size();
        this->buffer->device()->set_trace_buffers_size(current_trace_buffers_size - this->buffer->size());
    }
}

}  // namespace tt::tt_metal
