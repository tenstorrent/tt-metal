// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trace_buffer.hpp"
#include "tt_metal/impl/device/device_impl.hpp"

namespace tt::tt_metal {

TraceBuffer::TraceBuffer(std::shared_ptr<detail::TraceDescriptor> desc, std::shared_ptr<Buffer> buffer) : desc(desc), buffer(buffer) {}

TraceBuffer::~TraceBuffer() {
    if (this->buffer and this->buffer->device()) {
        this->buffer->device()->trace_buffers_size -= this->buffer->size();
    }
}

}
