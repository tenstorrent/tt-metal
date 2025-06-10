// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trace.hpp"

#include <command_queue.hpp>
#include <device.hpp>
#include <host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <cstddef>
#include <memory>
#include <variant>
#include <vector>

#include "allocator_types.hpp"
#include "assert.hpp"
#include "buffer.hpp"
#include "buffer_types.hpp"
#include "math.hpp"
#include "tt_metal/impl/trace/dispatch.hpp"

namespace tt::tt_metal {

std::atomic<uint32_t> Trace::global_trace_id = 0;

uint32_t Trace::next_id() { return global_trace_id++; }

std::shared_ptr<TraceBuffer> Trace::create_empty_trace_buffer() {
    return std::make_shared<TraceBuffer>(std::make_shared<TraceDescriptor>(), nullptr);
}

void Trace::initialize_buffer(CommandQueue& cq, const std::shared_ptr<TraceBuffer>& trace_buffer) {
    std::vector<uint32_t>& trace_data = trace_buffer->desc->data;
    uint64_t unpadded_size = trace_data.size() * sizeof(uint32_t);
    size_t page_size = trace_dispatch::compute_interleaved_trace_buf_page_size(
        unpadded_size, cq.device()->allocator()->get_num_banks(BufferType::DRAM));
    uint64_t padded_size = round_up(unpadded_size, page_size);
    size_t numel_padding = (padded_size - unpadded_size) / sizeof(uint32_t);
    if (numel_padding > 0) {
        trace_data.resize(trace_data.size() + numel_padding, 0 /*padding value*/);
    }
    const auto current_trace_buffers_size = cq.device()->get_trace_buffers_size();
    cq.device()->set_trace_buffers_size(current_trace_buffers_size + padded_size);
    auto trace_region_size = cq.device()->allocator()->get_config().trace_region_size;
    TT_FATAL(
        cq.device()->get_trace_buffers_size() <= trace_region_size,
        "Creating trace buffers of size {}B on device {}, but only {}B is allocated for trace region.",
        cq.device()->get_trace_buffers_size(),
        cq.device()->id(),
        trace_region_size);
    // Commit trace to device DRAM
    trace_buffer->buffer =
        Buffer::create(cq.device(), padded_size, page_size, BufferType::TRACE, TensorMemoryLayout::INTERLEAVED);
    EnqueueWriteBuffer(cq, trace_buffer->buffer, trace_data, true /* blocking */);
    log_trace(
        LogMetalTrace,
        "Trace issue buffer unpadded size={}, padded size={}, num_pages={}",
        unpadded_size,
        padded_size,
        trace_buffer->buffer->num_pages());
}

}  // namespace tt::tt_metal
