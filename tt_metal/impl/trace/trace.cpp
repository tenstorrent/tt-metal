// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <trace.hpp>

#include <memory>

#include <logger.hpp>
#include <tt_metal.hpp>
#include <host_api.hpp>
#include <device.hpp>
#include <command_queue.hpp>
#include <trace.hpp>
#include "tt_metal/trace.hpp"

namespace {
// Labels to make the code more readable
static constexpr bool kBlocking = true;
static constexpr bool kNonBlocking = false;

// Min size is bounded by NOC transfer efficiency
// Max size is bounded by Prefetcher CmdDatQ size
static constexpr uint32_t kExecBufPageMin = 1024;
static constexpr uint32_t kExecBufPageMax = 4096;

// Assumes pages are interleaved across all banks starting at 0
size_t interleaved_page_size(
    const uint32_t buf_size, const uint32_t num_banks, const uint32_t min_size, const uint32_t max_size) {
    // Populate power of 2 numbers within min and max as candidates
    TT_FATAL(
        min_size > 0 and min_size <= max_size,
        "min_size {} not positive and less than or equal to max_size {}.",
        min_size,
        max_size);
    std::vector<uint32_t> candidates;
    candidates.reserve(__builtin_clz(min_size) - __builtin_clz(max_size) + 1);
    for (uint32_t size = 1; size <= max_size; size <<= 1) {
        if (size >= min_size) {
            candidates.push_back(size);
        }
    }
    uint32_t min_waste = -1;
    uint32_t pick = 0;
    // Pick the largest size that minimizes waste
    for (const uint32_t size : candidates) {
        // Pad data to the next fully banked size
        uint32_t fully_banked = num_banks * size;
        uint32_t padded_size = (buf_size + fully_banked - 1) / fully_banked * fully_banked;
        uint32_t waste = padded_size - buf_size;
        if (waste <= min_waste) {
            min_waste = waste;
            pick = size;
        }
    }
    TT_FATAL(
        pick >= min_size and pick <= max_size,
        "pick {} not between min_size {} and max_size {}",
        pick,
        min_size,
        max_size);
    return pick;
}
}  // namespace

namespace tt::tt_metal {

std::atomic<uint32_t> Trace::global_trace_id = 0;

uint32_t Trace::next_id() { return global_trace_id++; }

std::shared_ptr<TraceBuffer> Trace::create_empty_trace_buffer() {
    return std::make_shared<TraceBuffer>(std::make_shared<TraceDescriptor>(), nullptr);
}

void Trace::initialize_buffer(CommandQueue& cq, const std::shared_ptr<TraceBuffer>& trace_buffer) {
    std::vector<uint32_t>& trace_data = trace_buffer->desc->data;
    uint64_t unpadded_size = trace_data.size() * sizeof(uint32_t);
    size_t page_size = interleaved_page_size(
        unpadded_size, cq.device()->allocator()->get_num_banks(BufferType::DRAM), kExecBufPageMin, kExecBufPageMax);
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
    EnqueueWriteBuffer(cq, trace_buffer->buffer, trace_data, kBlocking);
    log_trace(
        LogMetalTrace,
        "Trace issue buffer unpadded size={}, padded size={}, num_pages={}",
        unpadded_size,
        padded_size,
        trace_buffer->buffer->num_pages());
}

// there is a cost to validation, please use it judiciously
void Trace::validate_instance(const TraceBuffer& trace_buffer) {
    std::vector<uint32_t> backdoor_data;
    detail::ReadFromBuffer(trace_buffer.buffer, backdoor_data);
    if (backdoor_data != trace_buffer.desc->data) {
        log_info(LogMetalTrace, "Trace buffer expected: {}", trace_buffer.desc->data);
        log_info(LogMetalTrace, "Trace buffer observed: {}", backdoor_data);
    }
    // add more checks
}

v1::CommandQueueHandle v1::GetCommandQueue(TraceHandle trace) { return trace.cq; }

v1::TraceHandle v1::BeginTraceCapture(CommandQueueHandle cq) {
    const auto tid = v0::BeginTraceCapture(GetDevice(cq), GetId(cq));
    return v1::TraceHandle{cq, tid};
}

void v1::EndTraceCapture(TraceHandle trace) {
    const auto cq = GetCommandQueue(trace);
    v0::EndTraceCapture(GetDevice(cq), GetId(cq), static_cast<std::uint32_t>(trace));
}

void v1::ReplayTrace(TraceHandle trace, bool blocking) {
    const auto cq = GetCommandQueue(trace);
    v0::ReplayTrace(GetDevice(cq), GetId(cq), static_cast<std::uint32_t>(trace), blocking);
}

void v1::ReleaseTrace(TraceHandle trace) {
    const auto cq = GetCommandQueue(trace);
    v0::ReleaseTrace(GetDevice(cq), static_cast<std::uint32_t>(trace));
}

void v1::EnqueueTrace(TraceHandle trace, bool blocking) {
    const auto cq = GetCommandQueue(trace);
    v0::EnqueueTrace(GetDevice(cq)->command_queue(GetId(cq)), static_cast<std::uint32_t>(trace), blocking);
}

}  // namespace tt::tt_metal
