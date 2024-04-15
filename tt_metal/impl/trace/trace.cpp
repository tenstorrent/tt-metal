// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/trace/trace.hpp"
#include <memory>
#include <string>

#include "dispatch/device_command.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/trace/trace.hpp"

namespace {
// Labels to make the code more readable
static constexpr bool kBlocking = true;
static constexpr bool kNonBlocking = false;

// Min size is bounded by NOC transfer efficiency
// Max size is bounded by Prefetcher CmdDatQ size
static constexpr uint32_t kExecBufPageMin = 1024;
static constexpr uint32_t kExecBufPageMax = 4096;

// Assumes pages are interleaved across all banks starting at 0
size_t interleaved_page_size(const uint32_t buf_size, const uint32_t num_banks, const uint32_t min_size, const uint32_t max_size) {
    // Populate power of 2 numbers within min and max as candidates
    TT_FATAL(min_size > 0 and min_size <= max_size);
    vector<uint32_t> candidates;
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
    TT_FATAL(pick >= min_size and pick <= max_size);
    return pick;
}
}

namespace tt::tt_metal {

unordered_map<uint32_t, TraceBuffer> Trace::buffer_pool;
std::mutex Trace::pool_mutex;

Trace::Trace() {
    this->reset();
}

void Trace::reset() {
    this->state = TraceState::EMPTY;
    this->tq = std::make_unique<CommandQueue>(*this);
}

void Trace::begin_capture() {
    TT_FATAL(this->state == TraceState::EMPTY, "Cannot begin capture in a non-empty state");
    TT_FATAL(this->queue().empty(), "Cannot begin trace on one that already captured commands");
    this->state = TraceState::CAPTURING;
}

void Trace::end_capture() {
    TT_FATAL(this->state == TraceState::CAPTURING, "Cannot end capture that has not begun");
    this->validate();
    this->state = TraceState::CAPTURED;
}

void Trace::validate() {
    for (const auto& cmd : this->queue().worker_queue) {
        if (cmd.blocking.has_value()) {
            // The workload being traced needs to be self-contained and not require any host interaction
            // Blocking by definition yields control back to the host, consider breaking it into multiple traces
            TT_FATAL(cmd.blocking.value() == false, "Only non-blocking commands can be captured in Metal Trace!");
        }
    }
}

uint32_t Trace::next_id() {
    static uint32_t global_trace_id = 0;
    return global_trace_id++;
}

// Stage the trace commands into device DRAM as an interleaved buffer for execution
uint32_t Trace::instantiate(CommandQueue& cq) {
    this->state = TraceState::INSTANTIATING;
    auto desc = std::make_shared<detail::TraceDescriptor>();

    // Record the captured Host API as commands via trace_commands,
    desc->data = cq.hw_command_queue().record_commands(desc, [&]() {
        for (auto cmd : this->queue().worker_queue) {
            cq.run_command(cmd);
        }
        cq.wait_until_empty();
    });

    uint32_t tid = Trace::instantiate(cq, desc);
    this->state = TraceState::READY;
    return tid;
}

uint32_t Trace::instantiate(CommandQueue& cq, shared_ptr<detail::TraceDescriptor> desc) {
    uint32_t tid = Trace::next_id();
    TT_FATAL(Trace::has_instance(tid) == false, "Trace ID " + std::to_string(tid) + " already exists");

    vector<uint32_t>& data = desc->data;

    // Add command to terminate the trace buffer
    DeviceCommand command_sequence(CQ_PREFETCH_CMD_BARE_MIN_SIZE);
    command_sequence.add_prefetch_exec_buf_end();
    for (int i = 0; i < command_sequence.size_bytes() / sizeof(uint32_t); i++) {
        data.push_back(((uint32_t*)command_sequence.data())[i]);
    }

    // Pad the trace buffer to the next fully banked page
    uint64_t unpadded_size = data.size() * sizeof(uint32_t);
    size_t page_size = interleaved_page_size(
        unpadded_size, cq.device()->num_banks(BufferType::DRAM), kExecBufPageMin, kExecBufPageMax);
    size_t numel_page = page_size / sizeof(uint32_t);
    size_t numel_padding = numel_page - data.size() % numel_page;
    if (numel_padding > 0) {
        data.resize(data.size() + numel_padding, 0/*padding value*/);
    }
    uint64_t padded_size = data.size() * sizeof(uint32_t);

    // Commit the trace buffer to device DRAM
    auto buffer = std::make_shared<Buffer>(cq.device(), padded_size, page_size, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED);

    EnqueueWriteBuffer(cq, buffer, data, kBlocking);
    Finish(cq);  // clear side effects flag

    // Pin the trace buffer in memory until explicitly released by the user
    Trace::add_instance(tid, {desc, buffer});
    log_trace(LogMetalTrace,
        "Trace {} instantiated with completion buffer num_entries={}, issue buffer unpadded size={}, padded size={}, num_pages={}",
        tid, desc->num_completion_q_reads, unpadded_size, padded_size, padded_size / page_size);
    return tid;
}

bool Trace::has_instance(const uint32_t tid) {
    return _safe_pool([&] {
        return Trace::buffer_pool.find(tid) != Trace::buffer_pool.end();
    });
}

void Trace::add_instance(const uint32_t tid, TraceBuffer buf) {
    _safe_pool([&] {
        TT_FATAL(Trace::buffer_pool.find(tid) == Trace::buffer_pool.end());
        Trace::buffer_pool.insert({tid, buf});
    });
}

void Trace::remove_instance(const uint32_t tid) {
    _safe_pool([&] {
        TT_FATAL(Trace::buffer_pool.find(tid) != Trace::buffer_pool.end());
        Trace::buffer_pool.erase(tid);
    });
}

void Trace::release_all() {
    _safe_pool([&] {
        Trace::buffer_pool.clear();
    });
}

// there is a cost to validation, please use it judiciously
void Trace::validate_instance(const uint32_t tid) {
    vector<uint32_t> backdoor_data;
    auto trace_inst = Trace::get_instance(tid);
    detail::ReadFromBuffer(trace_inst.buffer, backdoor_data);
    if (backdoor_data != trace_inst.desc->data) {
        log_info(LogMetalTrace, "Trace buffer expected: {}", trace_inst.desc->data);
        log_info(LogMetalTrace, "Trace buffer observed: {}", backdoor_data);
        TT_THROW("Trace buffer data mismatch for instance {}", tid);
    }
    // add more checks
}

TraceBuffer Trace::get_instance(const uint32_t tid) {
    return _safe_pool([&] {
        TT_FATAL(Trace::buffer_pool.find(tid) != Trace::buffer_pool.end());
        return Trace::buffer_pool[tid];
    });
}

}  // namespace tt::tt_metal
