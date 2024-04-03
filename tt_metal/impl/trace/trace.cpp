// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>

#include "dispatch/device_command.hpp"
#include "logger.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

namespace tt::tt_metal {

// List of supported commands for tracing
// const unordered_set<EnqueueCommandType> trace_supported_commands = {
//     EnqueueCommandType::ENQUEUE_PROGRAM,
// };

unordered_map<uint32_t, shared_ptr<Buffer>> Trace::buffer_pool;
std::mutex Trace::pool_mutex;

Trace::Trace() : state(TraceState::EMPTY) {
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
            TT_FATAL(cmd.blocking.value() == false, "Blocking commands are not supported in traces");
        }
        // if (trace_supported_commands.find(cmd.type) == trace_supported_commands.end()) {
        //     TT_THROW("Unsupported command type for tracing");
        // }
    }
}

uint32_t Trace::next_id() {
    static uint32_t global_trace_id = 0;
    return global_trace_id++;
}

// Stage the trace commands into device DRAM as an interleaved buffer for execution
uint32_t Trace::instantiate(CommandQueue& cq) {
    this->state = TraceState::INSTANTIATING;
    uint32_t tid = next_id();
    TT_FATAL(this->has_instance(tid) == false, "Trace ID " + std::to_string(tid) + " already exists");

    // Record the captured Host API as commands via bypass mode
    SystemMemoryManager& cq_manager = cq.device()->sysmem_manager();
    cq_manager.set_bypass_mode(kEnableCQBypass, kClearBuffer);
    for (auto cmd : this->queue().worker_queue) {
        log_debug(LogMetalTrace, "Trace::instantiate found command {}", cmd.type);
        // cmd.blocking = kNonBlocking;  // skip the blocking check for bypass mode
        cq.run_command(cmd);
    }
    cq.wait_until_empty();

    // Extract the data from the bypass buffer and allocate it into a DRAM buffer
    SystemMemoryManager& manager = cq.hw_command_queue().manager;
    std::vector<uint32_t>& data = cq_manager.get_bypass_data();
    uint64_t data_size = data.size() * sizeof(uint32_t);

    // TODO: add CQ_PREFETCH_EXEC_BUF_END command and pad to the next page

    size_t numel_page = DeviceCommand::PROGRAM_PAGE_SIZE / sizeof(uint32_t);
    size_t numel_padding = numel_page - data.size() % numel_page;
    if (numel_padding > 0) {
        data.resize(data.size() + numel_padding, 0);
    }
    log_debug(LogMetalTrace, "Trace buffer size = {}, padded size = {}, data = {}", data_size, data.size() * sizeof(uint32_t), data);

    // Commit the trace buffer to device DRAM in a blocking fashion before clearing the bypass mode and data
    auto buffer = std::make_shared<Buffer>(
        cq.device(),
        data.size() * sizeof(uint32_t),
        DeviceCommand::PROGRAM_PAGE_SIZE,
        BufferType::DRAM,
        TensorMemoryLayout::INTERLEAVED);
    cq_manager.set_bypass_mode(kDisableCQBypass, kClearBuffer);
    EnqueueWriteBuffer(cq, buffer, data, kBlocking);

    // Pin the trace buffer in memory until explicitly released by the user
    this->add_instance(tid, buffer);
    this->state = TraceState::READY;
    return tid;
}

bool Trace::has_instance(const uint32_t tid) {
    return _safe_pool([&] {
        return Trace::buffer_pool.find(tid) != Trace::buffer_pool.end();
    });
}

void Trace::add_instance(const uint32_t tid, shared_ptr<Buffer> buffer) {
    _safe_pool([&] {
        TT_FATAL(Trace::buffer_pool.find(tid) == Trace::buffer_pool.end());
        Trace::buffer_pool.insert({tid, buffer});
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

shared_ptr<Buffer> Trace::get_instance(const uint32_t tid) {
    return _safe_pool([&] {
        TT_FATAL(Trace::buffer_pool.find(tid) != Trace::buffer_pool.end());
        return Trace::buffer_pool[tid];
    });
}

}  // namespace tt::tt_metal
