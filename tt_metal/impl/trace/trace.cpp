// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>

#include "logger.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

namespace tt::tt_metal {

// List of supported commands for tracing
const unordered_set<EnqueueCommandType> trace_supported_commands = {
    EnqueueCommandType::ENQUEUE_PROGRAM,
};

unordered_map<uint32_t, shared_ptr<Buffer>> Trace::buffer_pool;
std::mutex Trace::pool_mutex;

Trace::Trace() : capture_complete(false), num_data_bytes(0) {
    this->tq = std::make_unique<CommandQueue>(this);
}

void Trace::record(const TraceNode& trace_node) {
    this->num_data_bytes += trace_node.num_data_bytes;
    this->history.push_back(trace_node);
}

void Trace::validate() {
    for (const auto& cmd : this->queue().worker_queue) {
        if (cmd.blocking.has_value()) {
            TT_FATAL(cmd.blocking.value() == false, "Blocking commands are not supported in traces");
        }
        if (trace_supported_commands.find(cmd.type) == trace_supported_commands.end()) {
            TT_THROW("Unsupported command type for tracing");
        }
    }
}

uint32_t Trace::next_id() {
    static uint32_t global_trace_id = 0;
    return global_trace_id++;
}

uint32_t Trace::instantiate(CommandQueue& cq) {
    uint32_t tid = next_id();
    TT_FATAL(this->has_instance(tid) == false, "Trace ID " + std::to_string(tid) + " already exists");

    // Stage the trace commands into device DRAM that the command queue will read from
    // - flatten commands into tightly packed data structure
    // - allocate the data into a DRAM interleaved buffer using 2KB page size
    // - commit the DRAM buffer via an enqueue WB command
    // - map the trace id to the DRAM buffer for later enqueue Trace

    this->history.clear();
    for (auto cmd : this->queue().worker_queue) {
        TT_FATAL(
            trace_supported_commands.find(cmd.type) != trace_supported_commands.end(),
            "Unsupported command type found in trace");
        cmd.trace = *this;
        // #6024: Trace command flattening to a buffer should avoid using CQ
        // however the use of it offloads work to a worker thread for speedup
        // while can be queued up behind other commands and requires sync before
        // yielding back to the main thread (eg. this->history usage below requires wait_until_empty)
        cq.run_command(cmd);
    }
    cq.wait_until_empty();

    vector<uint32_t> trace_data;

    SystemMemoryManager& manager = cq.hw_command_queue().manager;
    uint32_t data_size = 0;
    for (const auto& node : this->history) {
        trace_data.insert(trace_data.end(), node.data.begin(), node.data.end());
        data_size += node.num_data_bytes;
    }
    tt::log_debug(tt::LogDispatch, "Trace data size = {}, trace num_bytes = {}", data_size, this->num_data_bytes);
    TT_FATAL(data_size == this->num_data_bytes, "Data size mismatch in trace");

    auto trace_buffer = std::make_shared<Buffer>(
        cq.device(),
        this->num_data_bytes,
        DeviceCommand::PROGRAM_PAGE_SIZE,
        BufferType::DRAM,
        TensorMemoryLayout::INTERLEAVED);

    // Pin the trace buffer in memory through trace memory mgmt
    this->add_instance(tid, trace_buffer);

    // Commit the trace buffer to device DRAM in a blocking fashion
    // Optional optimization: use a non-blocking enqueue WB command
    EnqueueWriteBuffer(cq, trace_buffer, trace_data, true);
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

}  // namespace tt::tt_metal
