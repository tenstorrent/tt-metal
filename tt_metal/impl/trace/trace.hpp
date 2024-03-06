// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include <algorithm>
// #include <chrono>
// #include <fstream>
// #include <thread>
#include <functional>
#include <memory>
#include <utility>

#include "impl/buffers/buffer.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/program/program.hpp"

namespace tt::tt_metal {

using std::pair;
using std::reference_wrapper;
using std::set;
using std::shared_ptr;
using std::tuple;
using std::unique_ptr;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using std::weak_ptr;

class CommandQueue;
enum class EnqueueCommandType;

// Trace instance id to buffer mapping mananged via instantiate and release calls
// a static map keeps trace buffers alive until explicitly released by the user
static unordered_map<uint32_t, shared_ptr<Buffer>> trace_buffer_pool;

class Trace {
   // TODO: delete the extra bloat not needed once implementation is complete
   private:
    struct TraceNode {
        DeviceCommand command;
        const vector<uint32_t> data;
        EnqueueCommandType command_type;
        uint32_t num_data_bytes;
    };

    bool trace_complete;
    vector<TraceNode> history;
    uint32_t num_data_bytes;

    // Trace queue used to capture commands
    unique_ptr<CommandQueue> tq;

    static uint32_t next_trace_id();
    void record(const TraceNode& trace_node);
    void validate();

    friend class CommandQueue;
    friend class EnqueueProgramCommand;
    friend CommandQueue& BeginTrace(Trace& trace);
    friend void EndTrace(Trace& trace);
    friend void EnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking);
    friend void EnqueueWriteBuffer(CommandQueue& cq, std::variant<reference_wrapper<Buffer>, shared_ptr<Buffer> > buffer, vector<uint32_t>& src, bool blocking);

   public :
    Trace();
    CommandQueue& queue() const { return *tq; };
    uint32_t instantiate(CommandQueue& tq);  // return a unique trace id
    static bool has_instance(uint32_t trace_id) { return trace_buffer_pool.find(trace_id) != trace_buffer_pool.end(); }
};

}
