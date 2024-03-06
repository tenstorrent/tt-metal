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

// namespace trace {
// static unordered_map<uint32_t, shared_ptr<Buffer>> buffer_pool;
// }

class Trace {
   // TODO: delete the extra bloat not needed once implementation is complete
   private:
    struct TraceNode {
        DeviceCommand command;
        const vector<uint32_t> data;
        EnqueueCommandType command_type;
        uint32_t num_data_bytes;
    };

    bool capture_complete;
    vector<TraceNode> history;
    uint32_t num_data_bytes;

    // Trace queue used to capture commands
    unique_ptr<CommandQueue> tq;

    // Trace instance id to buffer mapping mananged via instantiate and release calls
    // a static map keeps trace buffers alive until explicitly released by the user
    static unordered_map<uint32_t, shared_ptr<Buffer>> buffer_pool;

    // Thread safe accessor to trace::buffer_pool
    static std::mutex pool_mutex;
    template <typename Func>
    static inline auto _safe_pool(Func func) {
        std::lock_guard<std::mutex> lock(Trace::pool_mutex);
        return func();
    }

    // Internal methods
    void record(const TraceNode& trace_node);
    static uint32_t next_id();

    // Friends of trace class
    friend class EnqueueProgramCommand;
    friend void EnqueueTrace(CommandQueue& cq, uint32_t tid, bool blocking);
    friend void EnqueueWriteBuffer(CommandQueue& cq, std::variant<reference_wrapper<Buffer>, shared_ptr<Buffer> > buffer, vector<uint32_t>& src, bool blocking);

   public :
    Trace();
    ~Trace() { TT_FATAL(this->captured(), "Trace capture incomplete before destruction!"); }

    // Return the captured trace queue
    CommandQueue& queue() const { return *tq; };

    // Stages a trace buffer into device DRAM via the CQ passed in and returns a unique trace id
    uint32_t instantiate(CommandQueue& tq);

    // Trace capture validation, completion query and setter
    void validate();
    bool captured() { return this->capture_complete; }
    void captured(bool complete) { this->capture_complete = complete; }

    // Thread-safe accessors to manage trace instances
    static bool has_instance(const uint32_t tid);
    static void add_instance(const uint32_t tid, shared_ptr<Buffer> buffer);
    static void remove_instance(const uint32_t tid);
};

}  // namespace tt::tt_metal
