// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include <algorithm>
// #include <chrono>
// #include <fstream>
// #include <thread>
#include <memory>
#include <utility>

#include "tt_metal/common/base.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/program/program.hpp"

namespace tt::tt_metal {

using std::pair;
using std::set;
using std::shared_ptr;
using std::tuple;
using std::unique_ptr;
using std::weak_ptr;

class CommandQueue;
enum class EnqueueCommandType;

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
    std::set<uint32_t> trace_instances;
    std::unique_ptr<CommandQueue> cq;
    static uint32_t next_trace_id();
    void record(const TraceNode& trace_node);
    void validate();

    friend class CommandQueue;
    friend class EnqueueProgramCommand;
    friend CommandQueue& BeginTrace(Trace& trace);
    friend void EndTrace(Trace& trace);
    friend void EnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking);

   public:
    Trace();
    CommandQueue& queue() const { return *cq; };
    uint32_t instantiate(CommandQueue& cq);  // return a unique trace id
};

}
