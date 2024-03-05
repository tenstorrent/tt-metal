// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>  // for copy() and assign()
#include <iterator>   // for back_inserter
#include <memory>
#include <string>

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

namespace tt::tt_metal {

Trace::Trace() : trace_complete(false), num_data_bytes(0) { this->cq = std::make_unique<CommandQueue>(this); }

void Trace::record(const TraceNode& trace_node) {
    TT_FATAL(not this->trace_complete, "Cannot record any more for a completed trace");
    this->num_data_bytes += trace_node.num_data_bytes;
    this->history.push_back(trace_node);
}

void Trace::validate() {
    for (const auto& cmd : this->queue().worker_queue) {
        if (cmd.blocking.has_value()) {
            TT_FATAL(cmd.blocking.value() == false, "Blocking commands are not supported in traces");
        }
    }
}

uint32_t Trace::next_trace_id() {
    static uint32_t global_trace_id = 0;
    return global_trace_id++;
}

uint32_t Trace::instantiate(CommandQueue& cq) {
    uint32_t trace_id = next_trace_id();
    cq.trace_ptr = this;

    // Stage the trace commands into device DRAM that the command queue will read from
    // - flatten commands into tightly packed data structure
    // - allocate the data into a DRAM interleaved buffer using 2KB page size
    // - commit the DRAM buffer via an enqueue WB command
    // - map the trace id to the DRAM buffer for later enqueue Trace

    if (trace_instances.count(trace_id)) {
        TT_THROW("Trace ID " + std::to_string(trace_id) + " already exists");
    }

    trace_instances.insert(trace_id);
    return trace_id;
}

}
