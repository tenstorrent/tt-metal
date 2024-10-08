// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <variant>

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/trace/trace_buffer.hpp"

namespace tt::tt_metal {
inline namespace v0 {

class Trace {
   private:
    static std::atomic<uint32_t> global_trace_id;

   public:
    Trace() = delete;

    static uint32_t next_id();

    // Thread-safe accessors to manage trace instances
    static void validate_instance(const TraceBuffer& trace_buffer);
    static void initialize_buffer(CommandQueue& cq, std::shared_ptr<TraceBuffer> trace_buffer);
    static std::shared_ptr<TraceBuffer> create_empty_trace_buffer();
};

}  // namespace v0
}  // namespace tt::tt_metal
