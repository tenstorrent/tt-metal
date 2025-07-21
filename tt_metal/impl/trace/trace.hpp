// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <variant>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/command_queue.hpp>

namespace tt {
namespace tt_metal {
class CommandQueue;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {
class TraceBuffer;

class Trace {
private:
    static std::atomic<uint32_t> global_trace_id;

public:
    Trace() = delete;

    static uint32_t next_id();

    // Thread-safe accessors to manage trace instances
    static void initialize_buffer(CommandQueue& cq, const std::shared_ptr<TraceBuffer>& trace_buffer);
    static std::shared_ptr<TraceBuffer> create_empty_trace_buffer();
};

}  // namespace tt::tt_metal
