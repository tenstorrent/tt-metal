// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

enum class TraceState {
    EMPTY,
    CAPTURING,
    CAPTURED,
    INSTANTIATING,
    READY
};

class Trace {
   private:
    friend class EnqueueProgramCommand;
    friend void EnqueueTrace(CommandQueue& cq, uint32_t tid, bool blocking);

    TraceState state;

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

    static uint32_t next_id();

   public :
    Trace();
    ~Trace() {
        TT_FATAL(this->state != TraceState::CAPTURING, "Trace capture incomplete before destruction!");
        TT_FATAL(this->state != TraceState::INSTANTIATING, "Trace instantiation incomplete before destruction!");
    }

    // Return the captured trace queue
    CommandQueue& queue() const { return *tq; };

    // Stages a trace buffer into device DRAM via the CQ passed in and returns a unique trace id
    uint32_t instantiate(CommandQueue& tq);

    // Trace capture, validation, and query methods
    void begin_capture();
    void end_capture();
    void validate();

    // Thread-safe accessors to manage trace instances
    static bool has_instance(const uint32_t tid);
    static void add_instance(const uint32_t tid, shared_ptr<Buffer> buffer);
    static void remove_instance(const uint32_t tid);
    static void release_all();  // note all instances across all devices are released
    static shared_ptr<Buffer> get_instance(const uint32_t tid);
};

}  // namespace tt::tt_metal
