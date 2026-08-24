// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json_fwd.hpp>
#include <stdint.h>
#include <any>
#include <array>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal {
class Buffer;
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

class Program;

struct TrackedArgument {
    std::any value;
    std::string (*to_string_fn)(const std::any&);
};

// Forward declaration only – the definition lives in ttnn/graph/graph_serialization.hpp
// which pulls in <reflect> and tt_stl/reflection.hpp.  This keeps the public API header
// free of those heavyweight dependencies.
template <typename T>
std::string serialize_tracked_arg(const std::any& a);

class IGraphProcessor {
public:
    enum class RunMode {
        NORMAL,      // running everything as is
        NO_DISPATCH  // don't do memory allocations and program runs.
    };

    IGraphProcessor() = default;

    // Returns false for background processors that are always
    // registered but should not make GraphTracker::is_enabled() return true.
    virtual bool is_capture_processor() const { return true; }

    virtual void track_allocate(const tt::tt_metal::Buffer* /*buffer*/) {};

    virtual void track_deallocate(tt::tt_metal::Buffer* /*buffer*/) {};

    virtual void track_allocate_cb(
        const CoreRangeSet& /*core_range_set*/,
        uint64_t /*addr*/,
        uint64_t /*size*/,
        bool /*is_globally_allocated*/,
        const IDevice* /*device*/) {};

    // borrows_memory: a view onto a tensor's buffer, so counting its bytes again double-counts.
    virtual void track_allocate_dataflow_buffer(
        const CoreRangeSet& /*core_range_set*/,
        uint64_t /*addr*/,
        uint64_t /*size*/,
        bool /*borrows_memory*/,
        const IDevice* /*device*/) {};

    virtual void track_allocate_scratchpad(
        const CoreRangeSet& /*core_range_set*/, uint64_t /*addr*/, uint64_t /*size*/, const IDevice* /*device*/) {};

    // Releases every kind of program-scope L1 above: they share one lifetime.
    virtual void track_deallocate_cb(const IDevice* /*device*/) {};

    virtual void track_program(tt::tt_metal::Program* /*program*/, const IDevice* /*device*/) {};

    virtual void track_function_start(
        std::string_view /*function_name*/, std::span<TrackedArgument> /*input_parameters*/){};

    virtual void track_function_end() {};
    virtual void track_function_end(const std::any& /*output_tensors*/) {};

    virtual void begin_capture(RunMode /*mode*/){};

    virtual nlohmann::json end_capture();

    virtual ~IGraphProcessor() = default;
};

// Support to GraphProcessor for track nonclosed scopes, during exceptions processing.
struct GraphFunctionAbort {
    std::string reason;
    bool unwind_all = false;
};

class IGraphHooks {
public:
    IGraphHooks() = default;
    virtual bool hook_allocate(const tt::tt_metal::Buffer* buffer) = 0;

    virtual bool hook_deallocate(tt::tt_metal::Buffer* buffer) = 0;

    virtual bool hook_program(Program* program) = 0;

    virtual bool hook_write_to_device(const tt::tt_metal::Buffer* buffer) = 0;

    virtual bool hook_read_from_device(tt::tt_metal::Buffer* buffer) = 0;

    virtual bool hook_read_from_device(const tt::tt_metal::distributed::MeshBuffer* mesh_buffer) = 0;

    virtual bool hook_write_to_device(const tt::tt_metal::distributed::MeshBuffer* mesh_buffer) = 0;

    virtual ~IGraphHooks() = default;
};

// Process-wide singleton that fans out op-dispatch events to registered
// processors and consults an optional hook to intercept buffer / program
// operations.
//
// Threading contract:
//   * The processor stack (`processors`) and `hook` are *per-thread*. A
//     `push_processor` / capture / `pop_processor` sequence is scoped to the
//     calling thread; ops dispatched on other threads are not observed by
//     that capture.
//   * `hooked_buffers` is process-wide and guarded by `hooked_buffers_mutex`.
//     This is the only piece of GraphTracker state that is shared across
//     threads.
class GraphTracker {
public:
    GraphTracker(const GraphTracker&) = delete;
    GraphTracker(GraphTracker&&) = delete;

    static GraphTracker& instance();

    bool is_enabled() const;

    // Whether any processor is registered, i.e. whether the track_* calls below do anything.
    // Unlike is_enabled() this also counts non-capture (background) processors, so it matches
    // exactly the condition each track_* call tests before fanning out.
    bool has_processors() const;

    void push_processor(const std::shared_ptr<IGraphProcessor>& processor);
    void pop_processor();

    bool add_hook(const std::shared_ptr<IGraphHooks>& hook);

    void track_allocate(const Buffer* buffer);

    void track_deallocate(Buffer* buffer);

    void track_allocate_cb(
        const CoreRangeSet& core_range_set,
        uint64_t addr,
        uint64_t size,
        bool is_globally_allocated,
        const IDevice* device);

    void track_allocate_dataflow_buffer(
        const CoreRangeSet& core_range_set, uint64_t addr, uint64_t size, bool borrows_memory, const IDevice* device);

    void track_allocate_scratchpad(
        const CoreRangeSet& core_range_set, uint64_t addr, uint64_t size, const IDevice* device);

    void track_deallocate_cb(const IDevice* device);

    void track_program(Program* program, const IDevice* device);

    // NOLINTBEGIN(cppcoreguidelines-missing-std-forward)
    template <class... Args>
    void track_function_start(std::string_view function_name, Args&&... args) {
        if (processors.empty()) {
            return;
        }
        std::array<TrackedArgument, sizeof...(Args)> params{
            TrackedArgument{std::any(std::ref(args)), &serialize_tracked_arg<std::remove_reference_t<Args>>}...};
        for (auto& it : processors) {
            it->track_function_start(function_name, params);
        }
    }
    // NOLINTEND(cppcoreguidelines-missing-std-forward)

    // Track op that doesn't return anything
    void track_function_end() {
        if (processors.empty()) {
            return;
        }
        for (auto& it : processors) {
            it->track_function_end();
        }
    }

    template <class ReturnType>
    void track_function_end(ReturnType& output_tensors) {
        if (processors.empty()) {
            return;
        }
        for (auto& it : processors) {
            it->track_function_end(std::ref(output_tensors));
        }
    }

    // Close a tracked scope that is being left by an exception. There is no output to report.
    void track_function_abort(std::string_view reason);

    // Close every scope the processors of this thread still hold open, marking each aborted.
    void unwind_open_functions(std::string_view reason);

    bool hook_allocate(const Buffer* buffer);

    bool hook_deallocate(Buffer* buffer);

    bool hook_write_to_device(const Buffer* buffer);

    bool hook_write_to_device(const distributed::MeshBuffer* mesh_buffer);

    bool hook_read_from_device(Buffer* buffer);

    bool hook_read_from_device(const distributed::MeshBuffer* mesh_buffer);

    bool hook_program(tt::tt_metal::Program* program);

    const std::vector<std::shared_ptr<IGraphProcessor>>& get_processors() const;

    const std::shared_ptr<IGraphHooks>& get_hook() const;

    void clear();

    void clear_hook();

private:
    GraphTracker() = default;
    ~GraphTracker() = default;

    // Per-thread state. See the class-level threading contract above.
    static thread_local std::vector<std::shared_ptr<IGraphProcessor>> processors;
    static thread_local std::shared_ptr<IGraphHooks> hook;

    std::mutex hooked_buffers_mutex;
    std::unordered_set<const Buffer*> hooked_buffers;
};

// RAII pairing of GraphTracker::track_function_start with its end.
//
// The two calls used to be written as plain statements around the tracked body, so an exception
// thrown in between skipped the end entirely. The processor was then left holding the dead scope:
// the next end closed the wrong function, and every event after it was recorded one level too
// deep, nested under an operation that never finished. That corrupts the whole remainder of a
// capture which outlives the failure, such as the per-test capture behind the TTNN Visualizer
// report.
//
// The destructor closes the scope on both paths and tells them apart, so a failing operation is
// reported as aborted instead of silently unbalancing the trace. Call end() on the success path to
// report the operation's output; anything left unclosed is finished by the destructor.
class ScopedTrackedFunction {
public:
    template <class... Args>
    explicit ScopedTrackedFunction(std::string_view function_name, Args&&... args) :
        entry_uncaught_exceptions_(std::uncaught_exceptions()),
        // Snapshot the processors that will receive the start. Holding the shared_ptrs
        // keeps that set alive if it is popped before this guard ends.
        started_processors_(GraphTracker::instance().get_processors()) {
        GraphTracker::instance().track_function_start(function_name, std::forward<Args>(args)...);
    }

    template <class ReturnType>
    void end(ReturnType& output_tensors) {
        if (std::exchange(ended_, true) || started_processors_.empty()) {
            return;
        }
        for (auto& processor : started_processors_) {
            processor->track_function_end(std::ref(output_tensors));
        }
    }

    void end() {
        if (std::exchange(ended_, true) || started_processors_.empty()) {
            return;
        }
        for (auto& processor : started_processors_) {
            processor->track_function_end();
        }
    }

    // Closes the scope as failed. Call from a catch block, where the message is in reach; the
    // destructor cannot supply one (see below).
    void abort(std::string_view reason) {
        if (std::exchange(ended_, true) || started_processors_.empty()) {
            return;
        }
        for (auto& processor : started_processors_) {
            processor->track_function_end(std::any(GraphFunctionAbort{std::string(reason), false}));
        }
    }

    ~ScopedTrackedFunction() {
        if (ended_ || started_processors_.empty()) {
            return;
        }
        // Destructors must not let an exception escape, least of all while one is already unwinding.
        try {
            if (std::uncaught_exceptions() > entry_uncaught_exceptions_) {
                // No message: std::current_exception() only reports an exception that a handler has
                // begun handling, and stack unwinding runs destructors before any handler is
                // entered, so there is nothing to read here. Callers that want the text must use
                // abort() from a catch block.
                for (auto& processor : started_processors_) {
                    processor->track_function_end(std::any(GraphFunctionAbort{{}, false}));
                }
            } else {
                for (auto& processor : started_processors_) {
                    processor->track_function_end();
                }
            }
        } catch (...) {  // NOLINT(bugprone-empty-catch)
        }
    }

    ScopedTrackedFunction(const ScopedTrackedFunction&) = delete;
    ScopedTrackedFunction(ScopedTrackedFunction&&) = delete;
    ScopedTrackedFunction& operator=(const ScopedTrackedFunction&) = delete;
    ScopedTrackedFunction& operator=(ScopedTrackedFunction&&) = delete;

private:
    int entry_uncaught_exceptions_;
    std::vector<std::shared_ptr<IGraphProcessor>> started_processors_;
    bool ended_ = false;
};
}  // namespace tt::tt_metal
