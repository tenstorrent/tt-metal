// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json_fwd.hpp>
#include <stdint.h>
#include <any>
#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
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

// Snapshot of the per-thread capture state (`processors` + `hook`). Copyable so
// it can be moved onto a worker thread and installed there for the duration of
// an offloaded task, letting that thread observe the same capture as the thread
// that enqueued the work. See GraphTracker::wrap_with_current_context().
struct CaptureContext {
    std::vector<std::shared_ptr<IGraphProcessor>> processors;
    std::shared_ptr<IGraphHooks> hook;

    bool empty() const { return processors.empty() && hook == nullptr; }
};

// Process-wide singleton that fans out op-dispatch events to registered
// processors and consults an optional hook to intercept buffer / program
// operations.
//
// Threading contract:
//   * The processor stack (`processors`) and `hook` are *per-thread*. A
//     `push_processor` / capture / `pop_processor` sequence is scoped to the
//     calling thread; ops dispatched on other threads are not observed by
//     that capture unless the capturing thread's context is explicitly
//     propagated (see below).
//   * Work that is offloaded onto worker/dispatch threads while a capture is
//     active (e.g. multi-threaded MeshWorkload compile, CCL/collective
//     dispatch) would otherwise run with an empty per-thread `processors`
//     list and silently drop its events. `wrap_with_current_context()`
//     snapshots the enqueuing thread's context and installs it on the worker
//     thread for the duration of the task (restoring the worker's previous
//     state afterwards). Storage stays per-thread, so this does not
//     reintroduce the concurrent push/pop race that motivated making
//     `processors` / `hook` thread_local (#44668); the shared `IGraphProcessor`
//     is itself internally synchronized.
//   * `hooked_buffers` is process-wide and guarded by `hooked_buffers_mutex`.
//     This is the only piece of GraphTracker state that is shared across
//     threads.
class GraphTracker {
public:
    GraphTracker(const GraphTracker&) = delete;
    GraphTracker(GraphTracker&&) = delete;

    static GraphTracker& instance();

    bool is_enabled() const;

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

    bool hook_allocate(const Buffer* buffer);

    bool hook_deallocate(Buffer* buffer);

    bool hook_write_to_device(const Buffer* buffer);

    bool hook_write_to_device(const distributed::MeshBuffer* mesh_buffer);

    bool hook_read_from_device(Buffer* buffer);

    bool hook_read_from_device(const distributed::MeshBuffer* mesh_buffer);

    bool hook_program(tt::tt_metal::Program* program);

    const std::vector<std::shared_ptr<IGraphProcessor>>& get_processors() const;

    const std::shared_ptr<IGraphHooks>& get_hook() const;

    // Return a copy of the calling thread's capture state (processors + hook).
    CaptureContext capture_context() const;

    // Replace the calling thread's capture state with `context`, returning the
    // previous state so a caller can restore it later.
    CaptureContext install_context(CaptureContext context);

    // Wrap `task` so that, when it later runs (possibly on another thread), the
    // calling thread's *current* capture context is installed for the duration
    // of the task and then restored. When no capture is active on the calling
    // thread this returns `task` unchanged (zero added overhead on the hot
    // dispatch path). Snapshotting happens now, on the calling thread, so it
    // must be invoked on the thread that owns the capture (i.e. at enqueue
    // time, before the work is handed to a worker thread).
    std::function<void()> wrap_with_current_context(std::function<void()> task);

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
}  // namespace tt::tt_metal
