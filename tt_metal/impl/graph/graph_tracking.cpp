// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <graph_tracking.hpp>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

thread_local std::vector<std::shared_ptr<IGraphProcessor>> GraphTracker::processors;
thread_local std::shared_ptr<IGraphHooks> GraphTracker::hook;

nlohmann::json IGraphProcessor::end_capture() { return nullptr; }

GraphTracker& GraphTracker::instance() {
    static GraphTracker tracker;
    return tracker;
}

bool GraphTracker::is_enabled() const {
    return std::any_of(processors.begin(), processors.end(), [](const auto& p) { return p->is_capture_processor(); });
}

void GraphTracker::push_processor(const std::shared_ptr<IGraphProcessor>& new_processor) {
    processors.push_back(new_processor);
}

void GraphTracker::pop_processor() {
    TT_ASSERT(not processors.empty(), "No processor to pop");
    processors.pop_back();
}

bool GraphTracker::add_hook(const std::shared_ptr<IGraphHooks>& new_hook) {
    if (hook) {
        return false;
    }
    hook = new_hook;
    return true;
}

void GraphTracker::track_allocate(const Buffer* buffer) {
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_allocate(buffer);
    }
}

void GraphTracker::track_deallocate(Buffer* buffer) {
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_deallocate(buffer);
    }
}

void GraphTracker::track_allocate_cb(
    const CoreRangeSet& core_range_set,
    uint64_t addr,
    uint64_t size,
    bool is_globally_allocated,
    const IDevice* device) {
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_allocate_cb(core_range_set, addr, size, is_globally_allocated, device);
    }
}

void GraphTracker::track_deallocate_cb(const IDevice* device) {
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_deallocate_cb(device);
    }
}

void GraphTracker::track_program(Program* program, const IDevice* device) {
    TT_ASSERT(program);
    TT_ASSERT(device);
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_program(program, device);
    }
}

bool GraphTracker::hook_allocate(const Buffer* buffer) {
    if (hook == nullptr) {
        return false;
    }

    bool hooked = hook->hook_allocate(buffer);
    if (hooked) {
        std::lock_guard<std::mutex> lock(hooked_buffers_mutex);
        bool inserted = hooked_buffers.insert(buffer).second;
        TT_FATAL(inserted, "Can't hook allocation of a buffer which is already allocated");
    }
    return hooked;
}

bool GraphTracker::hook_deallocate(Buffer* buffer) {
    if (hook == nullptr) {
        return false;
    }

    bool hooked = hook->hook_deallocate(buffer);
    if (hooked) {
        std::lock_guard<std::mutex> lock(hooked_buffers_mutex);
        auto buffer_it = hooked_buffers.find(buffer);
        if (buffer_it == hooked_buffers.end()) {
            log_warning(tt::LogMetal, "Can't hook deallocation of a buffer which allocation wasn't hooked");
        } else {
            hooked_buffers.erase(buffer_it);
        }
    }
    return hooked;
}

bool GraphTracker::hook_write_to_device(const tt::tt_metal::Buffer* buffer) {
    if (hook == nullptr) {
        return false;
    }
    return hook->hook_write_to_device(buffer);
}

bool GraphTracker::hook_write_to_device(const tt::tt_metal::distributed::MeshBuffer* mesh_buffer) {
    if (hook == nullptr) {
        return false;
    }
    return hook->hook_write_to_device(mesh_buffer);
}

bool GraphTracker::hook_read_from_device(tt::tt_metal::Buffer* buffer) {
    if (hook == nullptr) {
        return false;
    }
    return hook->hook_read_from_device(buffer);
}

bool GraphTracker::hook_read_from_device(const tt::tt_metal::distributed::MeshBuffer* mesh_buffer) {
    if (hook == nullptr) {
        return false;
    }
    return hook->hook_read_from_device(mesh_buffer);
}

bool GraphTracker::hook_program(tt::tt_metal::Program* program) {
    if (hook == nullptr) {
        return false;
    }
    return hook->hook_program(program);
}

const std::vector<std::shared_ptr<IGraphProcessor>>& GraphTracker::get_processors() const { return processors; }

const std::shared_ptr<IGraphHooks>& GraphTracker::get_hook() const { return hook; }

std::vector<std::shared_ptr<IGraphProcessor>> GraphTracker::exchange_processors(
    std::vector<std::shared_ptr<IGraphProcessor>> incoming) {
    std::vector<std::shared_ptr<IGraphProcessor>> previous = std::move(processors);
    processors = std::move(incoming);
    return previous;
}

std::function<void()> GraphTracker::wrap_with_current_context(std::function<void()> task) {
    // Fast path: nothing worth observing on this thread, so hand back the task
    // untouched. This must test `is_enabled()` rather than `processors.empty()`:
    // background processors (e.g. ShmTrackingProcessor) are registered at device
    // init and never removed, so `processors` is non-empty on an ordinary run and
    // an emptiness test would put every dispatch on the copying path.
    if (!is_enabled()) {
        return task;
    }
    return [context = processors, task = std::move(task)]() mutable {
        auto& tracker = GraphTracker::instance();
        auto previous = tracker.exchange_processors(std::move(context));
        // Put the worker thread's own stack back even if `task` throws. The exchange also hands
        // `context` back, so moving out of it above costs no allocation yet leaves the wrapper
        // callable again -- std::function does not promise single invocation.
        struct Restore {
            GraphTracker& tracker;
            std::vector<std::shared_ptr<IGraphProcessor>>& previous;
            std::vector<std::shared_ptr<IGraphProcessor>>& context;
            ~Restore() { context = tracker.exchange_processors(std::move(previous)); }
        } restore{tracker, previous, context};
        task();
    };
}

void GraphTracker::clear() {
    processors.clear();
    clear_hook();
}

void GraphTracker::clear_hook() {
    {
        std::lock_guard<std::mutex> lock(hooked_buffers_mutex);
        hooked_buffers.clear();
    }
    hook = nullptr;
}

}  // namespace tt::tt_metal
