// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <graph_tracking.hpp>
#include <internal/graph_tracking.hpp>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

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

namespace internal {

bool register_background_processor_once(
    const std::type_info& type, const std::function<std::shared_ptr<IGraphProcessor>()>& factory) {
    auto& tracker = GraphTracker::instance();
    // One exclusive lock over both the lookup and the insertion. Two separate operations
    // would let concurrent device initialization register the observer twice and double-count
    // every buffer event.
    std::unique_lock lock(tracker.background_processors_mutex);
    const bool already_registered = std::any_of(
        tracker.background_processors.begin(), tracker.background_processors.end(), [&](const auto& processor) {
            return processor != nullptr && typeid(*processor) == type;
        });
    if (already_registered) {
        return false;
    }
    auto processor = factory();
    TT_FATAL(processor != nullptr, "register_background_processor_once: factory returned nullptr");
    // Bound to a reference first: typeid() of a shared_ptr dereference is an operand with a
    // side effect, which -Wpotentially-evaluated-expression rejects.
    const IGraphProcessor& created = *processor;
    TT_FATAL(
        typeid(created) == type,
        "register_background_processor_once: factory produced a {}, but the type asked about was {}",
        typeid(created).name(),
        type.name());
    tracker.background_processors.push_back(std::move(processor));
    return true;
}

}  // namespace internal

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
    for (auto& it : processors) {
        it->track_allocate(buffer);
    }
    // Background processors observe every thread, not just the one that registered them.
    std::shared_lock lock(background_processors_mutex);
    for (auto& it : background_processors) {
        it->track_allocate(buffer);
    }
}

void GraphTracker::track_deallocate(Buffer* buffer) {
    for (auto& it : processors) {
        it->track_deallocate(buffer);
    }
    std::shared_lock lock(background_processors_mutex);
    for (auto& it : background_processors) {
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

void GraphTracker::track_allocate_dataflow_buffer(
    const CoreRangeSet& core_range_set, uint64_t addr, uint64_t size, bool borrows_memory, const IDevice* device) {
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_allocate_dataflow_buffer(core_range_set, addr, size, borrows_memory, device);
    }
}

void GraphTracker::track_allocate_scratchpad(
    const CoreRangeSet& core_range_set, uint64_t addr, uint64_t size, const IDevice* device) {
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_allocate_scratchpad(core_range_set, addr, size, device);
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

namespace {

// A Metal 2.0 program holds its per-core L1 in dataflow buffers and kernel scratchpads, which -
// unlike circular buffers - have no allocation-time tracking hook.
void track_program_l1(GraphTracker& tracker, detail::ProgramImpl& program, const IDevice* device) {
    for (const auto& dfb : program.dataflow_buffers()) {
        // Alias secondaries share the primary's L1 region instead of adding one.
        if (dfb->alias_primary_id.has_value()) {
            continue;
        }
        tracker.track_allocate_dataflow_buffer(
            dfb->core_ranges, /*addr=*/0, dfb->total_size(), dfb->borrows_memory(), device);
    }
    // Scratchpads stack on the same program-scope L1 region as the dataflow buffers, one region per
    // binding: kernels may only share a scratchpad spec across disjoint cores.
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t core_type = 0; core_type < hal.get_programmable_core_type_count(); core_type++) {
        for (const auto& [_, kernel] : program.get_kernels(core_type)) {
            for (const auto& scratchpad : kernel->scratchpad_binding_handles()) {
                tracker.track_allocate_scratchpad(kernel->core_range_set(), /*addr=*/0, scratchpad.size_bytes, device);
            }
        }
    }
}

}  // namespace

void GraphTracker::track_program(Program* program, const IDevice* device) {
    TT_ASSERT(program);
    TT_ASSERT(device);
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_program(program, device);
    }
    // A hooked program never runs, so no allocation will report its L1 later. Addresses are not
    // assigned yet either, matching how a capture reports this program's circular buffers.
    if (hook_program(program)) {
        track_program_l1(*this, program->impl(), device);
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
