// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/graph/graph_tracking.hpp"

namespace tt::tt_metal {

bool GraphTracker::is_enabled() const {
    return (not processors.empty());
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

void GraphTracker::track_allocate(Buffer* buffer, bool bottom_up) {
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_allocate(buffer, bottom_up);
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

void GraphTracker::track_allocate_cb(const CoreRangeSet &core_range_set, uint64_t addr, uint64_t size) {
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_allocate_cb(core_range_set, addr, size);
    }
}

void GraphTracker::track_deallocate_cb() {
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_deallocate_cb();
    }
}

void GraphTracker::track_program(Program* program) {
    TT_ASSERT(program);
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_program(program);
    }
}

bool GraphTracker::hook_allocate(Buffer* buffer, bool bottom_up) {
    if (hook == nullptr)
        return false;

    return hook->hook_allocate(buffer, bottom_up);
}

bool GraphTracker::hook_deallocate(Buffer* buffer) {
    if (hook == nullptr)
        return false;
    return hook->hook_deallocate(buffer);
}

bool GraphTracker::hook_program(tt::tt_metal::Program* program) {
    if (hook == nullptr) {
        return false;
    }
    return hook->hook_program(program);
}

const std::vector<std::shared_ptr<IGraphProcessor>>& GraphTracker::get_processors() const {
    return processors;
}

const std::shared_ptr<IGraphHooks>& GraphTracker::get_hook() const {
    return hook;
}

void GraphTracker::clear() {
    processors.clear();
    hook = nullptr;
}

void GraphTracker::clear_hook() {
    hook = nullptr;
}

}
