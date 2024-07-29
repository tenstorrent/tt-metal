// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/graph_tracking.hpp"


using namespace tt::tt_metal;


size_t GraphTracker::add_processor(const std::shared_ptr<IGraphProcessor>& new_processor) {
    processors.push_back(new_processor);
    return processors.size();
}

void GraphTracker::add_hook(const std::shared_ptr<IGraphHooks>& new_hook) {
    hook = new_hook;
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

void GraphTracker::track_allocate_cb(const CoreRange &core_range, uint64_t addr, uint64_t size) {
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_allocate_cb(core_range, addr, size);
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

bool GraphTracker::block_run_program() {
    if (hook == nullptr) {
        return false;
    }
    return hook->block_run_program();
}
