// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "emule_deferred_mesh_dispatch.hpp"

#include <cstdlib>
#include <mutex>
#include <set>

#include "emulated_program_runner.hpp"

namespace tt::tt_metal::emule {
namespace {

struct DeferredMeshDispatchState {
    std::mutex mutex;
    bool pending = false;
    std::set<const void*> queues;
};

DeferredMeshDispatchState& deferred_state() {
    static DeferredMeshDispatchState state;
    return state;
}

}  // namespace

bool deferred_mesh_dispatch_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("TT_EMULE_DEFER_MESH_DISPATCH");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

bool deferred_mesh_dispatch_has_queue(const void* queue) {
    auto& state = deferred_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.queues.contains(queue);
}

void register_deferred_mesh_dispatch_queue(const void* queue) {
    auto& state = deferred_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.pending = true;
    state.queues.insert(queue);
}

void flush_deferred_mesh_dispatch() {
    auto& state = deferred_state();
    bool had_pending = false;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        had_pending = state.pending;
        if (had_pending) {
            state.pending = false;
            state.queues.clear();
        }
    }

    // run_mesh_dispatch takes the dispatch mutex and can re-enter the queue. Run outside the state
    // lock; clearing the registration set and executing the generation are separate transitions.
    if (had_pending) {
        run_mesh_dispatch();
    }
}

}  // namespace tt::tt_metal::emule
