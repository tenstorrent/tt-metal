// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/memory_tracking/buffer_allocation_observer.hpp"

#include <algorithm>
#include <shared_mutex>
#include <vector>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

namespace {

// Process-wide, and never emptied: an observer is registered once during device init and lives
// for the life of the process. Guarded by a shared_mutex because the read side is on the buffer
// allocation path and the write side happens once.
std::shared_mutex& observers_mutex() {
    static std::shared_mutex mutex;
    return mutex;
}

std::vector<std::shared_ptr<BufferAllocationObserver>>& observers() {
    static std::vector<std::shared_ptr<BufferAllocationObserver>> registry;
    return registry;
}

}  // namespace

bool register_buffer_allocation_observer_once(
    const std::type_info& type, const std::function<std::shared_ptr<BufferAllocationObserver>()>& factory) {
    std::unique_lock lock(observers_mutex());

    auto& registry = observers();
    const bool already_registered = std::any_of(registry.begin(), registry.end(), [&](const auto& observer) {
        return observer != nullptr && typeid(*observer) == type;
    });
    if (already_registered) {
        return false;
    }

    auto observer = factory();
    TT_FATAL(observer != nullptr, "register_buffer_allocation_observer_once: factory returned nullptr");
    // Bound to a reference first: typeid() of a shared_ptr dereference is an operand with a side
    // effect, which -Wpotentially-evaluated-expression rejects.
    const BufferAllocationObserver& created = *observer;
    TT_FATAL(
        typeid(created) == type,
        "register_buffer_allocation_observer_once: factory produced a {}, but the type asked about was {}",
        typeid(created).name(),
        type.name());

    registry.push_back(std::move(observer));
    return true;
}

void notify_buffer_allocated(const Buffer* buffer) {
    std::shared_lock lock(observers_mutex());
    for (auto& observer : observers()) {
        observer->track_allocate(buffer);
    }
}

void notify_buffer_deallocated(Buffer* buffer) {
    std::shared_lock lock(observers_mutex());
    for (auto& observer : observers()) {
        observer->track_deallocate(buffer);
    }
}

}  // namespace tt::tt_metal
