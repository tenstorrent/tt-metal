// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
class MeshWorkloadImpl;
}  // namespace distributed

namespace sub_device_scope {

// How many scopes are open across all threads. The helpers below are called on every program launch, so they check
// this counter first. A process that never opens a scope pays for one load and nothing else.
extern std::atomic<uint32_t> num_open;

inline bool any_open() { return num_open.load(std::memory_order_relaxed) != 0; }

const std::vector<SubDeviceId>* ids_for_thread();
void suspend_for_thread();
void resume_for_thread();
void validate(distributed::MeshWorkloadImpl& workload, distributed::MeshDevice* mesh_device);

}  // namespace sub_device_scope

// Returns the id list that MeshDevice::get_sub_device_ids() should report on this thread, or null if the thread is
// not in a scope (or the scope is suspended). The list stays valid until the scope is closed.
inline const std::vector<SubDeviceId>* scoped_sub_device_ids_for_thread() {
    return sub_device_scope::any_open() ? sub_device_scope::ids_for_thread() : nullptr;
}

// Turns the current thread's scope off for as long as this object lives, so MeshDevice describes the whole device
// again. Metal creates one whenever it needs the real device: opening or closing it, changing sub-device managers,
// and dispatching a workload.
class [[nodiscard]] SuspendSubDeviceScope {
public:
    SuspendSubDeviceScope() : suspended_(sub_device_scope::any_open()) {
        if (suspended_) [[unlikely]] {
            sub_device_scope::suspend_for_thread();
        }
    }
    ~SuspendSubDeviceScope() {
        if (suspended_) [[unlikely]] {
            sub_device_scope::resume_for_thread();
        }
    }
    SuspendSubDeviceScope(const SuspendSubDeviceScope&) = delete;
    SuspendSubDeviceScope& operator=(const SuspendSubDeviceScope&) = delete;

private:
    bool suspended_;
};

// Raises if the loaded sub-device manager has no sub-device with this id.
void validate_scoped_sub_device(const IDevice& device, SubDeviceId scoped);

// Does nothing outside a scope. Inside one, raises if the workload has a kernel, semaphore, circular buffer or
// dataflow buffer on another sub-device's cores. Call this before compiling the workload, so that a rejected
// workload is left untouched and can still be enqueued later.
inline void validate_sub_device_scope(distributed::MeshWorkloadImpl& workload, distributed::MeshDevice* mesh_device) {
    if (sub_device_scope::any_open()) [[unlikely]] {
        sub_device_scope::validate(workload, mesh_device);
    }
}

}  // namespace tt::tt_metal
