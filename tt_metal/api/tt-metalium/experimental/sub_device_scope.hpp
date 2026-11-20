// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal::experimental {

// EXPERIMENTAL. This is a best-effort helper, not a guarantee about where operations run, and it may change or be
// removed. Read the known limitations in tech_reports/SubDevices/SubDevices.md (section 1.5) before using it.
//
// A sub-device scope makes the operations you launch from the current thread run on one sub-device.
//
// While a scope is open, a MeshDevice pretends to be just that sub-device. compute_with_storage_grid_size() returns
// a grid only large enough to cover the sub-device's Tensix cores, and get_sub_device_ids() returns only its id.
// Most operations call one of those two to decide which cores to run on, so they end up on the sub-device without
// any changes. An operation that picks its cores some other way is not affected by the scope.
//
// Nothing lists which operations work inside a scope. The safety net is a check at dispatch: if a program has a
// kernel, semaphore, circular buffer or dataflow buffer on another sub-device's cores, it is rejected before it is
// compiled.
//
// Scopes can be nested, and the innermost one wins. The id is looked up in whichever sub-device manager is loaded at
// the time, so with no manager loaded a scope for sub-device 0 does nothing.
void PushCurrentSubDeviceIdForThread(SubDeviceId sub_device_id);
void PopCurrentSubDeviceIdForThread();
std::optional<SubDeviceId> GetCurrentSubDeviceIdForThread();

// Opens a scope when constructed and closes it when destroyed.
class [[nodiscard]] SubDeviceScope {
public:
    explicit SubDeviceScope(SubDeviceId sub_device_id) { PushCurrentSubDeviceIdForThread(sub_device_id); }
    ~SubDeviceScope() { PopCurrentSubDeviceIdForThread(); }
    SubDeviceScope(const SubDeviceScope&) = delete;
    SubDeviceScope& operator=(const SubDeviceScope&) = delete;
    SubDeviceScope(SubDeviceScope&&) = delete;
    SubDeviceScope& operator=(SubDeviceScope&&) = delete;
};

}  // namespace tt::tt_metal::experimental
