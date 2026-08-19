// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/mesh_trace_id.hpp>

namespace tt::tt_metal::distributed {

class MeshDevice;

namespace trace_allocation_tracker {

void register_active_trace(MeshDevice* device, const MeshTraceId& trace_id);
void unregister_active_trace(MeshDevice* device, const MeshTraceId& trace_id);
std::unordered_map<size_t, std::string> get_unsafe_tracked_ids(const MeshDevice* device, const MeshTraceId& trace_id);
void remove_unsafe_tracked_id(MeshDevice* device, size_t buffer_unique_id);
std::vector<size_t> drain_pending_traceback_ids();
std::vector<size_t> drain_retired_traceback_ids();
void push_corruptible_allocation_scope(MeshDevice* device);
void pop_corruptible_allocation_scope(MeshDevice* device);

}  // namespace trace_allocation_tracker
}  // namespace tt::tt_metal::distributed
