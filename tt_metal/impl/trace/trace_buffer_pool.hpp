// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <memory>

#include <assert.hpp>

#include "sub_device_types.hpp"

namespace tt::tt_metal {

// TraceBuffers are valid and can be enqueued for replay as long as the SubDeviceManagerId is active on device.
// This class provides a convenient abstraction for keeping a collection of TraceBuffers along with the
// SubDeviceManagerId they were captured on, keyed by TraceId.
template <typename TraceId, typename TraceBufferType>
class TraceBufferPool {
public:
    // Emplaces `trace_buffer` and `sub_device_manager_id` by `trace_id`.
    // Throws if there is already an entry for `trace_id`.
    std::shared_ptr<TraceBufferType> emplace(
        TraceId trace_id, SubDeviceManagerId sub_device_manager_id, std::shared_ptr<TraceBufferType> trace_buffer);

    // Erases entry for `trace_id`.
    // Throws if no trace exists for `trace_id`.
    void erase(TraceId trace_id);

    struct TraceEntry {
        SubDeviceManagerId sub_device_manager_id;
        std::shared_ptr<TraceBufferType> trace_buffer;
    };

    // Returns an entry for `trace_id`, or nullopt if it doesn't exist.
    std::optional<TraceEntry> get_trace(TraceId trace_id);

private:
    std::unordered_map<TraceId, TraceEntry> traces_;
};

template <typename TraceId, typename TraceBufferType>
std::shared_ptr<TraceBufferType> TraceBufferPool<TraceId, TraceBufferType>::emplace(
    TraceId trace_id, SubDeviceManagerId sub_device_manager_id, std::shared_ptr<TraceBufferType> trace_buffer) {
    auto [it, emplaced] = traces_.emplace(trace_id, TraceEntry{sub_device_manager_id, std::move(trace_buffer)});
    TT_FATAL(emplaced, "Entry with trace_id {} already exists", trace_id);
    return it->second.trace_buffer;
}

template <typename TraceId, typename TraceBufferType>
void TraceBufferPool<TraceId, TraceBufferType>::erase(TraceId trace_id) {
    auto it = traces_.find(trace_id);
    TT_FATAL(it != traces_.end(), "Entry with trace_id {} does not exist", trace_id);
    traces_.erase(trace_id);
}

template <typename TraceId, typename TraceBufferType>
std::optional<typename TraceBufferPool<TraceId, TraceBufferType>::TraceEntry>
TraceBufferPool<TraceId, TraceBufferType>::get_trace(TraceId trace_id) {
    auto it = traces_.find(trace_id);
    return it != traces_.end() ? std::optional<TraceEntry>{it->second} : std::nullopt;
}

}  // namespace tt::tt_metal
