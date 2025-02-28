// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <memory>

#include <assert.hpp>

namespace tt::tt_metal {

// Provides an interface for keeping a collection of TraceBuffers keyed by TraceId.
template <typename TraceId, typename TraceBufferType>
class TraceBufferPool {
public:
    // Emplaces `trace_buffer` by `trace_id`.
    // Throws if there is already an entry for `trace_id`.
    std::shared_ptr<TraceBufferType> emplace(TraceId trace_id, std::shared_ptr<TraceBufferType> trace_buffer);

    // Erases trace buffer for `trace_id`.
    // Throws if no trace buffer exists for `trace_id`.
    void erase(TraceId trace_id);

    // Returns a trace buffer, or nullptr if it doesn't exist.
    std::shared_ptr<TraceBufferType> get(TraceId trace_id);

private:
    std::unordered_map<TraceId, std::shared_ptr<TraceBufferType>> trace_buffers_;
};

template <typename TraceId, typename TraceBufferType>
std::shared_ptr<TraceBufferType> TraceBufferPool<TraceId, TraceBufferType>::emplace(
    TraceId trace_id, std::shared_ptr<TraceBufferType> trace_buffer) {
    auto [it, emplaced] = trace_buffers_.emplace(trace_id, std::move(trace_buffer));
    TT_FATAL(emplaced, "Trace buffer with trace_id {} already exists", trace_id);
    return it->second;
}

template <typename TraceId, typename TraceBufferType>
void TraceBufferPool<TraceId, TraceBufferType>::erase(TraceId trace_id) {
    auto it = trace_buffers_.find(trace_id);
    TT_FATAL(it != trace_buffers_.end(), "Trace buffer with trace_id {} does not exist", trace_id);
    trace_buffers_.erase(trace_id);
}

template <typename TraceId, typename TraceBufferType>
std::shared_ptr<TraceBufferType> TraceBufferPool<TraceId, TraceBufferType>::get(TraceId trace_id) {
    auto it = trace_buffers_.find(trace_id);
    return it != trace_buffers_.end() ? it->second : nullptr;
}

}  // namespace tt::tt_metal
