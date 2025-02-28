// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <memory>

#include <assert.hpp>

namespace tt::tt_metal {

template <typename TraceId, typename TraceBufferType>
class TraceBufferPool {
public:
    std::shared_ptr<TraceBufferType> emplace_trace_buffer(
        TraceId trace_id, std::shared_ptr<TraceBufferType> trace_buffer) {
        auto [it, emplaced] = trace_buffer_pool_.emplace(trace_id, std::move(trace_buffer));
        TT_FATAL(emplaced, "Trace buffer with trace_id {} already exists", trace_id);
        return it->second;
    }

    void release_trace_buffer(TraceId trace_id) { trace_buffer_pool_.erase(trace_id); }

    std::shared_ptr<TraceBufferType> get_trace_buffer(TraceId trace_id) {
        auto it = trace_buffer_pool_.find(trace_id);
        TT_FATAL(it != trace_buffer_pool_.end(), "Trace buffer with trace_id {} not found", trace_id);
        return it->second;
    }

private:
    std::unordered_map<TraceId, std::shared_ptr<TraceBufferType>> trace_buffer_pool_;
};

}  // namespace tt::tt_metal
