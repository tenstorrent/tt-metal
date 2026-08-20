// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt_stl/assert.hpp>

#include "impl/allocator/allocator.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {

namespace {

thread_local std::vector<size_t> pending_traceback_ids;
thread_local std::vector<size_t> retired_traceback_ids;
thread_local std::vector<std::unordered_set<const AllocatorImpl*>> corruptible_allocation_scope_stack;
thread_local std::vector<std::string> allocation_context_stack;
const std::string empty_context;

}  // namespace

bool trace_allocation_tracking_enabled() { return llrt::RunTimeOptions::get_trace_allocation_tracking_enabled(); }

bool trace_allocation_diagnostics_enabled() { return llrt::RunTimeOptions::get_trace_allocation_diagnostics_enabled(); }

bool trace_allocation_skip_program_cache_enabled() {
    return llrt::RunTimeOptions::get_trace_allocation_skip_program_cache_enabled();
}

void push_allocation_context(std::string_view ctx) { allocation_context_stack.emplace_back(ctx); }

void pop_allocation_context() {
    TT_ASSERT(!allocation_context_stack.empty(), "pop_allocation_context called with empty stack");
    allocation_context_stack.pop_back();
}

const std::string& current_allocation_context() {
    return allocation_context_stack.empty() ? empty_context : allocation_context_stack.back();
}

bool allocation_context_contains(std::string_view ctx) {
    return std::any_of(
        allocation_context_stack.begin(), allocation_context_stack.end(), [ctx](const std::string& entry) {
            return entry == ctx;
        });
}

void AllocatorImpl::push_corruptible_allocation_scope(const std::vector<AllocatorImpl*>& allocators) {
    corruptible_allocation_scope_stack.emplace_back(allocators.begin(), allocators.end());
}

void AllocatorImpl::pop_corruptible_allocation_scope() {
    TT_ASSERT(!corruptible_allocation_scope_stack.empty(), "pop_corruptible_allocation_scope called with empty stack");
    corruptible_allocation_scope_stack.pop_back();
}

bool AllocatorImpl::in_corruptible_allocation_scope() const {
    return std::any_of(
        corruptible_allocation_scope_stack.rbegin(),
        corruptible_allocation_scope_stack.rend(),
        [this](const auto& allocators) { return allocators.contains(this); });
}

void AllocatorImpl::clear_trace_allocation_state() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (traceback_capture_enabled_) {
        for (const auto& allocation : unsafe_allocation_contexts_) {
            retired_traceback_ids.push_back(allocation.first);
        }
    }
    unsafe_allocation_contexts_.clear();
    unsafe_tracked_ids_by_manager_and_trace_.clear();
    active_traces_by_manager_.clear();
    allocations_unsafe_ = false;
}

void AllocatorImpl::register_active_trace(SubDeviceManagerId manager_id, const distributed::MeshTraceId& trace_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_traces_by_manager_[manager_id].insert(trace_id).second) {
        return;
    }

    if (tracking_enabled_) {
        unsafe_tracked_ids_by_manager_and_trace_[manager_id].try_emplace(trace_id);
    } else {
        allocations_unsafe_ = true;
    }
}

void AllocatorImpl::unregister_active_trace(SubDeviceManagerId manager_id, const distributed::MeshTraceId& trace_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto active_manager_it = active_traces_by_manager_.find(manager_id);
    if (active_manager_it == active_traces_by_manager_.end() || active_manager_it->second.erase(trace_id) == 0) {
        return;
    }
    if (active_manager_it->second.empty()) {
        active_traces_by_manager_.erase(active_manager_it);
    }

    allocations_unsafe_ = !tracking_enabled_ && !active_traces_by_manager_.empty();
    if (!tracking_enabled_) {
        return;
    }

    auto unsafe_manager_it = unsafe_tracked_ids_by_manager_and_trace_.find(manager_id);
    TT_ASSERT(unsafe_manager_it != unsafe_tracked_ids_by_manager_and_trace_.end());
    auto trace_it = unsafe_manager_it->second.find(trace_id);
    TT_ASSERT(trace_it != unsafe_manager_it->second.end());
    auto removed_buffer_ids = std::move(trace_it->second);
    unsafe_manager_it->second.erase(trace_it);
    if (unsafe_manager_it->second.empty()) {
        unsafe_tracked_ids_by_manager_and_trace_.erase(unsafe_manager_it);
    }
    for (size_t buffer_unique_id : removed_buffer_ids) {
        this->retire_buffer_if_unreferenced(buffer_unique_id);
    }
}

void AllocatorImpl::unregister_active_traces(SubDeviceManagerId manager_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_traces_by_manager_.erase(manager_id) == 0) {
        return;
    }

    allocations_unsafe_ = !tracking_enabled_ && !active_traces_by_manager_.empty();
    if (!tracking_enabled_) {
        return;
    }

    auto unsafe_manager_it = unsafe_tracked_ids_by_manager_and_trace_.find(manager_id);
    TT_ASSERT(unsafe_manager_it != unsafe_tracked_ids_by_manager_and_trace_.end());
    std::unordered_set<size_t> removed_buffer_ids;
    for (const auto& trace_buffers : unsafe_manager_it->second) {
        removed_buffer_ids.insert(trace_buffers.second.begin(), trace_buffers.second.end());
    }
    unsafe_tracked_ids_by_manager_and_trace_.erase(unsafe_manager_it);
    for (size_t buffer_unique_id : removed_buffer_ids) {
        this->retire_buffer_if_unreferenced(buffer_unique_id);
    }
}

void AllocatorImpl::retire_buffer_if_unreferenced(size_t buffer_unique_id) {
    for (const auto& manager_traces : unsafe_tracked_ids_by_manager_and_trace_) {
        for (const auto& trace_buffers : manager_traces.second) {
            if (trace_buffers.second.contains(buffer_unique_id)) {
                return;
            }
        }
    }

    const bool was_tracked = unsafe_allocation_contexts_.erase(buffer_unique_id) > 0;
    if (was_tracked && traceback_capture_enabled_) {
        retired_traceback_ids.push_back(buffer_unique_id);
    }
}

void AllocatorImpl::record_allocation_if_unsafe(Buffer* buffer) {
    if (buffer->buffer_type() == BufferType::TRACE || unsafe_tracked_ids_by_manager_and_trace_.empty() ||
        allocation_context_contains("trace_storage")) {
        return;
    }

    const auto& allocation_context = current_allocation_context();
    if (this->in_corruptible_allocation_scope() ||
        (skip_program_cache_ && allocation_context.starts_with("program_cache:"))) {
        return;
    }

    for (auto& manager_traces : unsafe_tracked_ids_by_manager_and_trace_) {
        for (auto& trace_buffers : manager_traces.second) {
            trace_buffers.second.insert(buffer->unique_id());
        }
    }
    unsafe_allocation_contexts_[buffer->unique_id()] = allocation_context;
    if (traceback_capture_enabled_) {
        pending_traceback_ids.push_back(buffer->unique_id());
    }
}

std::unordered_map<size_t, std::string> AllocatorImpl::get_unsafe_tracked_ids(
    SubDeviceManagerId manager_id, const distributed::MeshTraceId& trace_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<size_t, std::string> result;
    auto manager_it = unsafe_tracked_ids_by_manager_and_trace_.find(manager_id);
    if (manager_it == unsafe_tracked_ids_by_manager_and_trace_.end()) {
        return result;
    }
    auto trace_it = manager_it->second.find(trace_id);
    if (trace_it == manager_it->second.end()) {
        return result;
    }

    std::unordered_set<size_t> allocated_buffer_ids;
    allocated_buffer_ids.reserve(allocated_buffers_.size());
    for (const auto* buffer : allocated_buffers_) {
        allocated_buffer_ids.insert(buffer->unique_id());
    }

    std::vector<size_t> retired_ids;
    for (size_t buffer_unique_id : trace_it->second) {
        if (!allocated_buffer_ids.contains(buffer_unique_id)) {
            retired_ids.push_back(buffer_unique_id);
            continue;
        }
        auto context_it = unsafe_allocation_contexts_.find(buffer_unique_id);
        result.emplace(
            buffer_unique_id, context_it == unsafe_allocation_contexts_.end() ? std::string{} : context_it->second);
    }

    // Keep deallocation identical to the ordinary hot path. Retire stale
    // accounting lazily when a trace is checked instead.
    for (size_t buffer_unique_id : retired_ids) {
        for (auto& manager_traces : unsafe_tracked_ids_by_manager_and_trace_) {
            for (auto& trace_buffers : manager_traces.second) {
                trace_buffers.second.erase(buffer_unique_id);
            }
        }
        const bool was_tracked = unsafe_allocation_contexts_.erase(buffer_unique_id) > 0;
        if (was_tracked && traceback_capture_enabled_) {
            retired_traceback_ids.push_back(buffer_unique_id);
        }
    }
    return result;
}

void AllocatorImpl::remove_unsafe_tracked_id(size_t buffer_unique_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& manager_traces : unsafe_tracked_ids_by_manager_and_trace_) {
        for (auto& trace_buffers : manager_traces.second) {
            trace_buffers.second.erase(buffer_unique_id);
        }
    }
    const bool was_tracked = unsafe_allocation_contexts_.erase(buffer_unique_id) > 0;
    if (was_tracked && traceback_capture_enabled_) {
        retired_traceback_ids.push_back(buffer_unique_id);
    }
}

std::vector<size_t> AllocatorImpl::drain_pending_traceback_ids() {
    std::vector<size_t> result;
    result.swap(pending_traceback_ids);
    return result;
}

std::vector<size_t> AllocatorImpl::drain_retired_traceback_ids() {
    std::vector<size_t> result;
    result.swap(retired_traceback_ids);
    return result;
}

}  // namespace tt::tt_metal
