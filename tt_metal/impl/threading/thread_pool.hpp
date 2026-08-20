// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "impl/context/context_types.hpp"

namespace tt::tt_metal {

class IDevice;

class ThreadPool {
public:
    virtual ~ThreadPool() = default;
    // Enqueue a function for the thread-pool to execute. The device_idx argument corresponds to the
    // physical device id this function must be executed for. For certain implementations (for example
    // DeviceBoundThreadPool), the worker thread "closest" to the physical device will be used to execute
    // the specified task. This can lead to better host performance. If not specified, the thread-pool
    // will choose a thread based on a round robin distribution strategy.
    //
    // Non-virtual so that every task, regardless of pool implementation or call site, gets any active
    // graph-capture context propagated onto the worker thread that runs it. Graph-capture state is
    // thread_local, so without this a task would run with an empty processor list and silently drop the
    // events it fires. This is a no-op unless a capture is active. See GraphTracker's threading contract.
    void enqueue(std::function<void()>&& f, std::optional<uint32_t> device_idx = std::nullopt);
    virtual void wait() = 0;

protected:
    virtual void enqueue_impl(std::function<void()>&& f, std::optional<uint32_t> device_idx) = 0;
    // True when enqueue runs the task synchronously on the calling thread. Such a pool never crosses a
    // thread boundary, so the task already sees the caller's graph-capture state and enqueue skips
    // propagation rather than paying to copy a processor vector into a task that runs where it came from.
    virtual bool runs_inline() const { return false; }
};

// API accespting the number of threads to spawn in the pool. Will bind each thread to a CPU core, but the
// binding strategy will not be NUMA aware. Used for testing and benchmarking host-code.
std::shared_ptr<ThreadPool> create_device_bound_thread_pool(ContextId context_id, int num_threads);
// API accepting the physical devices the pool will be bound to. The threads will be bound to CPU cores in a
// NUMA aware manner (will be "closest" to the device it serves). Used for production data-paths.
// All physical devices must belong to the same context ID.
std::shared_ptr<ThreadPool> create_device_bound_thread_pool(
    ContextId context_id, const std::vector<tt::tt_metal::IDevice*>& physical_devices);
std::shared_ptr<ThreadPool> create_passthrough_thread_pool(ContextId context_id);
}  // namespace tt::tt_metal
