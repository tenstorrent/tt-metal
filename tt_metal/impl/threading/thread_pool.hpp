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
    virtual void enqueue(std::function<void()>&& f, std::optional<uint32_t> device_idx = std::nullopt) = 0;
    virtual void wait() = 0;
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

// Binds the calling thread to every CPU core on `numa_node`. Node granularity rather than one core: these
// are long-lived data-path threads whose first-touched memory must stay on the node they run on, and
// pinning one to a single core serializes it against everything else placed there.
void bind_current_thread_to_numa_node(int numa_node);

// Binds `bytes` at `base` to `numa_node`. Call before the pages are faulted, so placement does not depend
// on which thread touches them first.
void bind_memory_to_numa_node(void* base, size_t bytes, int numa_node);
}  // namespace tt::tt_metal
