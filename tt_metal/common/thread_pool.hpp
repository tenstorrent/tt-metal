// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <vector>
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

std::shared_ptr<ThreadPool> create_boost_thread_pool(int num_threads);
std::shared_ptr<ThreadPool> create_distributed_boost_thread_pool(int num_threads);
// API accespting the number of threads to spawn in the pool. Will bind each thread to a CPU core, but the
// binding strategy will not be NUMA aware. Used for testing and benchmarking host-code.
std::shared_ptr<ThreadPool> create_device_bound_thread_pool(int num_threads);
// API accepting the physical devices the pool will be bound to. The threads will be bound to CPU cores in a
// NUMA aware manner (will be "closest" to the device it serves). Used for production data-paths.
std::shared_ptr<ThreadPool> create_device_bound_thread_pool(
    const std::vector<tt::tt_metal::IDevice*>& physical_devices);
std::shared_ptr<ThreadPool> create_passthrough_thread_pool();
}  // namespace tt::tt_metal
