// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <thread>

namespace tt::tt_metal {

class ThreadPool {
public:
    virtual ~ThreadPool() = default;
    virtual void enqueue(std::function<void()>&& f, uint32_t thread_idx = 0) = 0;
    virtual void wait() = 0;
};

std::shared_ptr<ThreadPool> create_boost_thread_pool(int num_threads);
std::shared_ptr<ThreadPool> create_device_bound_thread_pool(int num_threads, uint32_t logical_cpu_offset = 0);

}  // namespace tt::tt_metal
