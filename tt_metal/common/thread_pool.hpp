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
    virtual void enqueue(std::function<void()>&& f) = 0;
    virtual void wait() = 0;
};

std::shared_ptr<ThreadPool> create_boost_thread_pool(int num_threads);

}  // namespace tt::tt_metal
