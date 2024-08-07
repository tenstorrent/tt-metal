// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "tt_metal/impl/dispatch/thread_safe_queue.hpp"

namespace tt {
namespace tt_metal {

struct WorkExecutorV2 {

    WorkExecutorV2(std::size_t cpu_core, std::size_t device_id);

    WorkExecutorV2(const WorkExecutorV2&) = delete;
    WorkExecutorV2& operator=(const WorkExecutorV2&) = delete;

    WorkExecutorV2(WorkExecutorV2&& other);

    WorkExecutorV2& operator=(WorkExecutorV2&& other);

    ~WorkExecutorV2();

    void push_work(std::function<void()>&& computation, bool blocking = false);

    void synchronize();

    void start();
    void stop();

    private:

    std::size_t cpu_core;
    std::size_t device_id;

    bool running = false;
    thread_safe_queue_t<std::function<void()>> worker_queue;
    std::thread worker_thread;
};

}  // namespace tt_metal

}  // namespace tt
