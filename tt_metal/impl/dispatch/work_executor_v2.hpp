// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "tt_metal/impl/dispatch/thread_safe_queue.hpp"

namespace tt {
namespace tt_metal {

struct WorkExecutorV2 {

    thread_safe_queue_t<std::function<void()>> worker_queue;
    std::thread worker_thread;
    bool run = true;

    WorkExecutorV2(std::size_t cpu_core);

    WorkExecutorV2(const WorkExecutorV2&) = delete;
    WorkExecutorV2& operator=(const WorkExecutorV2&) = delete;

    WorkExecutorV2(WorkExecutorV2&& other);

    WorkExecutorV2& operator=(WorkExecutorV2&& other);

    void push_work(std::function<void()>&& computation);

    void synchronize();

    ~WorkExecutorV2();
};

}  // namespace tt_metal

}  // namespace tt
