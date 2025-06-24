// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <taskflow/taskflow.hpp>
#include <thread>
#include <stdexcept>

namespace tt::tt_metal::detail {
inline static const size_t EXECUTOR_NTHREADS = std::getenv("TT_METAL_THREADCOUNT")
                                                   ? std::stoi(std::getenv("TT_METAL_THREADCOUNT"))
                                                   : std::thread::hardware_concurrency();

inline tf::Executor& GetExecutor() {
    static tf::Executor exec(EXECUTOR_NTHREADS);
    return exec;
}

// When a taskflow worker thread throws an exception, Taskflow does not propagate this to main thread
// (https://github.com/taskflow/taskflow/issues/479) This wrapper ensures that exceptions are re-thrown by packaged_task
// whenever .get() is called on future object
// https://stackoverflow.com/questions/16344663/is-there-a-packaged-taskset-exception-equivalent
// launch an async tf job only if not all workers are occupied
template <class F, class... Args>
auto async(F&& func, Args&&... args) {
    auto task = std::make_shared<std::packaged_task<std::invoke_result_t<F, Args...>()>>(
        std::bind(std::forward<F>(func), std::forward<Args>(args)...));
    auto res = task->get_future();

    GetExecutor().silent_async([task] { (*task)(); });
    return res;
}
}  // namespace tt::tt_metal::detail
