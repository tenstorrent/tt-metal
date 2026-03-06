// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pthread.h>
#include <taskflow/taskflow.hpp>
#include <cstdlib>
#include <thread>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::detail {
inline static const size_t EXECUTOR_NTHREADS = std::getenv("TT_METAL_THREADCOUNT")
                                                   ? std::stoi(std::getenv("TT_METAL_THREADCOUNT"))
                                                   : std::thread::hardware_concurrency();

using Executor = tf::Executor;
using ExecTask = tf::Task;

inline Executor& GetExecutor() {
    // Child process needs to reinitialize the executor after fork()
    // otherwise it will hang because it will try to reference stale thread state
    // copied from the parent process.
    // Also ensure that no work is in-flight on the main process before forking.
    static Executor* exec = [] {
        auto* e = new Executor(EXECUTOR_NTHREADS);
        std::atexit([] {
            delete exec;
            exec = nullptr;
        });
        pthread_atfork(
            /*prepare=*/
            [] {
                TT_FATAL(
                    exec->num_topologies() == 0,
                    "fork() called while executor has in-flight work "
                    "(num_topologies={}). All tasks must complete before forking.",
                    exec->num_topologies());
                delete exec;
                exec = nullptr;
            },
            /*parent=*/[] { exec = new Executor(EXECUTOR_NTHREADS); },
            /*child=*/[] { exec = new Executor(EXECUTOR_NTHREADS); });
        return e;
    }();
    return *exec;
}

inline std::mutex& GetExecutorMutex() {
    static std::mutex exec_mutex;
    return exec_mutex;
}

// When a taskflow worker thread throws an exception, Taskflow does not propagate this to main thread
// (https://github.com/taskflow/taskflow/issues/479) This wrapper ensures that exceptions are re-thrown by packaged_task
// whenever .get() is called on future object
// https://stackoverflow.com/questions/16344663/is-there-a-packaged-taskset-exception-equivalent
// launch an async tf job only if not all workers are occupied
template <class F, class... Args>
auto async(F&& func, Args&&... args) {
    using return_type = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(func), std::forward<Args>(args)...));
    std::shared_future<return_type> res(std::move(task->get_future()));
    GetExecutorMutex().lock();

    if (GetExecutor().num_topologies() >= GetExecutor().num_workers()) {
        GetExecutorMutex().unlock();
        (*task)();
        return res;
    }

    GetExecutor().silent_async([task] { (*task)(); });
    GetExecutorMutex().unlock();
    return res;
}
}  // namespace tt::tt_metal::detail
