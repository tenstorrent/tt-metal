// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "third_party/taskflow/taskflow/taskflow.hpp"
#include <thread>
#include <stdexcept>

namespace tt::tt_metal::detail {
    static const size_t EXECUTOR_NTHREADS = std::getenv("TT_METAL_THREADCOUNT") ? std::stoi( std::getenv("TT_METAL_THREADCOUNT") ) : std::thread::hardware_concurrency();

    using Executor = tf::Executor;
    using ExecTask = tf::Task;

    static Executor& GetExecutor() {
        static Executor exec(EXECUTOR_NTHREADS);
        return exec;
    }

    static std::mutex& GetExecutorMutex() {
        static std::mutex exec_mutex;
        return exec_mutex;
    }

    // When a taskflow worker thread throws an exception, Taskflow does not propagate this to main thread (https://github.com/taskflow/taskflow/issues/479)
    // This wrapper ensures that exceptions are re-thrown by packaged_task whenever .get() is called on future object
    // https://stackoverflow.com/questions/16344663/is-there-a-packaged-taskset-exception-equivalent
    // launch an async tf job only if not all workers are occupied
    template<class F, class... Args>
    auto async(F&& func, Args&&... args)
    {

        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(func), std::forward<Args>(args)...));
        std::shared_future<return_type> res( std::move( task->get_future() ) );
        GetExecutorMutex().lock();
        
        if (GetExecutor().num_topologies() >= GetExecutor().num_workers()) {
            GetExecutorMutex().unlock();
            (*task)();
            return res;
        }

        GetExecutor().silent_async( [task] {  (*task)(); });
        GetExecutorMutex().unlock();
        return res;
    }
}
