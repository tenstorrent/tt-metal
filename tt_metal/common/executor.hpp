// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __linux__
#include <pthread.h>
#endif
#if __has_include(<sanitizer/lsan_interface.h>)
#include <sanitizer/lsan_interface.h>
#endif
#include <taskflow/taskflow.hpp>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace tt::tt_metal::detail {
inline static const size_t EXECUTOR_NTHREADS = std::getenv("TT_METAL_THREADCOUNT")
                                                   ? std::stoi(std::getenv("TT_METAL_THREADCOUNT"))
                                                   : std::thread::hardware_concurrency();

using Executor = tf::Executor;
using ExecTask = tf::Task;

inline Executor& GetExecutor() {
#ifdef __linux__
    // After fork(), the child process must reinitialize the Executor because
    // the parent's worker threads do not survive across fork boundaries.
    // Without this, the child hangs trying to dispatch work to dead threads.
    //
    // Only the *child* handler recreates the Executor.  The parent's Executor
    // is left untouched -- destroying and recreating it in prepare/parent
    // would needlessly tear down the thread pool on every fork() call
    // (including those from posix_spawn fallback paths or Python subprocess).
    static Executor* exec = [] {
        auto* e = new Executor(EXECUTOR_NTHREADS);
        std::atexit([] {
            delete exec;
            exec = nullptr;
        });
        pthread_atfork(
            /*prepare=*/
            [] {
                if (exec && exec->num_topologies() != 0) {
                    fprintf(
                        stderr,
                        "WARNING: fork() called while executor has in-flight work "
                        "(num_topologies=%zu). This may cause hangs in the child process.\n",
                        exec->num_topologies());
                }
            },
            /*parent=*/nullptr,
            /*child=*/
            [] {
                // The parent's threads are gone in the child; replace with a
                // fresh Executor.  Intentionally leak the old one -- its
                // internal state (mutexes, threads) is invalid post-fork and
                // destroying it would deadlock or crash.
#ifdef __lsan_ignore_object
                __lsan_ignore_object(exec);
#endif
                exec = new Executor(EXECUTOR_NTHREADS);
            });
        return e;
    }();
    return *exec;
#else
    static Executor exec(EXECUTOR_NTHREADS);
    return exec;
#endif
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
