// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#include <unistd.h>
#endif
// Include LSan interface only when present
// __has_feature(leak_sanitizer) / __has_feature(address_sanitizer) for clang
// compile-time predicates; __SANITIZE_ADDRESS__ / __SANITIZE_LEAK__ for gcc.
// __has_feature is a Clang built-in; GCC does not define it.  Provide a
// fallback so the #if below is well-formed on both compilers.
#ifndef __has_feature
#  define __has_feature(x) 0
#endif
#if __has_feature(leak_sanitizer) || __has_feature(address_sanitizer) || \
    defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_LEAK__)
#define TT_LSAN_ACTIVE 1
#include <sanitizer/lsan_interface.h>
#endif
#include <taskflow/taskflow.hpp>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace tt::tt_metal::detail {
inline static const size_t EXECUTOR_NTHREADS = std::getenv("TT_METAL_THREADCOUNT")
                                                   ? std::stoi(std::getenv("TT_METAL_THREADCOUNT"))
                                                   : std::thread::hardware_concurrency();

using Executor = tf::Executor;
using ExecTask = tf::Task;

inline Executor& GetExecutor() {
#if defined(__linux__) || defined(__APPLE__)
    // After fork(), the child process must reinitialize the Executor because
    // the parent's worker threads do not survive across fork boundaries.
    // Without this, the child hangs trying to dispatch work to dead threads.
    //
    // Only the *child* handler recreates the Executor.  The parent's Executor
    // is left untouched -- destroying and recreating it in prepare/parent
    // would needlessly tear down the thread pool on every fork() call.
    // This matters because fork() is triggered in ways callers do not control:
    //   - Python subprocess.Popen() may fall back from posix_spawn to fork()
    //   - Python multiprocessing.Process() always uses fork() by default
    //   - MPI launchers may fork internally
    // In all these cases the parent's thread pool should remain undisturbed.
    //
    // In the child we intentionally leak the old Executor rather than delete
    // it.  This is the canonical approach for fork-unsafe objects: the threads
    // and synchronization primitives inherited from the parent are in an
    // indeterminate state post-fork, and calling the destructor would deadlock
    // or crash.  The same pattern is used by jemalloc and glibc malloc.
    //
    // The leak is bounded: it exists only in the child process for the
    // child's lifetime.  The kernel reclaims all memory when the child
    // exits.  The parent heap is unaffected -- the child's assignment to
    // exec happens in the child's private copy-on-write address space.
    // The Executor holds no OS resources (no FDs, no shared memory, no
    // GPU handles), so no handles are stranded.
    static Executor* exec = [] {
        auto* e = new Executor(EXECUTOR_NTHREADS);
        std::atexit([] {
            delete exec;
            exec = nullptr;
        });
        pthread_atfork(
            /*prepare=*/
            [] {
                // fork() with in-flight Taskflow work is unsafe: the child will
                // inherit a half-initialized executor state and may hang when it
                // tries to dispatch tasks to the dead parent threads.  We warn
                // here rather than aborting because fork() is sometimes called
                // from paths the application cannot control (e.g. Python's
                // subprocess fallback from posix_spawn).
                const auto num_topologies = exec ? exec->num_topologies() : 0;
                if (num_topologies != 0) {
                    // Use write because it is async-signal-safe,
                    char buf[128];
                    int n = std::snprintf(
                        buf,
                        sizeof(buf),
                        "WARNING: fork() called while executor has in-flight work "
                        "(num_topologies=%zu). This may cause hangs in the child process.\n",
                        num_topologies);
                    if (n > 0) {
                        [[maybe_unused]] auto _ = ::write(STDERR_FILENO, buf, static_cast<size_t>(n));
                    }
                }
            },
            /*parent=*/nullptr,
            /*child=*/
            [] {
                // Suppress the leak report in ASan/LSan builds so it is not
                // flagged as unintentional.
#ifdef TT_LSAN_ACTIVE
                __lsan_ignore_object(exec);
#undef TT_LSAN_ACTIVE
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
