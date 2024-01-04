// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "third_party/taskflow/taskflow/taskflow.hpp"
#include <thread>
#include <stdexcept>
#include "third_party/taskflow/taskflow/utility/traits.hpp"
namespace tt::tt_metal::detail {

    static const size_t EXECUTOR_NTHREADS = std::getenv("TT_METAL_THREADCOUNT") ? std::stoi( std::getenv("TT_METAL_THREADCOUNT") ) : std::thread::hardware_concurrency();

    using Executor = tf::Executor;
    using ExecTask = tf::Task;
    static Executor& GetExecutor() {
        static Executor exec(EXECUTOR_NTHREADS);
        return exec;
    }

    // When a taskflow worker thread throws an exception, Taskflow does not propagate this to main thread (https://github.com/taskflow/taskflow/issues/479)
    // This wrapper ensures that exceptions are re-thrown by packaged_task whenever .get() is called on future object
    // https://stackoverflow.com/questions/16344663/is-there-a-packaged-taskset-exception-equivalent
    template<class F>
    auto async(F&& func)
    {

        using return_type = typename std::invoke_result<F>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(func);

        std::future<return_type> res = task->get_future();
        GetExecutor().silent_async( [task] {  (*task)(); });

        return res;
    }

    template<class F,  typename... Tasks,
    std::enable_if_t<tf::all_same_v<tf::AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
    >
    tf::AsyncTask silent_dependent_async(F && func, Tasks&&... tasks )
    {
        auto task = std::make_shared<std::packaged_task<void>>(func);
        return GetExecutor().silent_dependent_async( [task] {  (*task)(); }, tasks... );
    }

    template<class F,  typename I,
    std::enable_if_t<!std::is_same_v<std::decay_t<I>, tf::AsyncTask>, void>* = nullptr
    >
    tf::AsyncTask silent_dependent_async(F && func, I first, I last )
    {
        auto task = std::make_shared<std::packaged_task<void>>(func);
        return GetExecutor().silent_dependent_async( [task] {  (*task)(); }, first, last );
    }
}
