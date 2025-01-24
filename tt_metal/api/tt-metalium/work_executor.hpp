// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

#include <condition_variable>
#include <functional>
#include <thread>

#include "lock_free_queue.hpp"
#include "tracy/Tracy.hpp"

namespace tt {

enum class WorkExecutorMode {
    SYNCHRONOUS = 0,
    ASYNCHRONOUS = 1,
};

enum class WorkerState {
    RUNNING = 0,
    TERMINATE = 1,
    IDLE = 2,
};

// Binds a device thread to a CPU core, determined using round-robin.
void set_device_thread_affinity(std::thread& worker_thread, int cpu_core_for_worker);

// Sets the process priority.
void set_process_priority(int requested_priority);

class WorkExecutor {
    // In asynchronous mode, each device has a worker thread that processes all host <--> cluster commands for this
    // device. Commands are pushed to the worker queue and picked up + executed asyncrhonously. Higher level functions
    // that have access to the device handle can queue up tasks asynchronously. In synchronous/pass through mode, we
    // bypass the queue and tasks are executed immediately after being pushed.
public:
    WorkExecutor(int cpu_core, int device_id) : cpu_core_for_worker_(cpu_core), managed_device_id_(device_id) {}
    ~WorkExecutor();

    WorkExecutor(const WorkExecutor&) = delete;
    WorkExecutor& operator=(const WorkExecutor&) = delete;
    WorkExecutor(WorkExecutor&& other) = delete;
    WorkExecutor& operator=(WorkExecutor&& other) = delete;

    void initialize();
    void reset();
    bool use_passthrough() const;
    void synchronize();
    void set_worker_mode(WorkExecutorMode mode);
    bool empty() const;

    template <typename F>
    void push_work(F&& work, bool blocking = false) {
        ZoneScopedN("PushWork");
        if (use_passthrough()) {
            // Worker is pushing to itself (nested work) or worker thread is not running. Execute work in current
            // thread.
            work();
        } else {
            // Push to worker queue.
            worker_queue_.push(std::forward<F>(work));
            {
                std::lock_guard lock(this->cv_mutex_);
                cv_.notify_one();
            }
            if (blocking) {
                this->synchronize();
            }
        }
    }

    WorkExecutorMode get_worker_mode() const { return work_executor_mode_; }
    std::thread::id get_parent_thread_id() const { return worker_queue_.parent_thread_id; }
    std::thread::id get_worker_thread_id() const { return worker_queue_.worker_thread_id; }

private:
    LockFreeQueue<std::function<void()>> worker_queue_;
    std::thread worker_thread_;
    WorkerState worker_state_;
    WorkExecutorMode work_executor_mode_;
    int cpu_core_for_worker_;
    int managed_device_id_;
    std::condition_variable cv_;
    std::mutex cv_mutex_;

    void run_worker();
    void start_worker();
    void stop_worker();
};

}  // namespace tt
