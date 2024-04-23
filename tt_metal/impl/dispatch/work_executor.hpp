// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <functional>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <unistd.h>

#include "common/env_lib.hpp"
#include "lock_free_queue.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

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

class WorkExecutor {
    // In asynchronous mode, each device has a worker thread that processes all host <--> cluster commands for this device.
    // Commands are pushed to the worker queue and picked up + executed asyncrhonously.
    // Higher level functions that have access to the device handle can queue up tasks asynchronously.
    // In synchronous/pass through mode, we bypass the queue and tasks are executed immediately after being pushed.
    public:
    LockFreeQueue<std::function<void()>> worker_queue;

    WorkExecutor(int device_id) : managed_device_id(device_id) {
        if (this->worker_queue_mode == WorkExecutorMode::ASYNCHRONOUS) {
            this->start_worker();
        }
    }

    WorkExecutor(WorkExecutor &&other) {
        worker_state = other.worker_state;
        managed_device_id = other.managed_device_id;
    }

    ~WorkExecutor() {
        if (this->worker_queue_mode == WorkExecutorMode::ASYNCHRONOUS) {
            stop_worker();
        }
    }

    inline void run_worker() {
        while (true) {
            {
                // Worker stalls until queue is non-empty or terminate signal is set
                std::unique_lock<std::mutex> lock(this->cv_mutex);
                this->cv.wait(lock, [this] {return (not this->worker_queue.empty()) or this->worker_state == WorkerState::TERMINATE;});
            }
            if (this->worker_state == WorkerState::TERMINATE) {
                // Terminate signal set, and queue is empty - worker exits
                if(this->worker_queue.empty()) break;
            }
            ZoneScopedN("PopWork");
            // Queue non-empty: run command
            auto func = this->worker_queue.pop();
            (*func)();
        }
    }

    inline void push_work(const std::function<void()>& work_executor, bool blocking = false) {
        ZoneScopedN("PushWork");
        if (this->worker_queue_mode == WorkExecutorMode::ASYNCHRONOUS) {
            if (std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_queue.parent_thread_id.load()) {
                // Push function executor to worker queue
                this->worker_queue.push(work_executor);
                {
                    std::lock_guard lock(this->cv_mutex);
                    cv.notify_one();
                }
                if (blocking) {
                    this->synchronize();
                }
            } else {
                TT_ASSERT(std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_queue.worker_thread_id.load(), "Only main thread or worker thread can push to device worker queue.");
                work_executor();
            }
        } else {
            // Synchronous execution: Run function right away.
            work_executor();
        }
    }

    inline void synchronize() {
        if (this->worker_queue_mode == WorkExecutorMode::ASYNCHRONOUS and std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_queue.parent_thread_id.load()) {
            // Blocking = wait for queue flushed. Only main thread can explcitly insert a synchronize, otherwise we have a deadlock.
            this->worker_queue.push([](){}); // Send flush command (i.e. empty function)
            {
                std::lock_guard lock(this->cv_mutex);
                cv.notify_one();
            }
            // Wait for queue empty, i.e. flush command picked up
            while(not this->worker_queue.empty()) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            };
        }
    }

    inline void set_worker_mode(const WorkExecutorMode& mode) {
         if (this->worker_queue_mode == mode) {
            return;
        }
        this->worker_queue_mode = mode;
        if (this->worker_queue_mode == WorkExecutorMode::ASYNCHRONOUS) {
            this->start_worker();
        } else if (this->worker_queue_mode == WorkExecutorMode::SYNCHRONOUS) {
            this->synchronize();
            this->stop_worker();
        }
    }

    WorkExecutorMode get_worker_mode() { return worker_queue_mode; }

    inline std::size_t get_parent_thread_id() { return this->worker_queue.parent_thread_id; }
    private:
    std::thread worker_thread;
    WorkerState worker_state = WorkerState::IDLE;
    int managed_device_id = 0;
    std::condition_variable cv;
    std::mutex cv_mutex;

    inline void start_worker() {
        this->worker_queue.parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        this->worker_state = WorkerState::RUNNING;
        this->worker_thread = std::thread(&WorkExecutor::run_worker, this);
        this->worker_queue.worker_thread_id = std::hash<std::thread::id>{}(this->worker_thread.get_id());
        // Bind a worker tied to a device to a specific CPU core in round robin fashion. Thread affinity == Better Perf.
        static int num_online_cores = sysconf(_SC_NPROCESSORS_ONLN);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(managed_device_id % num_online_cores, &cpuset);
        int rc = pthread_setaffinity_np(worker_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
        if (rc) {
            log_warning(tt::LogMetal, "Unable to bind worker thread to CPU Core. May see performance degradation. Error Code: {}", rc);
        }
    }

    inline void stop_worker() {
        if (this->worker_state == WorkerState::IDLE) {
            return;
        }
        this->worker_state = WorkerState::TERMINATE;
        {
            std::lock_guard lock(this->cv_mutex);
            cv.notify_one();
        }
        this->worker_thread.join();
        this->worker_state = WorkerState::IDLE;
    }

    static WorkExecutorMode default_worker_queue_mode() {
        static int value = parse_env<int>("TT_METAL_ASYNC_DEVICE_QUEUE", static_cast<int>(WorkExecutorMode::SYNCHRONOUS));
        return static_cast<WorkExecutorMode>(value);
    }

    WorkExecutorMode worker_queue_mode = default_worker_queue_mode();
};

} // namespace tt
