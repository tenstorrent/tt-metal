// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <thread>

#include "common/env_lib.hpp"
#include "lock_free_queue.hpp"

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

    WorkExecutor() {
        if (this->worker_queue_mode == WorkExecutorMode::ASYNCHRONOUS) {
            this->start_worker();
        }
    }

    WorkExecutor(WorkExecutor &&other) = default;
    WorkExecutor& operator=(WorkExecutor &&other) = default;

    ~WorkExecutor() {
        if (this->worker_queue_mode == WorkExecutorMode::ASYNCHRONOUS) {
            stop_worker();
        }
    }

    inline void run_worker() {
        this->worker_queue.worker_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        while (true) {
            if(this->worker_queue.empty()) {
                if (this->worker_state == WorkerState::TERMINATE) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            else {
                auto func = this->worker_queue.pop();
                (*func)();
            }
        }
    }

    inline void push_work(const std::function<void()>& work_executor, bool blocking = false) {
        if (this->worker_queue_mode == WorkExecutorMode::ASYNCHRONOUS) {
            if (std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_queue.parent_thread_id.load()) {
                // Push function executor to worker queue
                this->worker_queue.push(work_executor);
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
        if (this->worker_queue_mode == WorkExecutorMode::ASYNCHRONOUS) {
            // Blocking = wait for queue flushed
            this->worker_queue.push([](){}); // Send flush command (i.e. empty function)
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
            this->worker_queue.parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
            this->start_worker();
        } else if (this->worker_queue_mode == WorkExecutorMode::SYNCHRONOUS) {
            this->synchronize();
            this->stop_worker();
        }
    }

    static WorkExecutorMode get_worker_mode() { return worker_queue_mode; }

    inline std::size_t get_parent_thread_id() { return this->worker_queue.parent_thread_id; }
    private:
    std::thread worker_thread;
    WorkerState worker_state = WorkerState::IDLE;

    inline void start_worker() {
        this->worker_queue.parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        this->worker_state = WorkerState::RUNNING;
        this->worker_thread = std::thread(&WorkExecutor::run_worker, this);
    }

    inline void stop_worker() {
        if (this->worker_state == WorkerState::IDLE) {
            return;
        }
        this->worker_state = WorkerState::TERMINATE;
        this->worker_thread.join();
        this->worker_state = WorkerState::IDLE;
    }

    static WorkExecutorMode default_worker_queue_mode() {
        static int value = parse_env<int>("TT_METAL_ASYNC_DEVICE_QUEUE", static_cast<int>(WorkExecutorMode::SYNCHRONOUS));
        return static_cast<WorkExecutorMode>(value);
    }

    inline static WorkExecutorMode worker_queue_mode = default_worker_queue_mode();
};

} // namespace tt
