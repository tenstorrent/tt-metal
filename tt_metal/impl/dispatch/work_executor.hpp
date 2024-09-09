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

#include "common/env_lib.hpp"
#include "lock_free_queue.hpp"
#include "tt_metal/third_party/tracy/public/common/TracySystem.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/third_party/umd/device/tt_cluster_descriptor_types.h"

#if defined(TRACY_ENABLE)
#define TracyTTThreadName(name, id)                     \
    std::string tmp = fmt::format("{} : {}", name, id); \
    tracy::SetThreadName(tmp.c_str());
#else
#define TracyTTThreadName(name, id)
#endif

namespace tt {

enum class WorkExecutorMode {
    SYNCHRONOUS = 0,
    ASYNCHRONOUS = 1,
};

enum class WorkerQueueMode {
    LOCKFREE = 0,
    LOCKBASED = 1,
};

enum class WorkerState {
    RUNNING = 0,
    TERMINATE = 1,
    IDLE = 2,
};

inline void set_device_thread_affinity(std::thread& thread_, int cpu_core_for_worker) {
    // Bind a device worker/reader thread to a CPU core, determined using round-robin.
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core_for_worker, &cpuset);
    int rc = pthread_setaffinity_np(thread_.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc) {
        log_warning(
            tt::LogMetal,
            "Unable to bind worker thread to CPU Core. May see performance degradation. Error Code: {}",
            rc);
    }
}

inline void set_process_priority(int requested_priority) {
    // Get priority for calling process
    int process_priority = getpriority(PRIO_PROCESS, 0);
    log_debug(tt::LogMetal, "Initial Process Priority: {}", process_priority);
    if (process_priority == requested_priority)
        return;
    // Set priority for calling process to user specified value
    int rc = setpriority(PRIO_PROCESS, 0, requested_priority);
    if (rc) {
        log_warning(tt::LogMetal, "Unable to set process priority to {}, error code: {}", requested_priority, rc);
    }
}

class WorkExecutor {
    // In asynchronous mode, each device has a worker thread that processes all host <--> cluster commands for this
    // device. Commands are pushed to the worker queue and picked up + executed asyncrhonously. Higher level functions
    // that have access to the device handle can queue up tasks asynchronously. In synchronous/pass through mode, we
    // bypass the queue and tasks are executed immediately after being pushed.
   public:
    LockFreeQueue<std::function<void()>> worker_queue;

    WorkExecutor(int cpu_core, umd::chip_id device_id) : cpu_core_for_worker(cpu_core), managed_device_id(device_id) {}

    WorkExecutor(WorkExecutor&& other) {
        worker_state = std::move(other.worker_state);
        cpu_core_for_worker = std::move(other.cpu_core_for_worker);
        managed_device_id = std::move(other.managed_device_id);
    }

    WorkExecutor& operator=(WorkExecutor &&other) {
        if (this != &other) {
            worker_state = std::move(other.worker_state);
            managed_device_id = std::move(other.managed_device_id);
            cpu_core_for_worker = std::move(other.cpu_core_for_worker);
        }
        return *this;
    }

    ~WorkExecutor() { reset(); }

    inline void initialize() {
        this->work_executor_mode = default_worker_executor_mode();
        this->worker_queue_mode = default_worker_queue_mode();
        this->worker_state = WorkerState::IDLE;
        this->worker_queue.parent_thread_id = 0;
        this->worker_queue.worker_thread_id = 0;
        set_process_priority(0);
        if (this->work_executor_mode == WorkExecutorMode::ASYNCHRONOUS) {
            this->set_worker_queue_mode(this->worker_queue_mode);
            this->start_worker();
        }
    }

    inline void reset() {
        if (this->work_executor_mode == WorkExecutorMode::ASYNCHRONOUS) {
            stop_worker();
        }
        this->work_executor_mode = WorkExecutorMode::SYNCHRONOUS;
    }

    inline void run_worker() {
        TracyTTThreadName("TT_WORKER_DEVICE_ID", this->managed_device_id);
        while (true) {
            {
                // Worker stalls until queue is non-empty or terminate signal is set
                std::unique_lock<std::mutex> lock(this->cv_mutex);
                this->cv.wait(lock, [this] {
                    return (not this->worker_queue.empty()) or this->worker_state == WorkerState::TERMINATE;
                });
            }
            if (this->worker_state == WorkerState::TERMINATE) {
                // Terminate signal set, and queue is empty - worker exits
                if (this->worker_queue.empty())
                    break;
            }
            ZoneScopedN("PopWork");
            // Queue non-empty: run command
            auto func = this->worker_queue.pop();
            (*func)();
        }
    }

    inline void push_work(const std::function<void()>& work_executor, bool blocking = false) {
        ZoneScopedN("PushWork");
        if (std::hash<std::thread::id>{}(std::this_thread::get_id()) == this->worker_queue.worker_thread_id.load() or
            not(this->worker_state == WorkerState::RUNNING)) {
            // Worker is pushing to itself (nested work) or worker thread is not running. Execute work in current
            // thread.
            work_executor();
        } else {
            // Push to worker queue.
            this->worker_queue.push(work_executor);
            {
                std::lock_guard lock(this->cv_mutex);
                cv.notify_one();
            }
            if (blocking) {
                this->synchronize();
            }
        }
    }

    inline void push_work(std::shared_ptr<std::function<void()>> work_executor, bool blocking = false) {
        // Latest push API, passing ptrs around for work container. Usually faster, since no data-copies.
        ZoneScopedN("PushWork");
        if (std::hash<std::thread::id>{}(std::this_thread::get_id()) == this->worker_queue.worker_thread_id.load() or
            not(this->worker_state == WorkerState::RUNNING)) {
            // Worker is pushing to itself (nested work) or worker thread is not running. Execute work in current
            // thread.
            (*work_executor)();
        } else {
            // Push to worker queue.
            this->worker_queue.push(work_executor);
            {
                std::lock_guard lock(this->cv_mutex);
                cv.notify_one();
            }
            if (blocking) {
                this->synchronize();
            }
        }
    }

    inline void synchronize() {
        if (this->work_executor_mode == WorkExecutorMode::ASYNCHRONOUS and
            not(std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_queue.worker_thread_id.load())) {
            // Blocking = wait for queue flushed. Worker thread cannot explcitly insert a synchronize, otherwise we have
            // a deadlock.
            this->worker_queue.push([]() {});  // Send flush command (i.e. empty function)
            {
                std::lock_guard lock(this->cv_mutex);
                cv.notify_one();
            }
            // Wait for queue empty, i.e. flush command picked up
            while (not this->worker_queue.empty()) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            };
        }
    }

    inline void set_worker_mode(const WorkExecutorMode& mode) {
        if (this->work_executor_mode == mode) {
            return;
        }
        this->work_executor_mode = mode;
        if (this->work_executor_mode == WorkExecutorMode::ASYNCHRONOUS) {
            this->start_worker();
        } else if (this->work_executor_mode == WorkExecutorMode::SYNCHRONOUS) {
            this->synchronize();
            this->stop_worker();
        }
    }

    WorkExecutorMode get_worker_mode() { return work_executor_mode; }

    inline void set_worker_queue_mode(const WorkerQueueMode& mode) {
        if (mode == WorkerQueueMode::LOCKFREE) {
            this->worker_queue.set_lock_free();
        } else {
            this->worker_queue.set_lock_based();
        }
        this->worker_queue_mode = mode;
    }

    WorkerQueueMode get_worker_queue_mode() { return worker_queue_mode; }

    inline std::size_t get_parent_thread_id() { return this->worker_queue.parent_thread_id; }

   private:
    std::thread worker_thread;
    WorkerState worker_state;
    int cpu_core_for_worker;
    umd::chip_id managed_device_id;
    std::condition_variable cv;
    std::mutex cv_mutex;

    inline void start_worker() {
        this->worker_queue.parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        this->worker_state = WorkerState::RUNNING;
        this->worker_thread = std::thread(&WorkExecutor::run_worker, this);
        this->worker_queue.worker_thread_id = std::hash<std::thread::id>{}(this->worker_thread.get_id());
        // Bind a worker tied to a device to a specific CPU core in round robin fashion. Thread affinity == Better Perf.
        set_device_thread_affinity(this->worker_thread, this->cpu_core_for_worker);
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

    static WorkExecutorMode default_worker_executor_mode() {
        static int value =
            parse_env<int>("TT_METAL_ASYNC_DEVICE_QUEUE", static_cast<int>(WorkExecutorMode::SYNCHRONOUS));
        return static_cast<WorkExecutorMode>(value);
    }

    static WorkerQueueMode default_worker_queue_mode() {
        static int value = parse_env<int>("TT_METAL_LOCK_BASED_QUEUE", static_cast<int>(WorkerQueueMode::LOCKFREE));
        return static_cast<WorkerQueueMode>(value);
    }

    WorkExecutorMode work_executor_mode;
    WorkerQueueMode worker_queue_mode;
};

}  // namespace tt
