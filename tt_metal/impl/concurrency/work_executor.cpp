// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <work_executor.hpp>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

#include "env_lib.hpp"
#include <thread>

#include "tracy/Tracy.hpp"

#if defined(TRACY_ENABLE)
#define TracyTTThreadName(name, id)                     \
    std::string tmp = fmt::format("{} : {}", name, id); \
    tracy::SetThreadName(tmp.c_str());
#else
#define TracyTTThreadName(name, id)
#endif

namespace tt {
namespace {

WorkExecutorMode default_worker_executor_mode() {
    static int value = parse_env<int>("TT_METAL_ASYNC_DEVICE_QUEUE", static_cast<int>(WorkExecutorMode::SYNCHRONOUS));
    return static_cast<WorkExecutorMode>(value);
}

}  // namespace

void set_device_thread_affinity(std::thread& worker_thread, int cpu_core_for_worker) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core_for_worker, &cpuset);
    int rc = pthread_setaffinity_np(worker_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc) {
        log_warning(
            tt::LogMetal,
            "Unable to bind worker thread to CPU Core. May see performance degradation. Error Code: {}",
            rc);
    }
}

void set_process_priority(int requested_priority) {
    // Get priority for calling process
    int process_priority = getpriority(PRIO_PROCESS, 0);
    log_debug(tt::LogMetal, "Initial Process Priority: {}", process_priority);
    if (process_priority == requested_priority) {
        return;
    }
    // Set priority for calling process to user specified value
    int rc = setpriority(PRIO_PROCESS, 0, requested_priority);
    if (rc) {
        log_warning(tt::LogMetal, "Unable to set process priority to {}, error code: {}", requested_priority, rc);
    }
}

WorkExecutor::~WorkExecutor() { reset(); }

void WorkExecutor::initialize() {
    work_executor_mode_ = default_worker_executor_mode();
    worker_state_ = WorkerState::IDLE;
    set_process_priority(0);
    if (work_executor_mode_ == WorkExecutorMode::ASYNCHRONOUS) {
        start_worker();
    }
}

void WorkExecutor::reset() {
    if (work_executor_mode_ == WorkExecutorMode::ASYNCHRONOUS) {
        stop_worker();
    }
    work_executor_mode_ = WorkExecutorMode::SYNCHRONOUS;
}

bool WorkExecutor::use_passthrough() const {
    return std::this_thread::get_id() == worker_queue_.worker_thread_id.load() || worker_state_ != WorkerState::RUNNING;
}

void WorkExecutor::synchronize() {
    if (work_executor_mode_ == WorkExecutorMode::ASYNCHRONOUS and
        not(std::this_thread::get_id() == worker_queue_.worker_thread_id.load())) {
        // Blocking = wait for queue flushed. Worker thread cannot explcitly insert a synchronize, otherwise we have
        // a deadlock.
        worker_queue_.push([]() {});  // Send flush command (i.e. empty function)
        {
            std::lock_guard lock(cv_mutex_);
            cv_.notify_one();
        }
        // Wait for queue empty, i.e. flush command picked up
        while (not worker_queue_.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        };
    }
}

void WorkExecutor::set_worker_mode(WorkExecutorMode mode) {
    if (work_executor_mode_ == mode) {
        return;
    }
    work_executor_mode_ = mode;
    if (work_executor_mode_ == WorkExecutorMode::ASYNCHRONOUS) {
        start_worker();
    } else if (work_executor_mode_ == WorkExecutorMode::SYNCHRONOUS) {
        synchronize();
        stop_worker();
    }
}

void WorkExecutor::run_worker() {
    TracyTTThreadName("TT_WORKER_DEVICE_ID", managed_device_id_);
    while (true) {
        {
            // Worker stalls until queue is non-empty or terminate signal is set
            std::unique_lock<std::mutex> lock(cv_mutex_);
            cv_.wait(lock, [this] { return (not worker_queue_.empty()) or worker_state_ == WorkerState::TERMINATE; });
        }
        // Terminate signal set, and queue is empty - worker exits
        if (worker_state_ == WorkerState::TERMINATE and worker_queue_.empty()) {
            break;
        }
        ZoneScopedN("PopWork");
        // Queue non-empty: run command
        auto func = worker_queue_.pop();
        (*func)();
    }
}

void WorkExecutor::start_worker() {
    worker_queue_.parent_thread_id = std::this_thread::get_id();
    worker_state_ = WorkerState::RUNNING;
    worker_thread_ = std::thread(&WorkExecutor::run_worker, this);
    worker_queue_.worker_thread_id = worker_thread_.get_id();
    // Bind a worker tied to a device to a specific CPU core in round robin fashion. Thread affinity == Better Perf.
    set_device_thread_affinity(worker_thread_, cpu_core_for_worker_);
}

void WorkExecutor::stop_worker() {
    if (worker_state_ == WorkerState::IDLE) {
        return;
    }
    worker_state_ = WorkerState::TERMINATE;
    {
        std::lock_guard lock(cv_mutex_);
        cv_.notify_one();
    }
    worker_thread_.join();
    worker_state_ = WorkerState::IDLE;
}

bool WorkExecutor::empty() const { return worker_queue_.empty(); }

}  // namespace tt
