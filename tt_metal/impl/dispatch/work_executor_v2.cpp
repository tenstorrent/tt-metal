// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "tt_metal/impl/dispatch/work_executor_v2.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/assert.hpp"

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

#include <thread>


namespace tt {
namespace tt_metal {

void set_device_thread_affinity(std::thread& thread, std::size_t cpu_core) {
    // Bind a device worker/reader thread to a CPU core, determined using round-robin.
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    auto return_code = pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (return_code) {
        log_warning(
            tt::LogMetal,
            "Unable to bind worker thread to CPU Core. May see performance degradation. Error Code: {}",
            return_code);
    }
}

WorkExecutorV2::WorkExecutorV2(std::size_t cpu_core, std::size_t device_id) : cpu_core{cpu_core}, device_id{device_id} {}

WorkExecutorV2::WorkExecutorV2(WorkExecutorV2&& other) :
    cpu_core(other.cpu_core),
    device_id(other.device_id),
    running(other.running),
    worker_queue(std::move(other.worker_queue)),
    worker_thread(std::move(other.worker_thread)) {
    other.running = {};
    other.worker_queue = {};
    other.worker_thread = {};
}

WorkExecutorV2& WorkExecutorV2::operator=(WorkExecutorV2&& other) {
    if (this == &other) {
        return *this;
    }
    this->cpu_core = other.cpu_core;
    this->device_id = other.device_id;
    this->running = other.running;
    this->worker_queue = std::move(other.worker_queue);
    this->worker_thread = std::move(other.worker_thread);
    other.running = {};
    other.worker_queue = {};
    other.worker_thread = {};
    return *this;
}

WorkExecutorV2::~WorkExecutorV2() {
    tt::log_debug(tt::LogAsync, "Device {}: Destroying Work Executor", this->device_id);
    this->stop();
    tt::log_debug(tt::LogAsync, "Device {}: Destroyed Work Executor", this->device_id);
}

void WorkExecutorV2::push_work(std::function<void()>&& computation, bool blocking) {
    if (not this->running) {
        tt::log_debug(tt::LogAsync, "Device {}: Running computation on main thread", this->device_id);
        computation();
        tt::log_debug(tt::LogAsync, "Device {}: Ran computation on main thread", this->device_id);
        return;
    }
    tt::log_debug(tt::LogAsync, "Device {}: Pushing computation", this->device_id);
    this->worker_queue.emplace_back(std::move(computation));
    tt::log_debug(tt::LogAsync, "Device {}: Pushed computation", this->device_id);

    blocking = true; // TODO: Remove this line
    if (blocking) {
        tt::log_debug(tt::LogAsync, "Device {}: Blocking until computation is done", this->device_id);
        while (not this->worker_queue.empty()) {}
        tt::log_debug(tt::LogAsync, "Device {}: Finished Blocking", this->device_id);
    }
}

void WorkExecutorV2::synchronize() {
    if (not this->running) {
        return;
    }
    tt::log_debug(tt::LogAsync, "Device {}: Synchronizing Work Executor", this->device_id);
    this->push_work(
        [this] {
            tt::log_debug(tt::LogAsync, "Device {}: Flush Work Executor", this->device_id);
        },
        true
    );
    tt::log_debug(tt::LogAsync, "Device {}: Synchronized Work Executor", this->device_id);
}

void WorkExecutorV2::start() {
    tt::log_debug(tt::LogAsync, "MAIN THREAD: Device {}: Starting Work Executor", this->device_id);
    TT_ASSERT(not this->running);
    TT_ASSERT(this->worker_queue.empty());
    this->running = true;
    this->worker_thread = std::thread([this] {
        while (this->running) {

            auto computation = this->worker_queue.pop_front();
            tt::log_debug(tt::LogAsync, "Device {}: Popped computation", this->device_id);
            computation();
            tt::log_debug(tt::LogAsync, "Device {}: Ran computation", this->device_id);
        }
    });
    set_device_thread_affinity(this->worker_thread, this->cpu_core);
    tt::log_debug(tt::LogAsync, "MAIN THREAD: Device {}: Started Work Executor", this->device_id);
}

void WorkExecutorV2::stop() {
    tt::log_debug(tt::LogAsync, "MAIN THREAD: Device {}: Stopping Work Executor", this->device_id);
    if (this->running) {
        this->synchronize();
        this->running = false;
        this->worker_queue.emplace_back([this] {
            tt::log_debug(tt::LogAsync, "Device {}: Sentinel", this->device_id);
        });
        this->worker_thread.join();
    }
    tt::log_debug(tt::LogAsync, "MAIN THREAD: Device {}: Stopped Work Executor", this->device_id);

}

}  // namespace tt_metal

}  // namespace tt
