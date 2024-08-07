// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "tt_metal/impl/dispatch/work_executor_v2.hpp"
#include "tt_metal/common/logger.hpp"

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

#include <condition_variable>
#include <thread>


namespace tt {
namespace tt_metal {

void set_device_thread_affinity(std::thread& thread, std::size_t cpu_core) {
    // Bind a device worker/reader thread to a CPU core, determined using round-robin.
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    int rc = pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc) {
        log_warning(
            tt::LogMetal,
            "Unable to bind worker thread to CPU Core. May see performance degradation. Error Code: {}",
            rc);
    }
}

WorkExecutorV2::WorkExecutorV2(std::size_t cpu_core) {
    worker_thread = std::thread([this] {
        while (this->run) {
            auto computation = this->worker_queue.pop_front();
            computation();
        }
    });
    // set_device_thread_affinity(worker_thread, cpu_core);
}

WorkExecutorV2::WorkExecutorV2(WorkExecutorV2&& other) :
    worker_queue(std::move(other.worker_queue)), worker_thread(std::move(other.worker_thread)), run(other.run) {
    }

WorkExecutorV2& WorkExecutorV2::operator=(WorkExecutorV2&& other) {
    this->worker_queue = std::move(other.worker_queue);
    this->worker_thread = std::move(other.worker_thread);
    this->run = other.run;
    return *this;
}

void WorkExecutorV2::push_work(std::function<void()>&& computation) {
    this->worker_queue.emplace_back(std::move(computation));
}

void WorkExecutorV2::synchronize() {
    while (not this->worker_queue.empty()) {}
}

WorkExecutorV2::~WorkExecutorV2() {
    this->run = false;
    this->worker_queue.emplace_back([] {}); // push sentinel computation
    this->worker_thread.join();
}

}  // namespace tt_metal

}  // namespace tt
