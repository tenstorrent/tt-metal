// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "pipeline_manager/pipeline_manager_types.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

// Abstract interface between the pipeline manager and the hardware pipeline.
// Concrete implementations: MockPipeline (testing), SocketPipeline (real hardware).
class PipelineInterface {
public:
    virtual ~PipelineInterface() = default;

    // Writer thread calls: inject one token into the pipeline. May block if pipeline is full.
    virtual void inject(const InjectDescriptor& desc) = 0;

    // Reader thread calls: blocking read of next result from pipeline output.
    virtual ResultDescriptor read_result() = 0;

    // Free KV cache for a user slot.
    virtual void reset_kv(int user_id) = 0;

    // Signal that threads should stop. read_result() must return a sentinel
    // ResultDescriptor{.user_id = -1} promptly after this is called.
    // Must NOT use the sockets to communicate with the device — threads may
    // still hold references. Only sets internal flags.
    virtual void request_stop() = 0;

    // Called AFTER all threads using inject()/read_result() have been joined.
    // Performs any final device-side cleanup (e.g. sending sentinel to kernel).
    // Safe to use sockets here — exclusive ownership is guaranteed.
    virtual void shutdown() = 0;
};

// Mock pipeline for isolated testing.
// Decode tokens produce deterministic output: actual_token = token_id + 1.
// Prefill tokens produce non-sampled results (KV populated, no output token).
//
// Optional latency: read_result() sleeps for a random duration in
// [latency_min_us, latency_max_us] before returning, simulating the reader
// waiting for data to traverse the pipeline.
class MockPipeline : public PipelineInterface {
public:
    MockPipeline(int latency_min_us = 0, int latency_max_us = 0, uint32_t seed = 42) :
        latency_min_us_(latency_min_us), latency_max_us_(latency_max_us), rng_(seed) {}

    void inject(const InjectDescriptor& desc) override {
        {
            std::lock_guard<std::mutex> lock(log_mtx_);
            inject_log_.push_back(desc);
        }

        ResultDescriptor result{};
        result.user_id = desc.user_id;
        result.position = desc.position;
        result.mode = desc.mode;
        result.spec_flag = desc.spec_flag;

        if (desc.mode == TokenMode::DECODE) {
            result.sampled = true;
            result.actual_token = desc.token_id + 1;
            result.predicted_token = EMPTY_TOKEN;
        } else {
            result.sampled = false;
            result.actual_token = EMPTY_TOKEN;
            result.predicted_token = EMPTY_TOKEN;
        }

        std::lock_guard<std::mutex> lock(mtx_);
        result_queue_.push(result);
        cv_.notify_one();
    }

    // Returns a snapshot of all injections so far (thread-safe).
    std::vector<InjectDescriptor> get_inject_log() const {
        std::lock_guard<std::mutex> lock(log_mtx_);
        return inject_log_;
    }

    ResultDescriptor read_result() override {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !result_queue_.empty() || shutdown_; });
        if (shutdown_ && result_queue_.empty()) {
            return ResultDescriptor{.user_id = -1, .sampled = false};
        }
        ResultDescriptor result = result_queue_.front();
        result_queue_.pop();
        lock.unlock();

        if (latency_max_us_ > 0) {
            std::lock_guard<std::mutex> rng_lock(rng_mtx_);
            int delay = std::uniform_int_distribution<int>(latency_min_us_, latency_max_us_)(rng_);
            std::this_thread::sleep_for(std::chrono::microseconds(delay));
        }

        return result;
    }

    void reset_kv(int /*user_id*/) override {}

    void request_stop() override {
        std::lock_guard<std::mutex> lock(mtx_);
        shutdown_ = true;
        cv_.notify_all();
    }

    void shutdown() override {}

private:
    int latency_min_us_;
    int latency_max_us_;
    std::mt19937 rng_;
    std::mutex rng_mtx_;
    std::queue<ResultDescriptor> result_queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool shutdown_ = false;
    std::vector<InjectDescriptor> inject_log_;
    mutable std::mutex log_mtx_;
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
