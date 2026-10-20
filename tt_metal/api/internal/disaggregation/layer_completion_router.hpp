// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LayerCompletionRouter: one per host in a pipelined-prefill MPI job. Owns the host-local ring the runner pushes
// into and runs a listener thread. Subordinates send every message to the master over MPI; the master forwards
// its own and theirs, as they arrive, into the scheduler-facing ring it owns. world_size == 1 needs no MPI.
// Teardown: each subordinate drains its ring and sends one sentinel; the master keeps receiving until every
// sentinel is in and its ring is empty, bounded by teardown_timeout_ms.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace tt::tt_metal::internal {

class LayerCompletionQueue;

struct LayerCompletionRouterConfig {
    int rank = 0;
    int world_size = 1;
    int master_rank = 0;
    std::string ring_shm_name;
    // Master only: the scheduler-facing ring.
    std::string scheduler_shm_name;
    // Published in the header of every ring this router creates.
    uint32_t completions_per_chunk = 0;
    int poll_idle_us = 100;
    // Master only: how long after stop() to wait for subordinate sentinels and for the scheduler to drain.
    int teardown_timeout_ms = 5000;
};

class LayerCompletionRouter {
public:
    explicit LayerCompletionRouter(LayerCompletionRouterConfig cfg);
    ~LayerCompletionRouter();

    LayerCompletionRouter(const LayerCompletionRouter&) = delete;
    LayerCompletionRouter& operator=(const LayerCompletionRouter&) = delete;

    // Idempotent: stop and join the listener, release the segments, rethrow a listener failure once.
    void stop();

    uint64_t processed() const noexcept { return processed_.load(std::memory_order_relaxed); }
    bool is_master() const noexcept { return cfg_.rank == cfg_.master_rank; }

private:
    void listen();
    void run_master();
    // Master fan-in: local ring and subordinates into `forward`, then the sentinel-coordinated teardown.
    template <typename Forward>
    void fan_in(const Forward& forward);
    void run_subordinate();
    void join();

    LayerCompletionRouterConfig cfg_;
    std::unique_ptr<LayerCompletionQueue> ring_;
    std::unique_ptr<LayerCompletionQueue> scheduler_;  // master only
    std::thread listener_;
    std::atomic<bool> stop_{false};
    std::chrono::steady_clock::time_point stopped_at_;
    std::atomic<bool> stopped_{false};
    std::atomic<uint64_t> processed_{0};
    std::mutex error_mutex_;
    std::exception_ptr error_;
};

}  // namespace tt::tt_metal::internal
