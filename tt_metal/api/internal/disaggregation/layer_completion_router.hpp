// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LayerCompletionRouter — one per host in a pipelined-prefill MPI job. Owns the
// host-local completion ring (the prefill runner connects and pushes) and runs a
// listener thread. Subordinates MPI-send every message to the master; the master
// also receives from every subordinate and emits to the scheduler-facing segment it
// owns: v1 reorders by seq and injects a count into an InterProcessCounterChannel,
// v2 forwards self-describing messages as they arrive into a LayerCompletionQueueV2
// (see layer_completion_message.hpp). world_size == 1: no MPI.
//
// Teardown: at stop() each subordinate drains its ring and sends one end-of-stream
// sentinel; the master keeps receiving until every sentinel has arrived and its own
// ring is empty, bounded by teardown_timeout_ms for a rank that died first.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <variant>

namespace tt::tt_metal::distributed {
class InterProcessCounterChannel;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::internal {

using tt::tt_metal::distributed::InterProcessCounterChannel;

struct LayerCompletionMessage;
struct LayerCompletionMessageV2;
template <typename MsgT>
class LayerCompletionQueueT;
using LayerCompletionQueue = LayerCompletionQueueT<LayerCompletionMessage>;
using LayerCompletionQueueV2 = LayerCompletionQueueT<LayerCompletionMessageV2>;

enum class LayerCompletionProtocol : uint8_t {
    kCountOnlyV1 = 1,  // the PREFILL_LAYER_COMPLETION_PROTOCOL / binding values
    kStructuredV2 = 2,
};

struct LayerCompletionRouterConfig {
    int rank = 0;
    int world_size = 1;
    int master_rank = 0;
    std::string ring_shm_name;
    LayerCompletionProtocol protocol = LayerCompletionProtocol::kCountOnlyV1;
    // Master-only: the scheduler-facing segment this router creates (v1: InterProcessCounterChannel,
    // v2: LayerCompletionQueueV2). Give each protocol its own name: the counter channel does not
    // validate what it attaches to.
    std::string scheduler_shm_name;
    int poll_idle_us = 100;
    // Master-only: how long after stop() to keep waiting for subordinate sentinels (and, on v2, for
    // the scheduler to drain its ring) before giving up.
    int teardown_timeout_ms = 5000;
};

class LayerCompletionRouter {
public:
    explicit LayerCompletionRouter(LayerCompletionRouterConfig cfg);
    ~LayerCompletionRouter();

    LayerCompletionRouter(const LayerCompletionRouter&) = delete;
    LayerCompletionRouter& operator=(const LayerCompletionRouter&) = delete;

    // Idempotent: signal and join the listener thread, release the segments, then rethrow a
    // listener failure once.
    void stop();

    uint64_t processed() const noexcept { return processed_.load(std::memory_order_relaxed); }
    bool is_master() const noexcept { return cfg_.rank == cfg_.master_rank; }

private:
    struct CountOnly {
        std::unique_ptr<LayerCompletionQueue> ring;
        std::unique_ptr<InterProcessCounterChannel> scheduler;  // master only
    };
    struct Structured {
        std::unique_ptr<LayerCompletionQueueV2> ring;
        std::unique_ptr<LayerCompletionQueueV2> scheduler;  // master only
    };

    void listen();
    void run_master(CountOnly& state);
    void run_master(Structured& state);
    // Master fan-in: drain the local ring and the subordinates, hand every message to `forward`,
    // then the sentinel-coordinated teardown.
    template <typename MsgT, typename Forward>
    void fan_in(LayerCompletionQueueT<MsgT>& ring, Forward&& forward);
    template <typename MsgT>
    void run_subordinate(LayerCompletionQueueT<MsgT>& ring);
    void join();

    LayerCompletionRouterConfig cfg_;
    std::variant<CountOnly, Structured> state_;
    std::thread listener_;
    std::atomic<bool> stop_{false};
    std::chrono::steady_clock::time_point stopped_at_{};
    std::atomic<bool> stopped_{false};
    std::atomic<uint64_t> processed_{0};
    std::mutex error_mutex_;
    std::exception_ptr error_;
};

}  // namespace tt::tt_metal::internal
