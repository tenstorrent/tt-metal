// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LayerCompletionRouter — one per host in a pipelined-prefill MPI job.
// Owns the host-local LayerCompletionQueue (the prefill runner connects
// and pushes into it) and runs a background listener thread:
//
//   * Subordinate host (rank != master_rank): drain the ring, MPI-send
//     each completion to the master rank.
//   * Master host (rank == master_rank): drain the local ring AND poll
//     MPI for completions from every subordinate, feed all of them
//     through a LayerCompletionReorderBuffer, and inject(1) into the
//     scheduler-facing InterProcessCounterChannel (which this router
//     owns) for each completion that becomes contiguous-in-order.
//
// world_size == 1 ⇒ master path uses no MPI (local ring only).
//
// Coordinated teardown: at stop(), each subordinate drains its ring and then
// sends one end-of-stream SENTINEL (see layer_completion_message.hpp). The
// master does NOT cancel mid-stream — it keeps receiving until it has seen a
// sentinel from every subordinate (and its own ring is drained), so no
// blocking subordinate send is ever left without a receiver and no
// already-arrived completion is dropped by a cancel. A teardown_timeout_ms
// safety net bounds the wait if a rank crashed without sending its sentinel.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

namespace tt::tt_metal::distributed {
class InterProcessCounterChannel;  // fwd — defined in api/internal/service/inter_process_counter_channel.hpp
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::internal {

using tt::tt_metal::distributed::InterProcessCounterChannel;  // api/internal/service/

class LayerCompletionQueue;  // fwd — defined in layer_completion_queue.hpp

struct LayerCompletionRouterConfig {
    int rank = 0;
    int world_size = 1;
    int master_rank = 0;
    std::string ring_shm_name;
    std::string scheduler_channel_shm_name;  // master-only
    int poll_idle_us = 100;
    // Master-only safety net: max time to wait at teardown for outstanding subordinate sentinels
    // before giving up and cancelling (so a crashed/stalled rank can't hang the listener join
    // forever). The clean path returns as soon as all sentinels arrive, well under this.
    int teardown_timeout_ms = 5000;
};

class LayerCompletionRouter {
public:
    explicit LayerCompletionRouter(LayerCompletionRouterConfig cfg);
    ~LayerCompletionRouter();

    LayerCompletionRouter(const LayerCompletionRouter&) = delete;
    LayerCompletionRouter& operator=(const LayerCompletionRouter&) = delete;

    void stop();  // idempotent; signals + joins the listener thread
    uint64_t processed() const noexcept { return processed_.load(std::memory_order_relaxed); }
    bool is_master() const noexcept { return cfg_.rank == cfg_.master_rank; }

private:
    void run_master();
    void run_subordinate();

    LayerCompletionRouterConfig cfg_;
    std::unique_ptr<LayerCompletionQueue> queue_;          // owner of the host-local ring
    std::unique_ptr<InterProcessCounterChannel> counter_;  // master-only
    std::thread listener_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> stopped_{false};
    std::atomic<uint64_t> processed_{0};
};

}  // namespace tt::tt_metal::internal
