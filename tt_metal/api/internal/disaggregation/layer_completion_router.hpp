// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LayerCompletionRouter — one per host in a pipelined-prefill MPI job.
// Owns the host-local completion ring (the prefill runner connects and
// pushes into it) and runs a background listener thread. Two protocol
// versions, selected once per job via the config (never mixed in a job):
//
//   V1 (kCountOnlyV1, frozen): the master feeds every completion through a
//   LayerCompletionReorderBuffer and inject(1)s into the scheduler-facing
//   InterProcessCounterChannel (which this router owns) for each completion
//   that becomes contiguous-in-order. The scheduler receives a bare count
//   and correlates it with its own in-order chunk FIFO — per-request
//   in-order consumption (head-of-line blocking, issue #54632).
//
//   V2 (kStructuredV2): completions are self-describing
//   (LayerCompletionMessageV2: request/slot/position range/layer range), so
//   the master FORWARDS AS ARRIVED — no reorder buffer, no head-of-line
//   blocking — into a scheduler-facing v2 ring (which this router owns and
//   the scheduler connects to). Subordinates MPI-forward verbatim, as in v1.
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

struct LayerCompletionMessage;    // fwd — defined in layer_completion_message.hpp
struct LayerCompletionMessageV2;  // fwd — defined in layer_completion_message.hpp
template <typename MsgT>
class LayerCompletionQueueT;     // fwd — defined in layer_completion_queue.hpp
class LayerCompletionQueueBase;  // fwd — defined in layer_completion_queue.hpp
using LayerCompletionQueue = LayerCompletionQueueT<LayerCompletionMessage>;
using LayerCompletionQueueV2 = LayerCompletionQueueT<LayerCompletionMessageV2>;

// SchedulerEgress — the master rank's scheduler-facing output. The router owns
// exactly one; cfg.protocol fixes the concrete type at construction (see the
// .cpp): v1 wraps an InterProcessCounterChannel (inject the reordered count),
// v2 wraps a LayerCompletionQueueV2 (forward self-describing messages
// as-arrived). Only the protocol-neutral teardown is on the interface.
class SchedulerEgress {
public:
    virtual ~SchedulerEgress() = default;
    virtual void shutdown() = 0;
};

enum class LayerCompletionProtocol : uint8_t {
    // Values match the user-facing PREFILL_LAYER_COMPLETION_PROTOCOL / binding arg.
    kCountOnlyV1 = 1,  // reorder → bare count into InterProcessCounterChannel (frozen default)
    kStructuredV2 = 2,  // forward-as-arrived self-describing messages into a scheduler-facing ring
};

struct LayerCompletionRouterConfig {
    int rank = 0;
    int world_size = 1;
    int master_rank = 0;
    std::string ring_shm_name;
    LayerCompletionProtocol protocol = LayerCompletionProtocol::kCountOnlyV1;
    // Master-only: name of the scheduler-facing shm segment. ONE name for both protocols — the
    // protocol decides what the master creates there (v1: InterProcessCounterChannel to inject
    // into; v2: structured completion ring to forward into), i.e. same name, different layout/size.
    std::string scheduler_shm_name;
    int poll_idle_us = 100;
    // Master-only safety net: max time to wait at teardown for outstanding subordinate sentinels
    // before giving up and cancelling (so a crashed/stalled rank can't hang the listener join
    // forever). The clean path returns as soon as all sentinels arrive, well under this. On the
    // v2 master it also bounds a blocked scheduler-ring push once stop_ is set.
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
    void run_master();       // master fan-in loop; the per-protocol output policy is set up inside
    void run_subordinate();  // dispatches on cfg_.protocol

    // Shared master skeleton: local-ring drain + subordinate MPI fan-in + sentinel-coordinated
    // teardown. `forward` is the per-protocol output action (v1: reorder → count; v2: forward).
    template <typename MsgT, typename Forward>
    void run_master_impl(Forward&& forward);

    // Identical for both protocol versions except the message type.
    template <typename MsgT>
    void run_subordinate_impl(LayerCompletionQueueT<MsgT>& queue);

    LayerCompletionRouterConfig cfg_;
    // Both polymorphic over the protocol (fixed at construction, so the master/subordinate loops
    // recover the concrete type with a static_cast — see the .cpp):
    std::unique_ptr<LayerCompletionQueueBase> queue_;  // owner of the host-local ring
    std::unique_ptr<SchedulerEgress> sched_egress_;    // master-only scheduler-facing output
    std::thread listener_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> stopped_{false};
    std::atomic<uint64_t> processed_{0};
};

}  // namespace tt::tt_metal::internal
