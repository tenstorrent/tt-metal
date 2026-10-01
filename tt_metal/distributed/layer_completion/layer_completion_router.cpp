// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/layer_completion_router.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <optional>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/distributed_context.hpp>

#include <internal/service/inter_process_counter_channel.hpp>
#include <internal/disaggregation/layer_completion_message.hpp>
#include <internal/disaggregation/layer_completion_queue.hpp>
#include <internal/disaggregation/layer_completion_reorder_buffer.hpp>

namespace tt::tt_metal::internal {

using tt::tt_metal::distributed::InterProcessCounterChannel;

namespace {
namespace mh = tt::tt_metal::distributed::multihost;
// Fixed MPI tag for layer-completion traffic. Distinct from any other
// host-to-host channel in the job.
constexpr mh::Tag kLayerCompletionTag{4242};

class CounterChannelEgress final : public SchedulerEgress {
public:
    explicit CounterChannelEgress(const std::string& shm_name) :
        channel_(std::make_unique<InterProcessCounterChannel>(shm_name)) {}
    void inject(uint32_t n) { channel_->inject(n); }
    void shutdown() override { channel_->shutdown(); }

private:
    std::unique_ptr<InterProcessCounterChannel> channel_;
};

class RingEgress final : public SchedulerEgress {
public:
    explicit RingEgress(const std::string& shm_name) : ring_(LayerCompletionQueueV2::create(shm_name)) {}
    bool try_push(const LayerCompletionMessageV2& m) { return ring_->try_push(m); }
    void shutdown() override { ring_->shutdown(); }

private:
    std::unique_ptr<LayerCompletionQueueV2> ring_;
};
}  // namespace

LayerCompletionRouter::LayerCompletionRouter(LayerCompletionRouterConfig cfg) : cfg_(std::move(cfg)) {
    TT_FATAL(
        !is_master() || !cfg_.scheduler_shm_name.empty(),
        "LayerCompletionRouter: master requires scheduler_shm_name (protocol {})",
        static_cast<int>(cfg_.protocol));
    switch (cfg_.protocol) {
        case LayerCompletionProtocol::kCountOnlyV1:
            queue_ = LayerCompletionQueue::create(cfg_.ring_shm_name);
            if (is_master()) {
                sched_egress_ = std::make_unique<CounterChannelEgress>(cfg_.scheduler_shm_name);
            }
            break;
        case LayerCompletionProtocol::kStructuredV2:
            queue_ = LayerCompletionQueueV2::create(cfg_.ring_shm_name);
            if (is_master()) {
                sched_egress_ = std::make_unique<RingEgress>(cfg_.scheduler_shm_name);
            }
            break;
    }
    listener_ = std::thread([this] {
        if (is_master()) {
            run_master();
        } else {
            run_subordinate();
        }
    });
}

LayerCompletionRouter::~LayerCompletionRouter() { stop(); }

void LayerCompletionRouter::stop() {
    if (stopped_.exchange(true)) {
        return;
    }
    stop_.store(true, std::memory_order_release);
    if (listener_.joinable()) {
        listener_.join();
    }
    if (queue_) {
        queue_->shutdown();
    }
    if (sched_egress_) {
        sched_egress_->shutdown();
    }
}

template <typename MsgT, typename Forward>
void LayerCompletionRouter::run_master_impl(Forward&& forward) {
    auto& queue = static_cast<LayerCompletionQueueT<MsgT>&>(*queue_);

    // Arm one irecv per subordinate (only when there is real MPI traffic).
    std::vector<int> subs;
    if (cfg_.world_size > 1) {
        for (int r = 0; r < cfg_.world_size; ++r) {
            if (r != cfg_.master_rank) {
                subs.push_back(r);
            }
        }
    }
    using Buf = std::array<std::byte, sizeof(MsgT)>;
    std::vector<Buf> bufs(subs.size());
    std::vector<mh::RequestPtr> reqs(subs.size());
    const mh::ContextPtr ctx = subs.empty() ? nullptr : mh::DistributedContext::get_current_world();
    for (std::size_t i = 0; i < subs.size(); ++i) {
        reqs[i] =
            ctx->irecv(ttsl::Span<std::byte>(bufs[i].data(), bufs[i].size()), mh::Rank(subs[i]), kLayerCompletionTag);
    }

    // Coordinated teardown: keep draining the local ring and receiving subordinate messages until
    // this rank is done producing (stop_) AND its ring is empty AND every subordinate has sent its
    // end-of-stream sentinel. Then no blocking subordinate send is ever left without a receiver, and
    // no already-arrived completion is dropped by a cancel. teardown_timeout_ms bounds the wait in
    // case a rank crashed without sending its sentinel.
    std::size_t sentinels_remaining = subs.size();
    std::optional<std::chrono::steady_clock::time_point> deadline;
    MsgT m{};
    while (true) {
        bool progressed = false;

        while (queue.try_pop(m)) {
            forward(m);
            progressed = true;
        }

        for (std::size_t i = 0; i < subs.size(); ++i) {
            if (reqs[i] && reqs[i]->test().has_value()) {
                MsgT recv{};
                std::memcpy(&recv, bufs[i].data(), sizeof(recv));
                progressed = true;
                if (is_layer_completion_sentinel(recv)) {
                    // End of stream from this subordinate — it sends nothing more; stop re-arming.
                    reqs[i].reset();
                    --sentinels_remaining;
                } else {
                    forward(recv);
                    reqs[i] = ctx->irecv(
                        ttsl::Span<std::byte>(bufs[i].data(), bufs[i].size()), mh::Rank(subs[i]), kLayerCompletionTag);
                }
            }
        }

        if (stop_.load(std::memory_order_acquire)) {
            // The runner stops pushing before it sets stop_, so the ring drain above leaves it empty.
            // Exit once every subordinate has also signalled end of stream — no cancel needed.
            if (sentinels_remaining == 0) {
                break;
            }
            if (!deadline) {
                deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(cfg_.teardown_timeout_ms);
            } else if (std::chrono::steady_clock::now() >= *deadline) {
                log_warning(
                    LogMetal,
                    "LayerCompletionRouter master (protocol {}): teardown timed out after {} ms with {} "
                    "subordinate sentinel(s) outstanding; cancelling — a stalled/crashed rank's tail "
                    "completions may be lost",
                    static_cast<int>(cfg_.protocol),
                    cfg_.teardown_timeout_ms,
                    sentinels_remaining);
                break;
            }
        }

        if (!progressed) {
            std::this_thread::sleep_for(std::chrono::microseconds(cfg_.poll_idle_us));
        }
    }

    // Only non-empty if we broke on the timeout above (a subordinate never sent its sentinel); in the
    // clean path every irecv was consumed or reset. Cancel so MPI can release the buffers.
    for (auto& r : reqs) {
        if (r && r->active()) {
            r->cancel();
        }
    }
}

void LayerCompletionRouter::run_master() {
    switch (cfg_.protocol) {
        case LayerCompletionProtocol::kCountOnlyV1: {
            auto& egress = static_cast<CounterChannelEgress&>(*sched_egress_);
            LayerCompletionReorderBuffer reorder;
            std::vector<LayerCompletionMessage> drained;
            run_master_impl<LayerCompletionMessage>([&](const LayerCompletionMessage& m) {
                const uint32_t n = reorder.insert(m, drained);
                if (n > 0) {
                    egress.inject(n);
                    processed_.fetch_add(n, std::memory_order_relaxed);
                }
            });
            break;
        }
        case LayerCompletionProtocol::kStructuredV2: {
            auto& egress = static_cast<RingEgress&>(*sched_egress_);
            uint64_t dropped = 0;
            bool scheduler_gone = false;
            run_master_impl<LayerCompletionMessageV2>([&](const LayerCompletionMessageV2& m) {
                if (scheduler_gone) {
                    ++dropped;
                    return;
                }
                std::optional<std::chrono::steady_clock::time_point> blocked_since;
                while (!egress.try_push(m)) {
                    const auto now = std::chrono::steady_clock::now();
                    if (!blocked_since) {
                        blocked_since = now;
                    } else if (
                        stop_.load(std::memory_order_acquire) &&
                        now - *blocked_since >= std::chrono::milliseconds(cfg_.teardown_timeout_ms)) {
                        scheduler_gone = true;
                        ++dropped;
                        return;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(cfg_.poll_idle_us));
                }
                processed_.fetch_add(1, std::memory_order_relaxed);
            });
            if (dropped > 0) {
                log_warning(
                    LogMetal,
                    "LayerCompletionRouter master (v2): dropped {} completion(s) — scheduler ring still full "
                    "{} ms after stop (scheduler wedged or gone)",
                    dropped,
                    cfg_.teardown_timeout_ms);
            }
            break;
        }
    }
}

void LayerCompletionRouter::run_subordinate() {
    switch (cfg_.protocol) {
        case LayerCompletionProtocol::kCountOnlyV1:
            run_subordinate_impl(static_cast<LayerCompletionQueue&>(*queue_));
            break;
        case LayerCompletionProtocol::kStructuredV2:
            run_subordinate_impl(static_cast<LayerCompletionQueueV2&>(*queue_));
            break;
    }
}

template <typename MsgT>
void LayerCompletionRouter::run_subordinate_impl(LayerCompletionQueueT<MsgT>& queue) {
    const mh::ContextPtr& ctx = mh::DistributedContext::get_current_world();
    auto send_blocking = [&](const MsgT& msg) {
        std::array<std::byte, sizeof(msg)> buf{};
        std::memcpy(buf.data(), &msg, sizeof(msg));
        ctx->send(ttsl::Span<std::byte>(buf.data(), buf.size()), mh::Rank(cfg_.master_rank), kLayerCompletionTag);
    };
    // Teardown sends are bounded (isend + deadline) so this thread can't wedge — and so hang stop() /
    // the dtor join — if the master already hit its own teardown_timeout_ms, stopped receiving, and
    // cancelled. Returns false when the send didn't complete in time (master gone). Symmetric with the
    // master's bound; the clean path never trips it.
    auto send_bounded = [&](const MsgT& msg) -> bool {
        std::array<std::byte, sizeof(msg)> buf{};
        std::memcpy(buf.data(), &msg, sizeof(msg));
        auto req =
            ctx->isend(ttsl::Span<std::byte>(buf.data(), buf.size()), mh::Rank(cfg_.master_rank), kLayerCompletionTag);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(cfg_.teardown_timeout_ms);
        while (!req->test().has_value()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                req->cancel();
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(cfg_.poll_idle_us));
        }
        return true;
    };

    MsgT m{};
    while (!stop_.load(std::memory_order_acquire)) {
        if (queue.try_pop(m)) {
            send_blocking(m);  // steady state: the master is actively receiving
            processed_.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(cfg_.poll_idle_us));
        }
    }
    // Teardown: drain anything that arrived between the last pop and stop_, then send the end-of-stream
    // sentinel — all via bounded sends. The master keeps a receive posted until it sees the sentinel
    // (run_master), so in the clean path every send completes promptly; if a send times out the master
    // has already given up, so abandon the rest (those completions are unrecoverable either way).
    bool master_alive = true;
    while (master_alive && queue.try_pop(m)) {
        master_alive = send_bounded(m);
        if (master_alive) {
            processed_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (master_alive) {
        const MsgT sentinel = layer_completion_sentinel<MsgT>(static_cast<uint32_t>(cfg_.rank));
        master_alive = send_bounded(sentinel);
    }
    if (!master_alive) {
        std::size_t lost = 1;  // the send that timed out (a completion or the sentinel)
        while (queue.try_pop(m)) {
            ++lost;
        }
        log_warning(
            LogMetal,
            "LayerCompletionRouter rank {}: master not receiving within {} ms at teardown; abandoning ~{} "
            "undelivered message(s) (master likely timed out or crashed)",
            cfg_.rank,
            cfg_.teardown_timeout_ms,
            lost);
    }
}

}  // namespace tt::tt_metal::internal
