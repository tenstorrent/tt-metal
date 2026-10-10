// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/layer_completion_router.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <mutex>
#include <optional>
#include <variant>
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

namespace {
namespace mh = tt::tt_metal::distributed::multihost;
constexpr mh::Tag kLayerCompletionTag{4242};  // distinct from every other host-to-host channel
}  // namespace

LayerCompletionRouter::LayerCompletionRouter(LayerCompletionRouterConfig cfg) : cfg_(std::move(cfg)) {
    TT_FATAL(
        !is_master() || !cfg_.scheduler_shm_name.empty(),
        "LayerCompletionRouter: master requires scheduler_shm_name (protocol {})",
        static_cast<int>(cfg_.protocol));
    switch (cfg_.protocol) {
        case LayerCompletionProtocol::kCountOnlyV1: {
            CountOnly state{LayerCompletionQueue::create(cfg_.ring_shm_name), nullptr};
            if (is_master()) {
                state.scheduler = std::make_unique<InterProcessCounterChannel>(cfg_.scheduler_shm_name);
            }
            state_ = std::move(state);
            break;
        }
        case LayerCompletionProtocol::kStructuredV2: {
            Structured state{LayerCompletionQueueV2::create(cfg_.ring_shm_name), nullptr};
            if (is_master()) {
                state.scheduler = LayerCompletionQueueV2::create(cfg_.scheduler_shm_name);
            }
            state_ = std::move(state);
            break;
        }
    }
    listener_ = std::thread([this] { listen(); });
}

LayerCompletionRouter::~LayerCompletionRouter() { join(); }

void LayerCompletionRouter::listen() {
    try {
        std::visit(
            [this](auto& state) {
                if (is_master()) {
                    run_master(state);
                } else {
                    run_subordinate(*state.ring);
                }
            },
            state_);
    } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_ = std::current_exception();
    }
}

void LayerCompletionRouter::join() {
    if (stopped_.exchange(true)) {
        return;
    }
    stopped_at_ = std::chrono::steady_clock::now();
    stop_.store(true, std::memory_order_release);
    if (listener_.joinable()) {
        listener_.join();
    }
    std::visit(
        [](auto& state) {
            if (state.ring) {
                state.ring->shutdown();
            }
            if (state.scheduler) {
                state.scheduler->shutdown();
            }
        },
        state_);
}

void LayerCompletionRouter::stop() {
    join();
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error = std::exchange(error_, nullptr);
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

template <typename MsgT, typename Forward>
void LayerCompletionRouter::fan_in(LayerCompletionQueueT<MsgT>& ring, Forward&& forward) {
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

    std::size_t sentinels_remaining = subs.size();
    std::optional<std::chrono::steady_clock::time_point> deadline;
    MsgT m{};
    while (true) {
        bool progressed = false;

        while (ring.try_pop(m)) {
            forward(m);
            progressed = true;
        }

        for (std::size_t i = 0; i < subs.size(); ++i) {
            if (reqs[i] && reqs[i]->test().has_value()) {
                MsgT recv{};
                std::memcpy(&recv, bufs[i].data(), sizeof(recv));
                progressed = true;
                if (is_layer_completion_sentinel(recv)) {
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
            // The runner stops pushing before stop(), so the drain above empties the ring; exit once
            // every subordinate has sent its sentinel.
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
    for (auto& r : reqs) {
        if (r && r->active()) {
            r->cancel();
        }
    }
}

void LayerCompletionRouter::run_master(CountOnly& state) {
    LayerCompletionReorderBuffer reorder;
    std::vector<LayerCompletionMessage> drained;
    fan_in(*state.ring, [&](const LayerCompletionMessage& m) {
        const uint32_t n = reorder.insert(m, drained);
        if (n > 0) {
            state.scheduler->inject(n);
            processed_.fetch_add(n, std::memory_order_relaxed);
        }
    });
}

void LayerCompletionRouter::run_master(Structured& state) {
    // A full scheduler ring is backpressure: wait. After stop() the wait is bounded by
    // teardown_timeout_ms measured from stop(), then the backlog is dropped and counted.
    uint64_t dropped = 0;
    bool scheduler_gone = false;
    fan_in(*state.ring, [&](const LayerCompletionMessageV2& m) {
        if (scheduler_gone) {
            ++dropped;
            return;
        }
        while (!state.scheduler->try_push(m)) {
            if (stop_.load(std::memory_order_acquire) &&
                std::chrono::steady_clock::now() - stopped_at_ >= std::chrono::milliseconds(cfg_.teardown_timeout_ms)) {
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
            "LayerCompletionRouter master (v2): dropped {} completion(s) — scheduler ring still full {} ms "
            "after stop (scheduler wedged or gone)",
            dropped,
            cfg_.teardown_timeout_ms);
    }
}

template <typename MsgT>
void LayerCompletionRouter::run_subordinate(LayerCompletionQueueT<MsgT>& ring) {
    const mh::ContextPtr ctx = mh::DistributedContext::get_current_world();
    auto send_blocking = [&](const MsgT& msg) {
        std::array<std::byte, sizeof(msg)> buf{};
        std::memcpy(buf.data(), &msg, sizeof(msg));
        ctx->send(ttsl::Span<std::byte>(buf.data(), buf.size()), mh::Rank(cfg_.master_rank), kLayerCompletionTag);
    };
    // Teardown sends are bounded so a master that already gave up cannot wedge this thread.
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
        if (ring.try_pop(m)) {
            send_blocking(m);
            processed_.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(cfg_.poll_idle_us));
        }
    }
    // Drain what arrived before stop_, then the end-of-stream sentinel.
    bool master_alive = true;
    while (master_alive && ring.try_pop(m)) {
        master_alive = send_bounded(m);
        if (master_alive) {
            processed_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (master_alive) {
        master_alive = send_bounded(layer_completion_sentinel<MsgT>(static_cast<uint32_t>(cfg_.rank)));
    }
    if (!master_alive) {
        std::size_t lost = 1;
        while (ring.try_pop(m)) {
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
