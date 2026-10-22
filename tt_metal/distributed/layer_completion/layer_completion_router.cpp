// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/layer_completion_router.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <mutex>
#include <optional>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/distributed_context.hpp>

#include <internal/disaggregation/layer_completion_message.hpp>
#include <internal/disaggregation/layer_completion_queue.hpp>

namespace tt::tt_metal::internal {

namespace {
namespace mh = tt::tt_metal::distributed::multihost;
constexpr mh::Tag kLayerCompletionTag{4242};
}  // namespace

LayerCompletionRouter::LayerCompletionRouter(LayerCompletionRouterConfig cfg) : cfg_(std::move(cfg)) {
    TT_FATAL(
        !is_master() || !cfg_.scheduler_shm_name.empty(), "LayerCompletionRouter: master requires scheduler_shm_name");
    TT_FATAL(cfg_.completions_per_chunk > 0, "LayerCompletionRouter: completions_per_chunk must be > 0");
    ring_ = LayerCompletionQueue::create(cfg_.ring_shm_name, cfg_.completions_per_chunk);
    if (is_master()) {
        scheduler_ = LayerCompletionQueue::create(cfg_.scheduler_shm_name, cfg_.completions_per_chunk);
    }
    listener_ = std::thread([this] { listen(); });
}

LayerCompletionRouter::~LayerCompletionRouter() { join(); }

void LayerCompletionRouter::listen() {
    try {
        if (is_master()) {
            run_master();
        } else {
            run_subordinate();
        }
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
    if (ring_) {
        ring_->shutdown();
    }
    if (scheduler_) {
        scheduler_->shutdown();
    }
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

template <typename Forward>
void LayerCompletionRouter::fan_in(const Forward& forward) {
    std::vector<int> subs;
    if (cfg_.world_size > 1) {
        for (int r = 0; r < cfg_.world_size; ++r) {
            if (r != cfg_.master_rank) {
                subs.push_back(r);
            }
        }
    }
    using Buf = std::array<std::byte, sizeof(LayerCompletionMessage)>;
    std::vector<Buf> bufs(subs.size());
    std::vector<mh::RequestPtr> reqs(subs.size());
    const mh::ContextPtr ctx = subs.empty() ? nullptr : mh::DistributedContext::get_current_world();
    for (std::size_t i = 0; i < subs.size(); ++i) {
        reqs[i] =
            ctx->irecv(ttsl::Span<std::byte>(bufs[i].data(), bufs[i].size()), mh::Rank(subs[i]), kLayerCompletionTag);
    }

    std::size_t sentinels_remaining = subs.size();
    std::optional<std::chrono::steady_clock::time_point> deadline;
    LayerCompletionMessage m{};
    while (true) {
        bool progressed = false;

        while (ring_->try_pop(m)) {
            forward(m);
            progressed = true;
        }

        for (std::size_t i = 0; i < subs.size(); ++i) {
            if (reqs[i] && reqs[i]->test().has_value()) {
                LayerCompletionMessage recv{};
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
            // The runner stops pushing before stop(), so the ring is empty here; exit once every sentinel is in.
            if (sentinels_remaining == 0) {
                break;
            }
            if (!deadline) {
                deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(cfg_.teardown_timeout_ms);
            } else if (std::chrono::steady_clock::now() >= *deadline) {
                log_warning(
                    LogMetal,
                    "LayerCompletionRouter master: teardown timed out after {} ms with {} subordinate sentinel(s) "
                    "outstanding; cancelling — a stalled/crashed rank's tail completions may be lost",
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

void LayerCompletionRouter::run_master() {
    // A full scheduler ring is backpressure. After stop() the wait is bounded by teardown_timeout_ms, then the
    // backlog is dropped and counted.
    uint64_t dropped = 0;
    bool scheduler_gone = false;
    fan_in([&](const LayerCompletionMessage& m) {
        if (scheduler_gone) {
            ++dropped;
            return;
        }
        while (!scheduler_->try_push(m)) {
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
            "LayerCompletionRouter master: dropped {} completion(s) — scheduler ring still full {} ms after stop "
            "(scheduler wedged or gone)",
            dropped,
            cfg_.teardown_timeout_ms);
    }
}

void LayerCompletionRouter::run_subordinate() {
    const mh::ContextPtr& ctx = mh::DistributedContext::get_current_world();
    auto send_blocking = [&](const LayerCompletionMessage& msg) {
        std::array<std::byte, sizeof(msg)> buf{};
        std::memcpy(buf.data(), &msg, sizeof(msg));
        ctx->send(ttsl::Span<std::byte>(buf.data(), buf.size()), mh::Rank(cfg_.master_rank), kLayerCompletionTag);
    };
    // Teardown sends are bounded so a master that already gave up cannot wedge this thread.
    auto send_bounded = [&](const LayerCompletionMessage& msg) -> bool {
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

    LayerCompletionMessage m{};
    while (!stop_.load(std::memory_order_acquire)) {
        if (ring_->try_pop(m)) {
            send_blocking(m);
            processed_.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(cfg_.poll_idle_us));
        }
    }
    // Drain what arrived before stop_, then the end-of-stream sentinel.
    bool master_alive = true;
    while (master_alive && ring_->try_pop(m)) {
        master_alive = send_bounded(m);
        if (master_alive) {
            processed_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (master_alive) {
        master_alive = send_bounded(layer_completion_sentinel());
    }
    if (!master_alive) {
        std::size_t lost = 1;
        while (ring_->try_pop(m)) {
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
