// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/layer_completion_router.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/distributed_context.hpp>

#include <internal/service/inter_process_counter_channel.hpp>
#include <internal/disaggregation/layer_completion_message.hpp>
#include <internal/disaggregation/layer_completion_queue.hpp>
#include <internal/disaggregation/layer_completion_reorder_buffer.hpp>

namespace tt::tt_metal::distributed {

namespace {
namespace mh = tt::tt_metal::distributed::multihost;
// Fixed MPI tag for layer-completion traffic. Distinct from any other
// host-to-host channel in the job.
constexpr mh::Tag kLayerCompletionTag{4242};
}  // namespace

LayerCompletionRouter::LayerCompletionRouter(LayerCompletionRouterConfig cfg) : cfg_(std::move(cfg)) {
    queue_ = LayerCompletionQueue::create(cfg_.ring_shm_name);
    if (is_master()) {
        counter_ = std::make_unique<InterProcessCounterChannel>(cfg_.scheduler_channel_shm_name);
        listener_ = std::thread([this] { run_master(); });
    } else {
        listener_ = std::thread([this] { run_subordinate(); });
    }
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
    if (counter_) {
        counter_->shutdown();
    }
}

void LayerCompletionRouter::run_master() {
    LayerCompletionReorderBuffer reorder;
    std::vector<LayerCompletionMessage> drained;

    // Arm one irecv per subordinate (only when there is real MPI traffic).
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

    auto ingest = [&](const LayerCompletionMessage& m) {
        const uint32_t n = reorder.insert(m, drained);
        if (n > 0) {
            counter_->inject(n);
            processed_.fetch_add(n, std::memory_order_relaxed);
        }
    };

    LayerCompletionMessage m{};
    while (!stop_.load(std::memory_order_acquire)) {
        bool progressed = false;

        while (queue_->try_pop(m)) {
            ingest(m);
            progressed = true;
        }

        for (std::size_t i = 0; i < subs.size(); ++i) {
            if (reqs[i] && reqs[i]->test().has_value()) {
                LayerCompletionMessage recv{};
                std::memcpy(&recv, bufs[i].data(), sizeof(recv));
                ingest(recv);
                reqs[i] = ctx->irecv(
                    ttsl::Span<std::byte>(bufs[i].data(), bufs[i].size()), mh::Rank(subs[i]), kLayerCompletionTag);
                progressed = true;
            }
        }

        if (!progressed) {
            std::this_thread::sleep_for(std::chrono::microseconds(cfg_.poll_idle_us));
        }
    }

    // Cancel still-outstanding receives so MPI can release them at teardown.
    for (auto& r : reqs) {
        if (r && r->active()) {
            r->cancel();
        }
    }
}

void LayerCompletionRouter::run_subordinate() {
    const mh::ContextPtr ctx = mh::DistributedContext::get_current_world();
    LayerCompletionMessage m{};
    while (!stop_.load(std::memory_order_acquire)) {
        if (queue_->try_pop(m)) {
            std::array<std::byte, sizeof(m)> buf{};
            std::memcpy(buf.data(), &m, sizeof(m));
            ctx->send(ttsl::Span<std::byte>(buf.data(), buf.size()), mh::Rank(cfg_.master_rank), kLayerCompletionTag);
            processed_.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(cfg_.poll_idle_us));
        }
    }
    // Drain any messages that arrived in the ring between the last pop and
    // stop_ being set — otherwise they would be silently dropped and the
    // master would never receive them.
    while (queue_->try_pop(m)) {
        std::array<std::byte, sizeof(m)> buf{};
        std::memcpy(buf.data(), &m, sizeof(m));
        ctx->send(ttsl::Span<std::byte>(buf.data(), buf.size()), mh::Rank(cfg_.master_rank), kLayerCompletionTag);
        processed_.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace tt::tt_metal::distributed
