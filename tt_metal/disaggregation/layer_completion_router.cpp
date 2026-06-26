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

    // Drain completions this (master) rank produced into its own ring between the last pop
    // and stop_ being set — mirror run_subordinate(); the master is itself a producer to this
    // ring, so without this its own last chunk's trailing layers are silently dropped (never
    // injected into the scheduler channel). (Subordinate-side in-flight messages still depend
    // on coordinated cross-rank teardown — tracked separately.)
    while (queue_->try_pop(m)) {
        ingest(m);
    }

    // Final sweep: harvest any subordinate completion that already landed in its receive buffer
    // before stop_ was observed — otherwise the cancel() below would discard a physically-arrived
    // message (and a lost seq head-of-line-blocks the reorder buffer). Then cancel whatever is still
    // genuinely outstanding so MPI can release it.
    //
    // RESIDUAL RACE (uncoordinated-teardown future work): a message that completes the irecv in the
    // window between this test() returning nullopt and the cancel() below is still lost — cancel()
    // does MPI_Cancel + MPI_Request_free with no MPI_Wait/MPI_Test_cancelled, so a request that
    // actually received data is freed and its bytes dropped, and that missing seq head-of-line-blocks
    // the reorder buffer. Closing it needs coordinated shutdown (subordinates send an end-of-stream
    // sentinel; the master drains until it has seen one per subordinate, then stops without cancel),
    // not a test-after-cancel (the request is already freed). Same root cause as the run_subordinate
    // blocking-send hang below.
    for (std::size_t i = 0; i < subs.size(); ++i) {
        if (reqs[i] && reqs[i]->test().has_value()) {
            LayerCompletionMessage recv{};
            std::memcpy(&recv, bufs[i].data(), sizeof(recv));
            ingest(recv);
            reqs[i].reset();
        }
    }
    for (auto& r : reqs) {
        if (r && r->active()) {
            r->cancel();
        }
    }
}

void LayerCompletionRouter::run_subordinate() {
    // TEARDOWN HANG (uncoordinated-teardown future work): the ctx->send() below is blocking. If this
    // rank sends after the master has already broken out of run_master and cancelled its matching
    // irecv (steady-state or in the post-stop drain), the send has no receiver and blocks forever, so
    // this thread never returns and stop()/~LayerCompletionRouter's join() hangs. The fix is the same
    // coordinated shutdown noted in run_master (drain to an end-of-stream sentinel, no mid-stream
    // cancel); a blocking send with no teardown handshake cannot be made safe on its own.
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
