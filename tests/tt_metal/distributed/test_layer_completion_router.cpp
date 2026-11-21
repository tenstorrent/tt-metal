// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-only tests for LayerCompletionRouter at world_size=1: the master forwards messages as they arrive into
// the scheduler-facing ring, with backpressure when it is full and a bounded drop at teardown.

#include <gtest/gtest.h>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>

#include <internal/disaggregation/layer_completion_message.hpp>
#include <internal/disaggregation/layer_completion_queue.hpp>
#include <internal/disaggregation/layer_completion_router.hpp>

namespace tt::tt_metal::internal {

namespace {

void unlink_if_exists(const std::string& shm_name) { std::remove(("/dev/shm" + shm_name).c_str()); }

std::string fresh_name(const char* tag) {
    static std::atomic<uint32_t> counter{0};
    return "/tt_lcr_test_" + std::string(tag) + "_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter.fetch_add(1));
}

// Poll `f` until it holds or the deadline passes; the router runs on its own thread.
template <typename F>
bool wait_until(const F& f, int timeout_ms = 10'000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (f()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return f();
}

// `tag` is carried in pos_start so arrival order can be checked on the scheduler side.
LayerCompletionMessage make_msg(uint32_t tag, uint32_t layer_start, uint32_t layer_end) {
    return LayerCompletionMessage{
        /*slot_id=*/2u, /*pos_start=*/tag, /*pos_end=*/tag + 50u, layer_start, layer_end, /*flags=*/0u};
}

}  // namespace

// Messages are forwarded in arrival order, with no reordering or gap wait.
TEST(LayerCompletionRouter, MasterForwardsAsArrived) {
    const std::string ring = fresh_name("fwd_ring");
    const std::string sched = fresh_name("fwd_sched");
    unlink_if_exists(ring);
    unlink_if_exists(sched);

    LayerCompletionRouterConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 1;
    cfg.master_rank = 0;
    cfg.ring_shm_name = ring;
    cfg.scheduler_shm_name = sched;
    cfg.completions_per_chunk = 61;
    auto router = std::make_unique<LayerCompletionRouter>(std::move(cfg));

    auto producer = LayerCompletionQueue::connect(ring, 5'000);
    auto scheduler = LayerCompletionQueue::connect(sched, 5'000);

    ASSERT_TRUE(producer->try_push(make_msg(/*tag=*/1, 5, 6)));
    LayerCompletionMessage out{};
    ASSERT_TRUE(wait_until([&] { return scheduler->try_pop(out); }));
    EXPECT_EQ(out.pos_start, 1u);
    EXPECT_EQ(out.layer_start, 5u);
    EXPECT_EQ(out.layer_end, 6u);

    // Interleaved chunks: arrival order is preserved end to end.
    ASSERT_TRUE(producer->try_push(make_msg(7, 0, 1)));
    ASSERT_TRUE(producer->try_push(make_msg(0, 0, 1)));
    ASSERT_TRUE(producer->try_push(make_msg(3, 0, 14)));  // one message, 14 layers
    const uint32_t want_order[] = {7, 0, 3};
    for (uint32_t want : want_order) {
        ASSERT_TRUE(wait_until([&] { return scheduler->try_pop(out); }));
        EXPECT_EQ(out.pos_start, want);
    }
    EXPECT_EQ(out.layer_end - out.layer_start, 14u);
    ASSERT_TRUE(wait_until([&] { return router->processed() == 4; }));

    producer->shutdown();
    scheduler->shutdown();
    router->stop();
}

// A full scheduler ring stalls the forward; draining resumes it in order.
TEST(LayerCompletionRouter, SchedulerRingBackpressure) {
    const std::string ring = fresh_name("bp_ring");
    const std::string sched = fresh_name("bp_sched");
    unlink_if_exists(ring);
    unlink_if_exists(sched);

    LayerCompletionRouterConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 1;
    cfg.master_rank = 0;
    cfg.ring_shm_name = ring;
    cfg.scheduler_shm_name = sched;
    cfg.completions_per_chunk = 61;
    auto router = std::make_unique<LayerCompletionRouter>(std::move(cfg));

    auto producer = LayerCompletionQueue::connect(ring, 5'000);
    auto scheduler = LayerCompletionQueue::connect(sched, 5'000);

    const uint32_t total = LayerCompletionQueue::capacity() + 17;
    for (uint32_t i = 0; i < total; ++i) {
        // The input ring itself provides backpressure once the master stalls.
        while (!producer->try_push(make_msg(i, i % 61, i % 61 + 1))) {
            std::this_thread::yield();
        }
    }

    // Master can only have forwarded capacity() messages — the scheduler ring is full.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_EQ(router->processed(), LayerCompletionQueue::capacity());

    // Drain everything; the master must resume and deliver the rest in order.
    LayerCompletionMessage out{};
    for (uint32_t i = 0; i < total; ++i) {
        ASSERT_TRUE(wait_until([&] { return scheduler->try_pop(out); })) << "stuck at " << i;
        EXPECT_EQ(out.pos_start, i);
    }
    ASSERT_TRUE(wait_until([&] { return router->processed() == total; }));

    producer->shutdown();
    scheduler->shutdown();
    router->stop();
}

// A full scheduler ring and no scheduler: stop() must return within the teardown bound.
TEST(LayerCompletionRouter, TeardownDropsWhenSchedulerGone) {
    const std::string ring = fresh_name("td_ring");
    const std::string sched = fresh_name("td_sched");
    unlink_if_exists(ring);
    unlink_if_exists(sched);

    LayerCompletionRouterConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 1;
    cfg.master_rank = 0;
    cfg.ring_shm_name = ring;
    cfg.scheduler_shm_name = sched;
    cfg.completions_per_chunk = 61;
    cfg.teardown_timeout_ms = 500;  // keep the test fast
    auto router = std::make_unique<LayerCompletionRouter>(std::move(cfg));

    auto producer = LayerCompletionQueue::connect(ring, 5'000);
    // No scheduler connector: fill the scheduler ring and overflow into the input ring.
    for (uint32_t i = 0; i < LayerCompletionQueue::capacity() + 64; ++i) {
        while (!producer->try_push(make_msg(i, 0, 1))) {
            std::this_thread::yield();
        }
    }
    ASSERT_TRUE(wait_until([&] { return router->processed() == LayerCompletionQueue::capacity(); }));

    const auto t0 = std::chrono::steady_clock::now();
    router->stop();  // must not wedge: bounded by teardown_timeout_ms per blocked push
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 30'000);
    producer->shutdown();
}

// A missing scheduler name or a zero completions_per_chunk fails at construction.
TEST(LayerCompletionRouter, ConfigValidation) {
    LayerCompletionRouterConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 1;
    cfg.master_rank = 0;
    cfg.ring_shm_name = fresh_name("cfg_ring");
    cfg.completions_per_chunk = 4;

    EXPECT_THROW(std::make_unique<LayerCompletionRouter>(cfg), std::exception);  // missing scheduler_shm_name

    cfg.scheduler_shm_name = fresh_name("cfg_sched");
    cfg.completions_per_chunk = 0;
    EXPECT_THROW(std::make_unique<LayerCompletionRouter>(cfg), std::exception);

    unlink_if_exists(cfg.ring_shm_name);
    unlink_if_exists(cfg.scheduler_shm_name);
}

}  // namespace tt::tt_metal::internal
