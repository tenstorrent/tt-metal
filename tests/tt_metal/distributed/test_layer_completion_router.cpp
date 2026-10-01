// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
#include <internal/service/inter_process_counter_channel.hpp>

namespace tt::tt_metal::internal {

namespace {

using tt::tt_metal::distributed::InterProcessCounterChannel;

void unlink_if_exists(const std::string& shm_name) { std::remove(("/dev/shm" + shm_name).c_str()); }

std::string fresh_name(const char* tag) {
    static std::atomic<uint32_t> counter{0};
    return "/tt_lcr_test_" + std::string(tag) + "_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter.fetch_add(1));
}

template <typename F>
bool wait_until(F&& f, int timeout_ms = 10'000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (f()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return f();
}

LayerCompletionMessage v1_msg(uint64_t seq) {
    return LayerCompletionMessage{seq, 0u, static_cast<uint32_t>(seq), 0u, 0u};
}

LayerCompletionMessageV2 v2_msg(uint64_t seq, uint32_t request_id, uint32_t layer_start, uint32_t layer_end) {
    return LayerCompletionMessageV2{
        seq,
        0u,
        request_id,
        2u,
        100u * request_id,
        100u * request_id + 50u,
        layer_start,
        layer_end,
        0u,
        1'000'000u + seq};
}

}  // namespace

TEST(LayerCompletionRouter, V1MasterReordersAndCounts) {
    const std::string ring = fresh_name("v1_ring");
    const std::string channel = fresh_name("v1_chan");
    unlink_if_exists(ring);
    unlink_if_exists(channel);

    LayerCompletionRouterConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 1;
    cfg.master_rank = 0;
    cfg.ring_shm_name = ring;
    cfg.protocol = LayerCompletionProtocol::kCountOnlyV1;
    cfg.scheduler_shm_name = channel;
    auto router = std::make_unique<LayerCompletionRouter>(std::move(cfg));

    auto producer = LayerCompletionQueue::connect(ring, 5'000);
    auto consumer = InterProcessCounterChannel::connect(channel, 5'000);

    ASSERT_TRUE(producer->try_push(v1_msg(1)));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_EQ(consumer->pending(), 0u);

    ASSERT_TRUE(producer->try_push(v1_msg(0)));
    ASSERT_TRUE(wait_until([&] { return consumer->try_consume_all() == 2; }));
    EXPECT_EQ(router->processed(), 2u);

    producer->shutdown();
    consumer->shutdown();
    router->stop();
}

TEST(LayerCompletionRouter, V2MasterForwardsAsArrived) {
    const std::string ring = fresh_name("v2_ring");
    const std::string sched = fresh_name("v2_sched");
    unlink_if_exists(ring);
    unlink_if_exists(sched);

    LayerCompletionRouterConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 1;
    cfg.master_rank = 0;
    cfg.ring_shm_name = ring;
    cfg.protocol = LayerCompletionProtocol::kStructuredV2;
    cfg.scheduler_shm_name = sched;
    auto router = std::make_unique<LayerCompletionRouter>(std::move(cfg));

    auto producer = LayerCompletionQueueV2::connect(ring, 5'000);
    auto scheduler = LayerCompletionQueueV2::connect(sched, 5'000);

    ASSERT_TRUE(producer->try_push(v2_msg(1, 0, 5, 6)));
    LayerCompletionMessageV2 out{};
    ASSERT_TRUE(wait_until([&] { return scheduler->try_pop(out); }));
    EXPECT_EQ(out.seq, 1u);
    EXPECT_EQ(out.layer_start, 5u);
    EXPECT_EQ(out.layer_end, 6u);
    EXPECT_EQ(out.host_ts_ns, 1'000'001u);

    ASSERT_TRUE(producer->try_push(v2_msg(7, 3, 0, 1)));
    ASSERT_TRUE(producer->try_push(v2_msg(0, 0, 0, 1)));
    ASSERT_TRUE(producer->try_push(v2_msg(3, 1, 0, 14)));
    const uint64_t want_order[] = {7, 0, 3};
    for (uint64_t want : want_order) {
        ASSERT_TRUE(wait_until([&] { return scheduler->try_pop(out); }));
        EXPECT_EQ(out.seq, want);
    }
    EXPECT_EQ(out.layer_end - out.layer_start, 14u);
    ASSERT_TRUE(wait_until([&] { return router->processed() == 4; }));

    producer->shutdown();
    scheduler->shutdown();
    router->stop();
}

TEST(LayerCompletionRouter, V2SchedulerRingBackpressure) {
    const std::string ring = fresh_name("v2bp_ring");
    const std::string sched = fresh_name("v2bp_sched");
    unlink_if_exists(ring);
    unlink_if_exists(sched);

    LayerCompletionRouterConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 1;
    cfg.master_rank = 0;
    cfg.ring_shm_name = ring;
    cfg.protocol = LayerCompletionProtocol::kStructuredV2;
    cfg.scheduler_shm_name = sched;
    auto router = std::make_unique<LayerCompletionRouter>(std::move(cfg));

    auto producer = LayerCompletionQueueV2::connect(ring, 5'000);
    auto scheduler = LayerCompletionQueueV2::connect(sched, 5'000);

    const uint64_t total = LayerCompletionQueueV2::capacity() + 17;
    for (uint64_t i = 0; i < total; ++i) {
        while (!producer->try_push(v2_msg(i, 0, static_cast<uint32_t>(i % 61), static_cast<uint32_t>(i % 61) + 1))) {
            std::this_thread::yield();
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_EQ(router->processed(), LayerCompletionQueueV2::capacity());

    LayerCompletionMessageV2 out{};
    for (uint64_t i = 0; i < total; ++i) {
        ASSERT_TRUE(wait_until([&] { return scheduler->try_pop(out); })) << "stuck at " << i;
        EXPECT_EQ(out.seq, i);
    }
    ASSERT_TRUE(wait_until([&] { return router->processed() == total; }));

    producer->shutdown();
    scheduler->shutdown();
    router->stop();
}

TEST(LayerCompletionRouter, V2TeardownDropsWhenSchedulerGone) {
    const std::string ring = fresh_name("v2td_ring");
    const std::string sched = fresh_name("v2td_sched");
    unlink_if_exists(ring);
    unlink_if_exists(sched);

    LayerCompletionRouterConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 1;
    cfg.master_rank = 0;
    cfg.ring_shm_name = ring;
    cfg.protocol = LayerCompletionProtocol::kStructuredV2;
    cfg.scheduler_shm_name = sched;
    cfg.teardown_timeout_ms = 500;
    auto router = std::make_unique<LayerCompletionRouter>(std::move(cfg));

    auto producer = LayerCompletionQueueV2::connect(ring, 5'000);
    for (uint64_t i = 0; i < LayerCompletionQueueV2::capacity() + 64; ++i) {
        while (!producer->try_push(v2_msg(i, 0, 0, 1))) {
            std::this_thread::yield();
        }
    }
    ASSERT_TRUE(wait_until([&] { return router->processed() == LayerCompletionQueueV2::capacity(); }));

    const auto t0 = std::chrono::steady_clock::now();
    router->stop();
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 30'000);
    producer->shutdown();
}

TEST(LayerCompletionRouter, ConfigValidation) {
    LayerCompletionRouterConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 1;
    cfg.master_rank = 0;
    cfg.ring_shm_name = fresh_name("cfg_ring");

    cfg.protocol = LayerCompletionProtocol::kCountOnlyV1;
    EXPECT_THROW(std::make_unique<LayerCompletionRouter>(cfg), std::exception);

    cfg.protocol = LayerCompletionProtocol::kStructuredV2;
    EXPECT_THROW(std::make_unique<LayerCompletionRouter>(cfg), std::exception);

    unlink_if_exists(cfg.ring_shm_name);
}

}  // namespace tt::tt_metal::internal
