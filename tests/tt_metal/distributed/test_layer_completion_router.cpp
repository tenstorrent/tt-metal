// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>

#include <internal/service/inter_process_counter_channel.hpp>
#include <layer_completion_message.hpp>
#include <layer_completion_queue.hpp>
#include <layer_completion_router.hpp>

namespace tt::tt_metal::distributed::test {

namespace {
void unlink_shm(const std::string& name) { std::remove(("/dev/shm" + name).c_str()); }

LayerCompletionMessage msg(uint64_t seq) {
    return LayerCompletionMessage{seq, /*source_rank=*/0u, /*layer_idx=*/static_cast<uint32_t>(seq), 0u, 0u};
}

// Spin until `pred()` or the deadline; keeps the test from hanging if the
// router thread never makes progress.
template <typename Pred>
bool wait_for(Pred pred, std::chrono::milliseconds timeout = std::chrono::milliseconds(2000)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return pred();
}
}  // namespace

// Single-rank master: completions pushed (out of order) to the local ring
// must be injected into the counter channel IN ORDER, one inject per
// contiguous seq. No MPI involved (world_size == 1).
TEST(LayerCompletionRouter, MasterReordersLocalRingIntoCounterChannel) {
    const std::string ring = "/tt_lcr_test_ring";
    const std::string sched = "/tt_lcr_test_sched";
    unlink_shm(ring);
    unlink_shm(sched);

    LayerCompletionRouterConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 1;
    cfg.master_rank = 0;
    cfg.ring_shm_name = ring;
    cfg.scheduler_channel_shm_name = sched;

    auto router = std::make_unique<LayerCompletionRouter>(cfg);
    EXPECT_TRUE(router->is_master());

    // Scheduler side: connect to the counter channel the router created.
    auto consumer = InterProcessCounterChannel::connect(sched, 5'000);

    // Producer side: connect to the ring and push 0..7 OUT OF ORDER.
    auto producer = LayerCompletionQueue::connect(ring, 5'000);
    const uint64_t order[] = {0, 2, 1, 4, 3, 6, 5, 7};
    for (uint64_t s : order) {
        ASSERT_TRUE(producer->try_push(msg(s)));
    }

    ASSERT_TRUE(wait_for([&] { return router->processed() == 8; }));

    // The counter channel saw exactly 8 ordered injects.
    EXPECT_EQ(consumer->try_consume_all(), 8u);

    router->stop();
    router.reset();
    consumer->shutdown();
}

}  // namespace tt::tt_metal::distributed::test
