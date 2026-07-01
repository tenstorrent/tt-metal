// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include <tt-metalium/distributed_context.hpp>

#include <internal/service/inter_process_counter_channel.hpp>
#include <layer_completion_message.hpp>
#include <layer_completion_queue.hpp>
#include <layer_completion_router.hpp>

namespace tt::tt_metal::distributed::test {

namespace mh = tt::tt_metal::distributed::multihost;

namespace {
void unlink_shm(const std::string& name) { std::remove(("/dev/shm" + name).c_str()); }
constexpr int kMaster = 0;
constexpr uint32_t kLayersPerRank = 16;  // each rank emits this many completions
}  // namespace

// Every rank owns a contiguous block of seqs:
//   rank r emits seqs [r*kLayersPerRank, (r+1)*kLayersPerRank).
// All blocks tile [0, world*kLayersPerRank) densely. The master router
// must inject exactly world*kLayersPerRank completions, in order.
TEST(LayerCompletionRoutingMPI, AllRanksRouteToMasterInOrder) {
    const auto& ctx = mh::DistributedContext::get_current_world();
    const int rank = *ctx->rank();
    const int world = *ctx->size();
    ASSERT_GE(world, 2) << "run with mpirun -np >=2";

    // Per-rank ring name so co-located ranks don't collide on /dev/shm.
    const std::string ring = "/tt_lcr_mpi_ring_" + std::to_string(rank);
    const std::string sched = "/tt_lcr_mpi_sched";
    unlink_shm(ring);
    if (rank == kMaster) {
        unlink_shm(sched);
    }
    ctx->barrier();  // master clears the shared sched name before anyone uses it

    LayerCompletionRouterConfig cfg;
    cfg.rank = rank;
    cfg.world_size = world;
    cfg.master_rank = kMaster;
    cfg.ring_shm_name = ring;
    cfg.scheduler_channel_shm_name = sched;
    auto router = std::make_unique<LayerCompletionRouter>(cfg);

    auto producer = LayerCompletionQueue::connect(ring, 10'000);

    // Master connects its scheduler-side counter channel consumer.
    std::unique_ptr<InterProcessCounterChannel> consumer;
    if (rank == kMaster) {
        consumer = InterProcessCounterChannel::connect(sched, 10'000);
    }

    ctx->barrier();  // routers + producers all up

    const uint64_t base = static_cast<uint64_t>(rank) * kLayersPerRank;
    for (uint32_t i = 0; i < kLayersPerRank; ++i) {
        LayerCompletionMessage m{base + i, static_cast<uint32_t>(rank), i, 0u, 0u};
        while (!producer->try_push(m)) {
            std::this_thread::yield();
        }
    }

    const uint64_t expected = static_cast<uint64_t>(world) * kLayersPerRank;
    uint64_t got = 0;
    if (rank == kMaster) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (got < expected && std::chrono::steady_clock::now() < deadline) {
            got += consumer->try_consume_all();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        EXPECT_EQ(got, expected) << "master did not receive all completions";
    }

    // Coordinated teardown: with end-of-stream sentinels the master waits for one sentinel per
    // subordinate, so stop() neither hangs nor drops the tail regardless of who stops first. Every
    // rank stops after a single barrier — no subordinate-before-master sequencing (which the old
    // cancel-based teardown required). A hang here would fail the test by timeout.
    ctx->barrier();
    router->stop();
    if (rank == kMaster) {
        // Sweep once more for anything injected during the master's final ring-drain inside stop();
        // coordinated teardown must not have lost a completion.
        got += consumer->try_consume_all();
        EXPECT_EQ(got, expected) << "master lost completions across coordinated teardown";
        consumer->shutdown();
    }
    producer->shutdown();
}

}  // namespace tt::tt_metal::distributed::test
