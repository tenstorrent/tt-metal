// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// MPI test for LayerCompletionRouter with participants that are a non-contiguous subset of the world ranks: the
// stage leaders of a pipeline whose middle stage spans two hosts. Host-only; launch with `mpirun -n 4`.

#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/distributed_context.hpp>

#include <internal/disaggregation/layer_completion_message.hpp>
#include <internal/disaggregation/layer_completion_queue.hpp>
#include <internal/disaggregation/layer_completion_router.hpp>

namespace tt::tt_metal::internal {

namespace {

namespace mh = tt::tt_metal::distributed::multihost;

constexpr int kWorldSize = 4;
constexpr int kMasterRank = 0;
constexpr uint32_t kMessagesPerLeader = 20;
constexpr uint32_t kCompletionsPerChunk = 61;

void unlink_if_exists(const std::string& shm_name) { std::remove(("/dev/shm" + shm_name).c_str()); }

template <typename F>
bool wait_until(const F& f, int timeout_ms = 20'000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (f()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return f();
}

// pos_start carries the sender's world rank, layer_start the per-sender sequence number.
LayerCompletionMessage make_msg(uint32_t sender_rank, uint32_t seq) {
    return LayerCompletionMessage{
        /*slot_id=*/2u, /*pos_start=*/sender_rank, /*pos_end=*/sender_rank + 50u, seq, seq + 1u, /*flags=*/0u};
}

}  // namespace

// World 4, leaders {0, 1, 3}, master 0: every leader's messages reach the scheduler ring in per-sender order and
// rank 2, which runs no router, takes no part.
TEST(LayerCompletionRouterMpi, NonContiguousParticipants) {
    const mh::ContextPtr& world = mh::DistributedContext::get_current_world();
    ASSERT_EQ(*world->size(), kWorldSize) << "launch with mpirun -n " << kWorldSize;
    const int rank = *world->rank();
    const std::vector<int> leaders = {0, 1, 3};
    const bool is_leader = std::find(leaders.begin(), leaders.end(), rank) != leaders.end();

    const std::string tag = std::to_string(::getppid());
    const std::string ring = "/tt_lcr_mpi_ring_" + tag + "_" + std::to_string(rank);
    const std::string sched = "/tt_lcr_mpi_sched_" + tag;

    if (is_leader) {
        unlink_if_exists(ring);
        if (rank == kMasterRank) {
            unlink_if_exists(sched);
        }
        LayerCompletionRouterConfig cfg;
        cfg.rank = rank;
        cfg.world_size = kWorldSize;
        cfg.master_rank = kMasterRank;
        cfg.participant_ranks = leaders;
        cfg.ring_shm_name = ring;
        cfg.scheduler_shm_name = rank == kMasterRank ? sched : "";
        cfg.completions_per_chunk = kCompletionsPerChunk;
        cfg.teardown_timeout_ms = 20'000;
        auto router = std::make_unique<LayerCompletionRouter>(std::move(cfg));
        EXPECT_EQ(router->is_master(), rank == kMasterRank);

        auto producer = LayerCompletionQueue::connect(ring, 5'000);
        for (uint32_t i = 0; i < kMessagesPerLeader; ++i) {
            ASSERT_TRUE(producer->try_push(make_msg(static_cast<uint32_t>(rank), i)));
        }

        if (rank == kMasterRank) {
            auto scheduler = LayerCompletionQueue::connect(sched, 5'000);
            std::map<uint32_t, std::vector<uint32_t>> seen;
            const std::size_t want = leaders.size() * kMessagesPerLeader;
            std::size_t got = 0;
            EXPECT_TRUE(wait_until([&scheduler, &seen, &got, want] {
                LayerCompletionMessage out{};
                while (scheduler->try_pop(out)) {
                    seen[out.pos_start].push_back(out.layer_start);
                    ++got;
                }
                return got >= want;
            })) << "received " << got << " of " << want;
            EXPECT_EQ(seen.size(), leaders.size());
            for (int leader : leaders) {
                const auto& seqs = seen[static_cast<uint32_t>(leader)];
                ASSERT_EQ(seqs.size(), kMessagesPerLeader) << "leader " << leader;
                for (uint32_t i = 0; i < kMessagesPerLeader; ++i) {
                    EXPECT_EQ(seqs[i], i) << "leader " << leader;
                }
            }
            EXPECT_EQ(seen.count(2u), 0u);
            scheduler->shutdown();
        }
        producer->shutdown();
        router->stop();
    }
    world->barrier();
}

}  // namespace tt::tt_metal::internal

int main(int argc, char** argv) {
    tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
