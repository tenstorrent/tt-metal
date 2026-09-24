// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Two-rank MPI test for LayerCompletionRouter v2 subordinate forwarding: exercises
// the typed MPI payload path (40B LayerCompletionMessageV2) and sentinel teardown
// that world_size=1 host-only tests cannot reach.
//
// Launch (from repo root, MPI distributed build):
//   mpirun -n 2 ./build/test/tt_metal/distributed/layer_completion_router_multihost_test \
//     --gtest_filter="LayerCompletionRouterMultihost.V2SubordinateForwardsToMaster"

#include <gtest/gtest.h>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/distributed_context.hpp>

#include <internal/disaggregation/layer_completion_message.hpp>
#include <internal/disaggregation/layer_completion_queue.hpp>
#include <internal/disaggregation/layer_completion_router.hpp>
#include "tests/tt_metal/multihost/common/multihost_test_tools.hpp"

namespace tt::tt_metal::internal {
namespace {

using mh = tt::tt_metal::distributed::multihost;

void unlink_if_exists(const std::string& shm_name) { std::remove(("/dev/shm" + shm_name).c_str()); }

std::string fresh_name(const char* tag) {
    static std::atomic<uint32_t> counter{0};
    return "/tt_lcr_mh_" + std::string(tag) + "_" + std::to_string(::getpid()) + "_" +
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

LayerCompletionMessageV2 v2_msg(uint32_t source_rank, uint64_t seq, uint32_t request_id, uint32_t layer_start, uint32_t layer_end) {
    return LayerCompletionMessageV2{
        seq,
        source_rank,
        request_id,
        /*slot_id=*/2u,
        /*pos_start=*/100u * request_id,
        /*pos_end=*/100u * request_id + 50u,
        layer_start,
        layer_end,
        /*flags=*/0u};
}

}  // namespace

TEST(LayerCompletionRouterMultihost, V2SubordinateForwardsToMaster) {
    const auto& world = mh::DistributedContext::get_current_world();
    ASSERT_EQ(*world->size(), 2) << "This test requires exactly 2 MPI ranks";
    const int rank = static_cast<int>(*world->rank());
    constexpr int master_rank = 0;

    const std::string ring = fresh_name("ring") + "_" + std::to_string(rank);
    const std::string sched = fresh_name("sched");
    unlink_if_exists(ring);
    if (rank == master_rank) {
        unlink_if_exists(sched);
    }
    world->barrier();

    LayerCompletionRouterConfig cfg;
    cfg.rank = rank;
    cfg.world_size = 2;
    cfg.master_rank = master_rank;
    cfg.ring_shm_name = ring;
    cfg.protocol = LayerCompletionProtocol::kStructuredV2;
    cfg.scheduler_shm_name = (rank == master_rank) ? sched : "";
    cfg.teardown_timeout_ms = 5'000;
    auto router = std::make_unique<LayerCompletionRouter>(std::move(cfg));
    auto producer = LayerCompletionQueueV2::connect(ring, 5'000);
    world->barrier();

    if (rank == master_rank) {
        auto scheduler = LayerCompletionQueueV2::connect(sched, 5'000);
        // Subordinate pushes first (barrier below), then master pushes locally.
        world->barrier();
        ASSERT_TRUE(producer->try_push(v2_msg(/*source_rank=*/0, /*seq=*/1, /*request_id=*/0, 0, 1)));

        std::vector<uint32_t> seen_request_ids;
        ASSERT_TRUE(wait_until([&] {
            LayerCompletionMessageV2 out{};
            if (!scheduler->try_pop(out)) {
                return false;
            }
            seen_request_ids.push_back(out.request_id);
            return seen_request_ids.size() == 2;
        }));
        EXPECT_EQ(seen_request_ids[0], 1u);
        EXPECT_EQ(seen_request_ids[1], 0u);
        EXPECT_EQ(router->processed(), 2u);

        producer->shutdown();
        scheduler->shutdown();
    } else {
        ASSERT_TRUE(producer->try_push(v2_msg(/*source_rank=*/1, /*seq=*/0, /*request_id=*/1, 5, 6)));
        producer->shutdown();
        world->barrier();
        ASSERT_TRUE(wait_until([&] { return router->processed() == 1; }));
    }

    router->stop();
    world->barrier();
}

}  // namespace tt::tt_metal::internal

int main(int argc, char** argv) { return multihost::common::multihost_main(argc, argv); }
