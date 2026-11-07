// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-only tests for the layer-completion SHM ring: owner and connector are two LayerCompletionQueue objects
// in this process.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <internal/disaggregation/layer_completion_message.hpp>
#include <internal/disaggregation/layer_completion_queue.hpp>
#include "tt_metal/distributed/layer_completion/layer_completion_ring_layout.hpp"
#include <tt-metalium/experimental/sockets/named_shm.hpp>

namespace tt::tt_metal::internal {

namespace {

// `i` is carried in layer_start so FIFO order and delivery can be checked on pop.
LayerCompletionMessage make_msg(uint32_t i) {
    return LayerCompletionMessage{
        /*slot_id=*/3u, /*pos_start=*/5120u, /*pos_end=*/10240u, /*layer_start=*/i, /*layer_end=*/i + 1u, /*flags=*/0u};
}

void unlink_if_exists(const std::string& shm_name) { std::remove(("/dev/shm" + shm_name).c_str()); }

// Unique SHM names per case so parallel/repeated runs never collide.
std::string fresh_name(const char* tag) {
    static std::atomic<uint32_t> counter{0};
    const std::string name = fmt::format("/tt_lcq_test_{}_{}_{}", tag, ::getpid(), counter.fetch_add(1));
    unlink_if_exists(name);
    return name;
}

}  // namespace

TEST(LayerCompletionLayout, Sentinel) {
    LayerCompletionMessage m{};
    EXPECT_FALSE(is_layer_completion_sentinel(m));
    m.flags = kLayerCompletionSentinel;
    EXPECT_TRUE(is_layer_completion_sentinel(m));
    EXPECT_TRUE(is_layer_completion_sentinel(layer_completion_sentinel()));
}

TEST(LayerCompletionQueue, FifoRoundtrip) {
    const std::string name = fresh_name("fifo");
    auto owner = LayerCompletionQueue::create(name, 4);
    auto conn = LayerCompletionQueue::connect(name, 5'000);

    for (uint32_t i = 0; i < 8; ++i) {
        EXPECT_TRUE(owner->try_push(make_msg(i)));
    }
    LayerCompletionMessage out{};
    for (uint32_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(conn->try_pop(out));
        EXPECT_EQ(out.layer_start, i);  // FIFO
    }
    EXPECT_FALSE(conn->try_pop(out));  // empty
    owner->shutdown();
}

TEST(LayerCompletionQueue, RejectsPushWhenFull) {
    const std::string name = fresh_name("full");
    auto owner = LayerCompletionQueue::create(name, 4);
    for (uint32_t i = 0; i < LayerCompletionQueue::capacity(); ++i) {
        ASSERT_TRUE(owner->try_push(make_msg(i)));
    }
    EXPECT_FALSE(owner->try_push(make_msg(9999)));  // full → reject, no overwrite
    owner->shutdown();
}

TEST(LayerCompletionQueue, WrapsAroundPastCapacity) {
    const std::string name = fresh_name("wrap");
    auto owner = LayerCompletionQueue::create(name, 4);
    auto conn = LayerCompletionQueue::connect(name, 5'000);

    LayerCompletionMessage out{};
    const uint32_t total = LayerCompletionQueue::capacity() * 3 + 7;
    uint32_t pushed = 0, popped = 0;
    while (popped < total) {
        while (pushed < total && owner->try_push(make_msg(pushed))) {
            ++pushed;
        }
        while (conn->try_pop(out)) {
            EXPECT_EQ(out.layer_start, popped);
            ++popped;
        }
    }
    EXPECT_EQ(popped, total);
    owner->shutdown();
}

// Multiple producer threads, single consumer — the prefill topology.
TEST(LayerCompletionQueue, MpscProducersAllDelivered) {
    const std::string name = fresh_name("mpsc");
    auto owner = LayerCompletionQueue::create(name, 4);
    auto conn = LayerCompletionQueue::connect(name, 5'000);

    constexpr uint32_t kProducers = 4;
    constexpr uint32_t kPerProducer = 500;
    constexpr uint32_t kTotal = kProducers * kPerProducer;

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (uint32_t p = 0; p < kProducers; ++p) {
        producers.emplace_back([&owner, p] {
            for (uint32_t i = 0; i < kPerProducer; ++i) {
                while (!owner->try_push(make_msg(p * kPerProducer + i))) {
                    std::this_thread::yield();
                }
            }
        });
    }

    std::vector<char> seen(kTotal, 0);
    LayerCompletionMessage out{};
    uint32_t popped = 0;
    while (popped < kTotal) {
        if (conn->try_pop(out)) {
            ASSERT_LT(out.layer_start, kTotal);
            seen[out.layer_start] = 1;
            ++popped;
        } else {
            std::this_thread::yield();
        }
    }
    for (auto& t : producers) {
        t.join();
    }
    for (uint32_t i = 0; i < kTotal; ++i) {
        EXPECT_TRUE(seen[i]) << "message " << i << " lost";
    }
    owner->shutdown();
}

TEST(LayerCompletionQueue, AllFieldsRoundTrip) {
    const std::string name = fresh_name("fields");
    auto owner = LayerCompletionQueue::create(name, 4);
    auto conn = LayerCompletionQueue::connect(name, 5'000);

    const LayerCompletionMessage in{
        /*slot_id=*/5u,
        /*pos_start=*/5120u,
        /*pos_end=*/10213u,
        /*layer_start=*/14u,
        /*layer_end=*/15u,
        /*flags=*/0u,
        /*host_timestamp_ns=*/1'760'000'000'123'456'789ull};
    ASSERT_TRUE(owner->try_push(in));

    LayerCompletionMessage out{};
    ASSERT_TRUE(conn->try_pop(out));
    EXPECT_EQ(out.slot_id, in.slot_id);
    EXPECT_EQ(out.pos_start, in.pos_start);
    EXPECT_EQ(out.pos_end, in.pos_end);
    EXPECT_EQ(out.layer_start, in.layer_start);
    EXPECT_EQ(out.layer_end, in.layer_end);
    EXPECT_EQ(out.flags, in.flags);
    EXPECT_EQ(out.host_timestamp_ns, in.host_timestamp_ns);
    owner->shutdown();
}

TEST(LayerCompletionQueue, PublishesAcksPerChunk) {
    const std::string name = fresh_name("acks");
    auto owner = LayerCompletionQueue::create(name, 61);
    auto conn = LayerCompletionQueue::connect(name, 5'000);
    EXPECT_EQ(owner->completions_per_chunk(), 61u);
    EXPECT_EQ(conn->completions_per_chunk(), 61u);
    EXPECT_THROW(LayerCompletionQueue::create(fresh_name("zero"), 0), std::invalid_argument);
    owner->shutdown();
}

// Another kind of segment at the name (short or unmagicked) is refused inside the grace, not after the budget.
TEST(LayerCompletionConnect, RefusesAForeignSegmentInsideTheBudget) {
    const std::string name = fmt::format("/lcq_foreign_{}", ::getpid());
    unlink_if_exists(name);
    auto foreign = tt::tt_metal::distributed::NamedShm::create(name, 128);
    const auto t0 = std::chrono::steady_clock::now();
    try {
        LayerCompletionQueue::connect(name, 60'000);
        FAIL() << "connect accepted a 128-byte segment";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("not a valid"), std::string::npos) << e.what();
    }
    EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(10));
    unlink_if_exists(name);
}

}  // namespace tt::tt_metal::internal
