// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Host-only tests for the layer-completion SHM ring, covering both protocol
// versions: v1 (LayerCompletionMessage, 24B packed cells, magic 'LCQ1') and
// v2 (LayerCompletionMessageV2, 40B self-describing messages, cache-line
// cells, magic 'LCQ2'). No device or MPI needed — owner and connector are
// two LayerCompletionQueueT objects in this process sharing /dev/shm.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <internal/disaggregation/layer_completion_message.hpp>
#include <internal/disaggregation/layer_completion_queue.hpp>
#include "tt_metal/distributed/layer_completion/layer_completion_ring_layout.hpp"

namespace tt::tt_metal::internal {

namespace {

template <typename MsgT>
MsgT make_msg(uint64_t seq);

template <>
LayerCompletionMessage make_msg<LayerCompletionMessage>(uint64_t seq) {
    return LayerCompletionMessage{
        seq,
        /*source_rank=*/1u,
        /*layer_idx=*/static_cast<uint32_t>(seq % 61),
        /*request_id=*/static_cast<uint32_t>(seq / 61),
        /*reserved=*/0u};
}

template <>
LayerCompletionMessageV2 make_msg<LayerCompletionMessageV2>(uint64_t seq) {
    return LayerCompletionMessageV2{
        seq,
        /*source_rank=*/1u,
        /*request_id=*/static_cast<uint32_t>(seq / 61),
        /*slot_id=*/3u,
        /*pos_start=*/5120u,
        /*pos_end=*/10240u,
        /*layer_start=*/static_cast<uint32_t>(seq % 61),
        /*layer_end=*/static_cast<uint32_t>(seq % 61) + 1u,
        /*flags=*/0u};
}

void unlink_if_exists(const std::string& shm_name) { std::remove(("/dev/shm" + shm_name).c_str()); }

}  // namespace

// ---------------------------------------------------------------------------
// Wire geometry (the compile-time contract lives beside the layouts in the
// headers; these EXPECTs surface a drift as a test failure too).
// ---------------------------------------------------------------------------

TEST(LayerCompletionLayout, V1GeometryIsFrozen) {
    EXPECT_EQ(sizeof(LayerCompletionMessage), 24u);
    EXPECT_EQ(alignof(LayerCompletionMessage), 8u);
    EXPECT_TRUE(std::is_trivially_copyable_v<LayerCompletionMessage>);
    EXPECT_EQ(sizeof(LayerCompletionCell), 32u);  // packed, two cells per cache line
    EXPECT_EQ(alignof(LayerCompletionCell), 8u);
    EXPECT_EQ(layer_completion_cells_offset<LayerCompletionMessage>(), 128u);
    EXPECT_EQ(kLayerCompletionRingBytes<LayerCompletionMessage>, 32896u);
    EXPECT_EQ(LayerCompletionRingTraits<LayerCompletionMessage>::magic, 0x4C435131u);
}

TEST(LayerCompletionLayout, V2Geometry) {
    EXPECT_EQ(sizeof(LayerCompletionMessageV2), 40u);
    EXPECT_EQ(alignof(LayerCompletionMessageV2), 8u);
    EXPECT_TRUE(std::is_trivially_copyable_v<LayerCompletionMessageV2>);
    // One cache line per cell — a packed 48B cell would straddle lines.
    EXPECT_EQ(sizeof(LayerCompletionCellV2), kLayerCompletionCacheLine);
    EXPECT_EQ(alignof(LayerCompletionCellV2), kLayerCompletionCacheLine);
    EXPECT_EQ(layer_completion_cells_offset<LayerCompletionMessageV2>() % kLayerCompletionCacheLine, 0u);
    EXPECT_EQ(kLayerCompletionRingBytes<LayerCompletionMessageV2> % kLayerCompletionCacheLine, 0u);
    EXPECT_EQ(LayerCompletionRingTraits<LayerCompletionMessageV2>::magic, 0x4C435132u);
    // The version key is what keeps a cross-protocol attach from corrupting.
    EXPECT_NE(
        LayerCompletionRingTraits<LayerCompletionMessage>::magic,
        LayerCompletionRingTraits<LayerCompletionMessageV2>::magic);
}

TEST(LayerCompletionLayout, HeaderIsSharedAcrossVersions) {
    EXPECT_EQ(offsetof(LayerCompletionRingHeader, enqueue_pos), 0u);
    EXPECT_EQ(offsetof(LayerCompletionRingHeader, dequeue_pos), kLayerCompletionCacheLine);
    EXPECT_EQ(sizeof(LayerCompletionRingHeader), 2 * kLayerCompletionCacheLine);
}

TEST(LayerCompletionLayout, CapacityIsPowerOfTwo) {
    EXPECT_NE(kLayerCompletionRingCapacity, 0u);
    EXPECT_EQ(kLayerCompletionRingCapacity & (kLayerCompletionRingCapacity - 1), 0u);
    EXPECT_EQ(kLayerCompletionRingMask, kLayerCompletionRingCapacity - 1);
}

TEST(LayerCompletionLayout, Sentinels) {
    LayerCompletionMessage v1{};
    LayerCompletionMessageV2 v2{};
    EXPECT_FALSE(is_layer_completion_sentinel(v1));
    EXPECT_FALSE(is_layer_completion_sentinel(v2));
    v1.reserved = kLayerCompletionSentinel;
    v2.flags = kLayerCompletionSentinel;
    EXPECT_TRUE(is_layer_completion_sentinel(v1));
    EXPECT_TRUE(is_layer_completion_sentinel(v2));
}

// ---------------------------------------------------------------------------
// Ring behaviour, both instantiations
// ---------------------------------------------------------------------------

template <typename MsgT>
class LayerCompletionRingTest : public ::testing::Test {
protected:
    using Queue = LayerCompletionQueueT<MsgT>;

    void SetUp() override { ring_names_.clear(); }

    // Unique SHM names per case so parallel/repeated runs never collide.
    std::string fresh_name(const char* tag) {
        std::string name = fmt::format("/tt_lcq_test_{}_{}_{}", tag, ::getpid(), ring_names_.size());
        ring_names_.push_back(name);
        return name;
    }

private:
    std::vector<std::string> ring_names_;
};

using RingTypes = ::testing::Types<LayerCompletionMessage, LayerCompletionMessageV2>;
TYPED_TEST_SUITE(LayerCompletionRingTest, RingTypes);

TYPED_TEST(LayerCompletionRingTest, FifoRoundtrip) {
    using MsgT = TypeParam;
    using Queue = LayerCompletionQueueT<MsgT>;
    const std::string name = this->fresh_name("fifo");
    unlink_if_exists(name);
    auto owner = Queue::create(name);
    auto conn = Queue::connect(name, 5'000);

    for (uint64_t i = 0; i < 8; ++i) {
        EXPECT_TRUE(owner->try_push(make_msg<MsgT>(i)));
    }
    MsgT out{};
    for (uint64_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(conn->try_pop(out));
        EXPECT_EQ(out.seq, i);  // FIFO
    }
    EXPECT_FALSE(conn->try_pop(out));  // empty
    owner->shutdown();
}

TYPED_TEST(LayerCompletionRingTest, RejectsPushWhenFull) {
    using MsgT = TypeParam;
    using Queue = LayerCompletionQueueT<MsgT>;
    const std::string name = this->fresh_name("full");
    unlink_if_exists(name);
    auto owner = Queue::create(name);
    for (uint32_t i = 0; i < Queue::capacity(); ++i) {
        ASSERT_TRUE(owner->try_push(make_msg<MsgT>(i)));
    }
    EXPECT_FALSE(owner->try_push(make_msg<MsgT>(9999)));  // full → reject, no overwrite
    owner->shutdown();
}

TYPED_TEST(LayerCompletionRingTest, WrapsAroundPastCapacity) {
    using MsgT = TypeParam;
    using Queue = LayerCompletionQueueT<MsgT>;
    const std::string name = this->fresh_name("wrap");
    unlink_if_exists(name);
    auto owner = Queue::create(name);
    auto conn = Queue::connect(name, 5'000);

    MsgT out{};
    const uint64_t total = static_cast<uint64_t>(Queue::capacity()) * 3 + 7;
    uint64_t pushed = 0, popped = 0;
    while (popped < total) {
        while (pushed < total && owner->try_push(make_msg<MsgT>(pushed))) {
            ++pushed;
        }
        while (conn->try_pop(out)) {
            EXPECT_EQ(out.seq, popped);
            ++popped;
        }
    }
    EXPECT_EQ(popped, total);
    owner->shutdown();
}

// Multiple producer threads, single consumer — the prefill topology.
TYPED_TEST(LayerCompletionRingTest, MpscProducersAllDelivered) {
    using MsgT = TypeParam;
    using Queue = LayerCompletionQueueT<MsgT>;
    const std::string name = this->fresh_name("mpsc");
    unlink_if_exists(name);
    auto owner = Queue::create(name);
    auto conn = Queue::connect(name, 5'000);

    constexpr uint32_t kProducers = 4;
    constexpr uint64_t kPerProducer = 500;
    constexpr uint64_t kTotal = kProducers * kPerProducer;

    std::vector<std::thread> producers;
    for (uint32_t p = 0; p < kProducers; ++p) {
        producers.emplace_back([&owner, p] {
            for (uint64_t i = 0; i < kPerProducer; ++i) {
                const uint64_t seq = p * kPerProducer + i;
                while (!owner->try_push(make_msg<MsgT>(seq))) {
                    std::this_thread::yield();
                }
            }
        });
    }

    std::vector<char> seen(kTotal, 0);
    MsgT out{};
    uint64_t popped = 0;
    while (popped < kTotal) {
        if (conn->try_pop(out)) {
            ASSERT_LT(out.seq, kTotal);
            seen[out.seq] = 1;
            ++popped;
        } else {
            std::this_thread::yield();
        }
    }
    for (auto& t : producers) {
        t.join();
    }
    for (uint64_t i = 0; i < kTotal; ++i) {
        EXPECT_TRUE(seen[i]) << "seq " << i << " lost";
    }
    owner->shutdown();
}

// ---------------------------------------------------------------------------
// V2-specific: full field round-trip, and the cross-version guard.
// ---------------------------------------------------------------------------

TEST(LayerCompletionQueueV2, AllFieldsRoundTrip) {
    const std::string name = "/tt_lcq_test_v2_fields";
    unlink_if_exists(name);
    auto owner = LayerCompletionQueueV2::create(name);
    auto conn = LayerCompletionQueueV2::connect(name, 5'000);

    const LayerCompletionMessageV2 in{
        /*seq=*/7u * 61 + 14,
        /*source_rank=*/2u,
        /*request_id=*/7u,
        /*slot_id=*/5u,
        /*pos_start=*/5120u,
        /*pos_end=*/10213u,
        /*layer_start=*/14u,
        /*layer_end=*/15u,
        /*flags=*/0u};
    ASSERT_TRUE(owner->try_push(in));

    LayerCompletionMessageV2 out{};
    ASSERT_TRUE(conn->try_pop(out));
    EXPECT_EQ(out.seq, in.seq);
    EXPECT_EQ(out.source_rank, in.source_rank);
    EXPECT_EQ(out.request_id, in.request_id);
    EXPECT_EQ(out.slot_id, in.slot_id);
    EXPECT_EQ(out.pos_start, in.pos_start);
    EXPECT_EQ(out.pos_end, in.pos_end);
    EXPECT_EQ(out.layer_start, in.layer_start);
    EXPECT_EQ(out.layer_end, in.layer_end);
    EXPECT_EQ(out.flags, in.flags);
    owner->shutdown();
}

// A range message (stage-level completion) travels as one slot, unsplit.
TEST(LayerCompletionQueueV2, RangeMessageRoundTrip) {
    const std::string name = "/tt_lcq_test_v2_range";
    unlink_if_exists(name);
    auto owner = LayerCompletionQueueV2::create(name);
    auto conn = LayerCompletionQueueV2::connect(name, 5'000);

    ASSERT_TRUE(owner->try_push(LayerCompletionMessageV2{
        /*seq=*/3u * 61,
        /*source_rank=*/0u,
        /*request_id=*/3u,
        /*slot_id=*/1u,
        /*pos_start=*/0u,
        /*pos_end=*/5120u,
        /*layer_start=*/0u,
        /*layer_end=*/14u,  // 14 layers, one message
        /*flags=*/0u}));

    LayerCompletionMessageV2 out{};
    ASSERT_TRUE(conn->try_pop(out));
    EXPECT_EQ(out.layer_end - out.layer_start, 14u);
    EXPECT_FALSE(conn->try_pop(out));
    owner->shutdown();
}

TEST(LayerCompletionQueue, CrossVersionConnectFails) {
    const std::string name = "/tt_lcq_test_xver";
    unlink_if_exists(name);
    auto owner = LayerCompletionQueue::create(name);  // v1 segment ('LCQ1')
    // A v2 connector must be rejected by the magic check, not map and corrupt.
    EXPECT_THROW(LayerCompletionQueueV2::connect(name, 5'000), std::runtime_error);
    owner->shutdown();

    unlink_if_exists(name);
    auto owner2 = LayerCompletionQueueV2::create(name);  // v2 segment ('LCQ2')
    EXPECT_THROW(LayerCompletionQueue::connect(name, 5'000), std::runtime_error);
    owner2->shutdown();
}

}  // namespace tt::tt_metal::internal
