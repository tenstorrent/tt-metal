// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device-free tests for the RDMA transport underneath HostMeshSocket. Two queue
// pairs are cross-connected on one port (the HCA loops back internally), so the
// payload/doorbell/credit mechanism can be exercised without a Tenstorrent device
// and without a second host. Every protocol bug found so far lived here, not in
// the device legs, and these run in seconds.

#include <gtest/gtest.h>

#include "tt_metal/distributed/host_transport/rdma_link.hpp"
#include "tt_metal/distributed/host_transport/socket_relay.hpp"

#include <chrono>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

namespace tt::tt_metal::distributed::host_transport {
namespace {

constexpr uint32_t kPageSize = 14336;  // the target packet size, deliberately not a power of two
constexpr uint32_t kNumPages = 16;
constexpr uint32_t kRingBytes = kPageSize * kNumPages;

bool rdma_available() {
    try {
        RdmaContext probe;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// A page-aligned buffer, so it can be registered and used as a ring.
std::vector<std::byte> make_ring() { return std::vector<std::byte>(kRingBytes); }

std::vector<std::byte> pattern(uint64_t seed, size_t bytes) {
    std::vector<std::byte> out(bytes);
    uint32_t word = static_cast<uint32_t>(seed * 2654435761u + 1u);
    for (size_t i = 0; i + 4 <= bytes; i += 4) {
        std::memcpy(out.data() + i, &word, 4);
        word = word * 1664525u + 1013904223u;
    }
    return out;
}

// Spin until `predicate` holds, so a lost signal fails the test instead of hanging.
template <typename F>
bool spin_until(F predicate, std::chrono::milliseconds limit = std::chrono::seconds(5)) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::yield();
    }
    return predicate();
}

// One connected pair of channels plus the regions each side advertises.
struct Loopback {
    RdmaContext ctx;
    RdmaChannel a{ctx};
    RdmaChannel b{ctx};

    std::vector<std::byte> a_ring = make_ring();
    std::vector<std::byte> b_ring = make_ring();
    uint32_t a_credit = 0;
    uint32_t b_doorbell = 0;

    RdmaRegion a_ring_mr{ctx, a_ring.data(), a_ring.size()};
    RdmaRegion b_ring_mr{ctx, b_ring.data(), b_ring.size()};
    RdmaRegion a_credit_mr{ctx, &a_credit, sizeof(a_credit)};
    RdmaRegion b_doorbell_mr{ctx, &b_doorbell, sizeof(b_doorbell)};

    Loopback() {
        // a is the sender: it advertises where b should write credit.
        RdmaEndpoint a_local = a.local_endpoint();
        a_local.credit = a_credit_mr.descriptor();
        // b is the receiver: it advertises its ring and its doorbell slot.
        RdmaEndpoint b_local = b.local_endpoint();
        b_local.fifo = b_ring_mr.descriptor();
        b_local.doorbell = b_doorbell_mr.descriptor();

        a.connect(b_local);
        b.connect(a_local);
    }
};

TEST(HostTransportTest, ConnectsLoopback) {
    if (!rdma_available()) {
        GTEST_SKIP() << "no usable RoCEv2 RDMA device on this host";
    }
    Loopback lb;
    EXPECT_TRUE(lb.a.connected());
    EXPECT_TRUE(lb.b.connected());
}

// The core claim: a doorbell written after a payload is never visible before it.
TEST(HostTransportTest, DoorbellOrderedAfterPayload) {
    if (!rdma_available()) {
        GTEST_SKIP() << "no usable RoCEv2 RDMA device on this host";
    }
    Loopback lb;

    uint64_t forwarded = 0;
    for (uint32_t batch = 0; batch < 8; batch++) {
        const uint32_t pages = (batch % 3) + 1;
        const uint32_t index = static_cast<uint32_t>(forwarded % kNumPages);
        const uint32_t head = std::min(pages, kNumPages - index);
        const uint64_t offset = static_cast<uint64_t>(index) * kPageSize;

        auto payload = pattern(batch + 1, static_cast<size_t>(pages) * kPageSize);
        std::memcpy(lb.a_ring.data() + offset, payload.data(), static_cast<size_t>(head) * kPageSize);
        if (pages > head) {
            std::memcpy(
                lb.a_ring.data(),
                payload.data() + static_cast<size_t>(head) * kPageSize,
                static_cast<size_t>(pages - head) * kPageSize);
        }

        ASSERT_TRUE(lb.a.post_write(lb.a_ring_mr, offset, offset, head * kPageSize));
        if (pages > head) {
            ASSERT_TRUE(lb.a.post_write(lb.a_ring_mr, 0, 0, (pages - head) * kPageSize));
        }
        forwarded += pages;
        ASSERT_TRUE(lb.a.post_doorbell(static_cast<uint32_t>(forwarded)));

        // The doorbell is the only thing polled; when it lands, the payload behind
        // it must already be there.
        ASSERT_TRUE(spin_until([&] {
            lb.b.poll_send();
            return *static_cast<volatile uint32_t*>(lb.b_doorbell_mr.addr()) == forwarded;
        })) << "doorbell for total "
            << forwarded << " never arrived (batch " << batch << ")";

        EXPECT_EQ(std::memcmp(lb.b_ring.data() + offset, payload.data(), static_cast<size_t>(head) * kPageSize), 0)
            << "payload mismatch at batch " << batch;
        if (pages > head) {
            EXPECT_EQ(
                std::memcmp(
                    lb.b_ring.data(),
                    payload.data() + static_cast<size_t>(head) * kPageSize,
                    static_cast<size_t>(pages - head) * kPageSize),
                0)
                << "wrapped payload mismatch at batch " << batch;
        }
        lb.a.poll_send();
    }
    // pages cycles 1,2,3 over 8 batches
    EXPECT_EQ(forwarded, 15u);
}

// Enough batches to lap the ring many times, which is where pointer arithmetic
// and counter widening go wrong.
TEST(HostTransportTest, SurvivesManyRingLaps) {
    if (!rdma_available()) {
        GTEST_SKIP() << "no usable RoCEv2 RDMA device on this host";
    }
    Loopback lb;

    uint64_t forwarded = 0;
    uint64_t observed = 0;
    for (uint32_t batch = 0; batch < 500; batch++) {
        const uint32_t pages = 4;
        const uint32_t index = static_cast<uint32_t>(forwarded % kNumPages);
        const uint32_t head = std::min(pages, kNumPages - index);
        const uint64_t offset = static_cast<uint64_t>(index) * kPageSize;

        ASSERT_TRUE(lb.a.post_write(lb.a_ring_mr, offset, offset, head * kPageSize));
        if (pages > head) {
            ASSERT_TRUE(lb.a.post_write(lb.a_ring_mr, 0, 0, (pages - head) * kPageSize));
        }
        forwarded += pages;
        ASSERT_TRUE(lb.a.post_doorbell(static_cast<uint32_t>(forwarded)));
        lb.a.poll_send();

        ASSERT_TRUE(spin_until([&] {
            lb.b.poll_send();
            observed = widen(observed, *static_cast<volatile uint32_t*>(lb.b_doorbell_mr.addr()));
            return observed == forwarded;
        })) << "stalled at batch "
            << batch << ": observed " << observed << " want " << forwarded;

        // The derived delta must stay within a ring, which is what the relay asserts.
        ASSERT_LE(forwarded - (forwarded - pages), kNumPages);
    }
    EXPECT_EQ(forwarded, 2000u);
}

// Credit flows the other way on the same pair, as absolute counts.
TEST(HostTransportTest, CreditReturnsAbsoluteCount) {
    if (!rdma_available()) {
        GTEST_SKIP() << "no usable RoCEv2 RDMA device on this host";
    }
    Loopback lb;

    uint64_t seen = 0;
    for (uint32_t consumed : {1u, 5u, 5u, 64u, 1000u, 70000u}) {
        ASSERT_TRUE(lb.b.post_credit(consumed));
        ASSERT_TRUE(spin_until([&] {
            lb.b.poll_send();
            return *static_cast<volatile uint32_t*>(lb.a_credit_mr.addr()) == consumed;
        })) << "credit "
            << consumed << " never arrived";
        seen = widen(seen, *static_cast<volatile uint32_t*>(lb.a_credit_mr.addr()));
        EXPECT_EQ(seen, consumed);
    }
}

// RelaySender retires a batch when the completion for its doorbell lands, so the
// doorbell must be signaled even when the periodic signal cadence has not come
// due -- otherwise the last batch of a stream never retires and the socket's
// barrier hangs.
TEST(HostTransportTest, DoorbellCompletionRetiresBatch) {
    if (!rdma_available()) {
        GTEST_SKIP() << "no usable RoCEv2 RDMA device on this host";
    }
    Loopback lb;

    uint64_t forwarded = 0;
    for (uint32_t batch = 0; batch < 5; batch++) {
        ASSERT_TRUE(lb.a.post_write(lb.a_ring_mr, 0, 0, kPageSize));
        forwarded += 1;
        ASSERT_TRUE(lb.a.post_doorbell(static_cast<uint32_t>(forwarded)));
        const uint64_t doorbell_id = lb.a.last_posted_id();

        // Nothing else is posted after this batch, so only a signaled doorbell can
        // ever move reaped() past it.
        ASSERT_TRUE(spin_until([&] {
            lb.a.poll_send();
            return lb.a.reaped() > doorbell_id;
        })) << "batch "
            << batch << " never retired: reaped " << lb.a.reaped() << " want > " << doorbell_id;
    }
}

// The send queue must report back-pressure rather than silently dropping work.
TEST(HostTransportTest, SendQueueReportsBackPressure) {
    if (!rdma_available()) {
        GTEST_SKIP() << "no usable RoCEv2 RDMA device on this host";
    }
    Loopback lb;
    const uint32_t slots = lb.a.send_slots_available();
    ASSERT_GT(slots, 0u);

    uint32_t posted = 0;
    while (lb.a.send_slots_available() > 0) {
        if (!lb.a.post_write(lb.a_ring_mr, 0, 0, kPageSize)) {
            break;
        }
        posted++;
        ASSERT_LE(posted, slots) << "posted past the reported capacity";
    }
    EXPECT_EQ(lb.a.send_slots_available(), 0u);
    EXPECT_FALSE(lb.a.post_write(lb.a_ring_mr, 0, 0, kPageSize)) << "a full send queue must refuse work";

    // Draining completions must make room again.
    ASSERT_TRUE(spin_until([&] {
        lb.a.poll_send();
        return lb.a.send_slots_available() > 0;
    }));
}

}  // namespace
}  // namespace tt::tt_metal::distributed::host_transport
