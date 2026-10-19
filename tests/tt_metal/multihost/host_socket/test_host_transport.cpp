// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The HostTransport contract the relay depends on, checked without a device:
// streams pages between two ranks and verifies every byte. Needs two ranks.

#include <gtest/gtest.h>

#include "tt_metal/distributed/host_transport/host_transport.hpp"

#include <tt-metalium/distributed_context.hpp>

#include "host_socket_test_utils.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace tt::tt_metal::distributed::host_transport {
namespace {

constexpr uint32_t kParityPageSize = 14336;  // the target packet size, deliberately not a power of two
constexpr uint32_t kParityNumPages = 16;

// Distinct per test, so a cancelled receive or an in-flight credit from an
// earlier test can never match a later one's.
int next_tag_base() {
    static int next = 4096;
    next += kTagsPerConnection;
    return next;
}

void fill_page(std::byte* page, uint64_t index) {
    uint32_t word = static_cast<uint32_t>(index * 2654435761u + 1u);
    for (uint32_t i = 0; i + 4 <= kParityPageSize; i += 4) {
        std::memcpy(page + i, &word, 4);
        word = word * 1664525u + 1013904223u;
    }
}

// Both ranks must agree, or one skips while the other waits on a handshake.
bool all_ranks_usable(const multihost::DistributedContext& ctx) {
    uint8_t mine = host_transport_available() ? 1 : 0;
    std::vector<uint8_t> all(static_cast<size_t>(*ctx.size()), 0);
    ctx.all_gather(
        std::span<std::byte>(reinterpret_cast<std::byte*>(&mine), 1),
        std::span<std::byte>(reinterpret_cast<std::byte*>(all.data()), all.size()));
    for (uint8_t v : all) {
        if (v == 0) {
            return false;
        }
    }
    return true;
}

class HostTransportTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_ = multihost::DistributedContext::get_current_world();
        if (*ctx_->size() < 2) {
            GTEST_SKIP() << "needs two ranks";
        }
        if (!all_ranks_usable(*ctx_)) {
            GTEST_SKIP() << "host transport unavailable on at least one rank";
        }
    }

    std::unique_ptr<HostTransport> make(bool is_sender, int tag_base, uint32_t page_size = kParityPageSize) {
        return make_host_transport(TransportParams{
            .geometry = {.page_size = page_size, .num_pages = kParityNumPages},
            .is_sender = is_sender,
            .ring = ring_.data(),
            .peer_rank = *ctx_->rank() == 0 ? 1 : 0,
            .context = ctx_,
            .tag_base = tag_base,
        });
    }

    // Host-to-host only: one page out, wait for the peer's consumed credit back.
    // That is the transport's put-to-credit round trip with no device on either
    // end, timed on the sender's clock. The receiver credits the instant it sees
    // the page, so there is no consumer in the loop.
    void pingpong(uint32_t page_size, uint32_t iterations) {
        auto transport = make(sender(), next_tag_base(), page_size);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        const uint32_t laps = iterations + host_socket_test::kWarmupIters;
        std::vector<double> rtt_us;
        rtt_us.reserve(iterations);

        if (sender()) {
            for (uint32_t i = 0; i < laps; i++) {
                const uint64_t want = i + 1;
                // Reuse of a slot needs the transport to be done reading it;
                // sixteen laps back, so this never actually waits.
                while (transport->pages_released() + kParityNumPages < want) {
                    ASSERT_LT(std::chrono::steady_clock::now(), deadline) << transport->describe();
                    transport->poll();
                }
                const auto start = std::chrono::steady_clock::now();
                while (!transport->can_send(1) || !transport->send(i, 1)) {
                    ASSERT_LT(std::chrono::steady_clock::now(), deadline) << transport->describe();
                    transport->poll();
                }
                while (transport->peer_consumed_pages() < want) {
                    ASSERT_LT(std::chrono::steady_clock::now(), deadline) << transport->describe();
                    transport->poll();
                }
                if (i >= host_socket_test::kWarmupIters) {
                    rtt_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
                }
            }
            // Let the transport retire its last sends before teardown.
            while (transport->pages_released() < laps && std::chrono::steady_clock::now() < deadline) {
                transport->poll();
            }
        } else {
            for (uint32_t i = 0; i < laps; i++) {
                const uint64_t want = i + 1;
                while (transport->pages_delivered() < want) {
                    ASSERT_LT(std::chrono::steady_clock::now(), deadline) << transport->describe();
                    transport->poll();
                }
                transport->set_consumed(want);
                while (!transport->post_credit(want)) {
                    ASSERT_LT(std::chrono::steady_clock::now(), deadline) << transport->describe();
                    transport->poll();
                }
            }
            // The last credit must leave before the transport is torn down.
            while (!transport->post_credit(laps) && std::chrono::steady_clock::now() < deadline) {
                transport->poll();
            }
            transport->poll();
        }
        ctx_->barrier();

        if (sender()) {
            const auto rtt = host_socket_test::summarize(rtt_us);
            GTEST_LOG_(INFO) << "host-to-host put-to-credit round trip @" << page_size << " B: min " << rtt.min
                             << " p50 " << rtt.p50 << " avg " << rtt.avg << " p99 " << rtt.p99 << " max " << rtt.max
                             << " us (" << rtt.count << " samples) via " << transport->describe();
            host_socket_test::record_latency("h2h_put_to_credit_rtt", page_size, rtt);
        }
    }

    bool sender() const { return *ctx_->rank() == 0; }

    // Streams `total` pages in batches of `batch` and verifies every byte. The
    // sender stands in for a device producing pages, the receiver for one
    // consuming them; `consume_every` pages the receiver holds before crediting,
    // which is how back-pressure gets exercised.
    void stream(uint32_t total, uint32_t batch, uint32_t consume_every = 1) {
        auto transport = make(sender(), next_tag_base());
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);

        if (sender()) {
            uint64_t produced = 0;
            while (transport->pages_released() < total) {
                ASSERT_LT(std::chrono::steady_clock::now(), deadline) << transport->describe();
                transport->poll();
                const uint32_t want = std::min(batch, total - static_cast<uint32_t>(produced));
                if (want == 0 || !transport->can_send(want)) {
                    continue;
                }
                // Never overwrite a page the transport still owns, nor one the
                // peer's consumer has not finished with.
                if (produced + want - transport->pages_released() > kParityNumPages ||
                    produced + want - transport->peer_consumed_pages() > kParityNumPages) {
                    continue;
                }
                for (uint32_t i = 0; i < want; i++) {
                    fill_page(ring_.data() + (produced + i) % kParityNumPages * kParityPageSize, produced + i);
                }
                ASSERT_TRUE(transport->send(static_cast<uint32_t>(produced), want)) << transport->describe();
                produced += want;
            }
            EXPECT_EQ(produced, total);
        } else {
            std::vector<std::byte> expected(kParityPageSize);
            uint64_t consumed = 0;
            while (consumed < total) {
                ASSERT_LT(std::chrono::steady_clock::now(), deadline) << transport->describe();
                transport->poll();
                const uint64_t delivered = transport->pages_delivered();
                ASSERT_LE(delivered - consumed, kParityNumPages) << "peer ran past a ring: " << transport->describe();
                while (consumed < delivered) {
                    fill_page(expected.data(), consumed);
                    ASSERT_EQ(
                        std::memcmp(
                            ring_.data() + consumed % kParityNumPages * kParityPageSize,
                            expected.data(),
                            kParityPageSize),
                        0)
                        << "page " << consumed << " mismatched";
                    consumed++;
                    if (consumed % consume_every == 0 || consumed == total) {
                        transport->set_consumed(consumed);
                        transport->post_credit(consumed);
                    }
                }
            }
            // The sender waits on credit for its last batch.
            while (std::chrono::steady_clock::now() < deadline && !transport->post_credit(consumed)) {
                transport->poll();
            }
            transport->poll();
        }
        ctx_->barrier();
    }

    std::shared_ptr<multihost::DistributedContext> ctx_;
    std::vector<std::byte> ring_ = std::vector<std::byte>(static_cast<size_t>(kParityPageSize) * kParityNumPages);
};

// 200 pages over a 16-page ring in batches of 3: the batch size does not divide
// the ring, so batches straddle the wrap at a shifting offset.
TEST_F(HostTransportTest, StreamsAcrossRingLaps) { stream(/*total=*/200, /*batch=*/3); }

TEST_F(HostTransportTest, StreamsOnePageAtATime) { stream(/*total=*/64, /*batch=*/1); }

// The receiver credits only every 8th page, so the sender spends most of the run
// blocked on credit rather than on the transport.
TEST_F(HostTransportTest, HoldsUnderCreditBackPressure) { stream(/*total=*/96, /*batch=*/4, /*consume_every=*/8); }

// Host-to-host alone, so the device legs can be subtracted from the end-to-end
// round trip. Sweeps the page size; the ring is sized for the largest.
TEST_F(HostTransportTest, PingPongLatency) {
    const uint32_t iterations = static_cast<uint32_t>(host_socket_test::env_or("TT_HOST_SOCKET_LAT_ITERS", 200));
    for (uint32_t page_size : {64u, 1024u, 4096u, 14336u, 65536u}) {
        if (page_size > kParityPageSize) {
            ring_.resize(static_cast<size_t>(page_size) * kParityNumPages);
        }
        pingpong(page_size, iterations);
        if (HasFatalFailure()) {
            return;
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::host_transport
