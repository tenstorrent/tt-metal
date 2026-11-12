// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests for the D2D host-mode pumps (no device): ordering, back-pressure, stop behaviour and failure
// propagation, over the in-process loopback channel.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include <tt-metalium/distributed_context.hpp>

#include "ttnn/tensor/d2d_host_pump.hpp"

namespace ttnn::d2d_host {
namespace {

using namespace std::chrono_literals;

constexpr size_t kPartWords = 1024;
constexpr size_t kNumParts = 3;

template <typename Pred>
bool eventually(Pred&& pred, std::chrono::milliseconds timeout = 10s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!pred()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(100us);
    }
    return true;
}

// Transfer n of a stream starting at `base` holds iota(base + n) across all parts, part p continuing where
// part p-1 ended.
void fill(std::vector<std::vector<uint32_t>>& buffers, uint32_t first) {
    uint32_t next = first;
    for (auto& b : buffers) {
        std::iota(b.begin(), b.end(), next);
        next += static_cast<uint32_t>(b.size());
    }
}

std::vector<Part> parts_of(std::vector<std::vector<uint32_t>>& buffers) {
    std::vector<Part> parts;
    for (auto& b : buffers) {
        parts.push_back(Part{reinterpret_cast<std::byte*>(b.data()), b.size() * sizeof(uint32_t)});
    }
    return parts;
}

// Fake sender device: the producer makes transfers available; the pump reads them.
struct FakeSender {
    std::vector<std::vector<uint32_t>> buffers = std::vector<std::vector<uint32_t>>(
        kNumParts, std::vector<uint32_t>(kPartWords));
    std::atomic<uint32_t> produced{0};
    std::atomic<uint32_t> read{0};

    SenderOps ops(uint32_t base) {
        SenderOps o;
        o.parts = parts_of(buffers);
        o.ready = [this] { return produced.load() > read.load(); };
        o.read = [this, base] {
            fill(buffers, base + read.load());
            read.fetch_add(1);
        };
        return o;
    }
};

struct FakeReceiver {
    std::vector<std::vector<uint32_t>> buffers = std::vector<std::vector<uint32_t>>(
        kNumParts, std::vector<uint32_t>(kPartWords));
    std::atomic<uint32_t> written{0};
    std::atomic<uint32_t> device_free{0};  // transfers the device is willing to take
    std::atomic<bool> content_ok{true};

    ReceiverOps ops(uint32_t base) {
        ReceiverOps o;
        o.parts = parts_of(buffers);
        o.write = [this, base] {
            // The device takes a transfer only when its consumer is done with the previous one.
            while (device_free.load() <= written.load()) {
                std::this_thread::sleep_for(50us);
            }
            std::vector<std::vector<uint32_t>> expect = buffers;
            fill(expect, base + written.load());
            if (expect != buffers) {
                content_ok = false;
            }
            written.fetch_add(1);
        };
        return o;
    }
};

TEST(D2DHostPump, LoopbackDeliversTransfersInOrder) {
    auto [tx, rx] = make_loopback_channels();
    FakeSender sender_dev;
    FakeReceiver receiver_dev;
    receiver_dev.device_free = 1000;
    SenderPump sender(sender_dev.ops(1000), std::move(tx));
    ReceiverPump receiver(receiver_dev.ops(1000), std::move(rx));

    constexpr uint32_t kTransfers = 50;
    for (uint32_t n = 0; n < kTransfers; ++n) {
        sender_dev.produced.fetch_add(1);
        ASSERT_TRUE(eventually([&receiver_dev, n] { return receiver_dev.written.load() == n + 1; }));
    }
    EXPECT_TRUE(receiver_dev.content_ok.load());
    EXPECT_EQ(sender.stats().transfers, kTransfers);
    EXPECT_EQ(receiver.stats().transfers, kTransfers);
    EXPECT_EQ(sender.stats().bytes, kTransfers * kNumParts * kPartWords * sizeof(uint32_t));
    sender.rethrow_if_failed();
    receiver.rethrow_if_failed();
}

// A transfer is not written to the receiver device before the device is ready for it.
TEST(D2DHostPump, ReceiverWaitsForDevice) {
    auto [tx, rx] = make_loopback_channels();
    FakeSender sender_dev;
    FakeReceiver receiver_dev;
    SenderPump sender(sender_dev.ops(0), std::move(tx));
    ReceiverPump receiver(receiver_dev.ops(0), std::move(rx));

    sender_dev.produced.fetch_add(2);
    std::this_thread::sleep_for(100ms);
    EXPECT_EQ(receiver_dev.written.load(), 0u);

    receiver_dev.device_free = 1;
    ASSERT_TRUE(eventually([&receiver_dev] { return receiver_dev.written.load() == 1; }));
    std::this_thread::sleep_for(100ms);
    EXPECT_EQ(receiver_dev.written.load(), 1u);

    receiver_dev.device_free = 2;
    ASSERT_TRUE(eventually([&receiver_dev] { return receiver_dev.written.load() == 2; }));
    EXPECT_TRUE(receiver_dev.content_ok.load());
}

TEST(D2DHostPump, DestructionWhileIdleDoesNotHang) {
    auto [tx, rx] = make_loopback_channels();
    FakeSender sender_dev;
    FakeReceiver receiver_dev;
    const auto start = std::chrono::steady_clock::now();
    {
        SenderPump sender(sender_dev.ops(0), std::move(tx));
        ReceiverPump receiver(receiver_dev.ops(0), std::move(rx));
        std::this_thread::sleep_for(20ms);
    }
    EXPECT_LT(std::chrono::steady_clock::now() - start, 5s);
}

TEST(D2DHostPump, CallbackFailureIsReported) {
    auto [tx, rx] = make_loopback_channels();
    FakeSender sender_dev;
    auto ops = sender_dev.ops(0);
    ops.read = [] { throw std::runtime_error("device read failed"); };
    SenderPump sender(std::move(ops), std::move(tx));
    sender_dev.produced.fetch_add(1);
    ASSERT_TRUE(eventually([&sender] { return sender.failed(); }));
    EXPECT_THROW(sender.rethrow_if_failed(), std::runtime_error);
}

TEST(D2DHostPump, LostTransferIsDetected) {
    auto [tx, rx] = make_loopback_channels();
    FakeReceiver receiver_dev;
    receiver_dev.device_free = 10;
    ReceiverPump receiver(receiver_dev.ops(0), std::move(rx));
    // Sequence number 7 arrives while the receiver expects 0.
    FakeSender sender_dev;
    std::atomic<bool> stop{false};
    ASSERT_TRUE(tx->send_transfer(7, parts_of(sender_dev.buffers), stop));
    ASSERT_TRUE(eventually([&receiver] { return receiver.failed(); }));
    EXPECT_ANY_THROW(receiver.rethrow_if_failed());
}

TEST(D2DHostPump, PartSizeMismatchIsDetected) {
    auto [tx, rx] = make_loopback_channels();
    FakeReceiver receiver_dev;
    receiver_dev.device_free = 10;
    ReceiverPump receiver(receiver_dev.ops(0), std::move(rx));
    std::vector<std::vector<uint32_t>> wrong(kNumParts, std::vector<uint32_t>(kPartWords / 2));
    std::atomic<bool> stop{false};
    ASSERT_TRUE(tx->send_transfer(0, parts_of(wrong), stop));
    ASSERT_TRUE(eventually([&receiver] { return receiver.failed(); }));
    EXPECT_ANY_THROW(receiver.rethrow_if_failed());
}

}  // namespace
}  // namespace ttnn::d2d_host
