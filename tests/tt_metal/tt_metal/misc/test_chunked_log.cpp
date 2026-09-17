// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <cstdint>
#include <latch>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "tt_metal/common/chunked_log.hpp"

namespace tt::tt_metal {
namespace {

struct Item {
    uint64_t index;
    uint64_t a, b;
    uint64_t check;
    static Item at(uint64_t i) {
        Item it{i, i * 0x9E3779B97F4A7C15ull, ~i, 0};
        it.check = it.index ^ it.a ^ it.b;
        return it;
    }
    bool consistent() const { return check == (index ^ a ^ b); }
};

constexpr uint64_t kChunk = ChunkedLog<Item>::kChunkItems;

TEST(ChunkedLog, CapacityIsWholeChunks) {
    EXPECT_EQ(ChunkedLog<Item>(0).capacity(), kChunk);
    EXPECT_EQ(ChunkedLog<Item>(kChunk - 1).capacity(), kChunk);
    EXPECT_EQ(ChunkedLog<Item>(3 * kChunk + 7).capacity(), 3 * kChunk);
}

TEST(ChunkedLog, RetainsTheNewestCapacityItems) {
    ChunkedLog<Item> log(2 * kChunk);
    const uint64_t n = 5 * kChunk + 3;
    for (uint64_t i = 0; i < n; i++) {
        log.push(Item::at(i));
    }
    EXPECT_EQ(log.count(), n);
    EXPECT_EQ(log.first(), 4 * kChunk);  // whole chunks retire: the newest 4099 items span two chunks
    Item it{};
    EXPECT_FALSE(log.read(log.first() - 1, it));
    EXPECT_FALSE(log.read(n, it));
    for (uint64_t i = log.first(); i < n; i += 97) {
        ASSERT_TRUE(log.read(i, it));
        EXPECT_EQ(it.index, i);
        EXPECT_TRUE(it.consistent());
    }
}

TEST(ChunkedLog, ClearRetiresEverythingAndKeepsIndicesGrowing) {
    ChunkedLog<Item> log(kChunk);
    for (uint64_t i = 0; i < 10; i++) {
        log.push(Item::at(i));
    }
    log.clear();
    EXPECT_EQ(log.first(), log.count());
    EXPECT_EQ(log.count(), kChunk);
    Item it{};
    EXPECT_FALSE(log.read(3, it));
    log.push(Item::at(kChunk));
    ASSERT_TRUE(log.read(kChunk, it));
    EXPECT_EQ(it.index, kChunk);
}

// The writer wraps a two-chunk log continuously while readers copy retained items at random: every successful
// read is an intact item at the index asked for.
TEST(ChunkedLog, ReadsRacingReuseAreIntactOrRejected) {
    ChunkedLog<Item> log(2 * kChunk);
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> good{0}, rejected{0}, bad{0};
    constexpr int kReaders = 4;
    std::latch start(kReaders + 1);
    std::vector<std::thread> readers;
    for (int r = 0; r < kReaders; r++) {
        readers.emplace_back([&, r] {
            std::mt19937_64 rng(r + 1);
            start.arrive_and_wait();
            Item it{};
            while (!stop.load(std::memory_order_relaxed)) {
                const uint64_t f = log.first(), n = log.count();
                if (n == f) {
                    continue;
                }
                const uint64_t i = f + rng() % (n - f);
                if (!log.read(i, it)) {
                    rejected++;
                    continue;
                }
                if (it.index == i && it.consistent()) {
                    good++;
                } else {
                    bad++;
                }
            }
        });
    }
    start.arrive_and_wait();
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    uint64_t i = 0;
    while (std::chrono::steady_clock::now() < until) {
        for (int k = 0; k < 1000; k++) {
            log.push(Item::at(i++));
        }
    }
    stop.store(true);
    for (auto& t : readers) {
        t.join();
    }
    EXPECT_EQ(bad.load(), 0u);
    EXPECT_GT(good.load(), 1000u);
    EXPECT_GT(i, 20 * kChunk) << "the writer wrapped the log many times";
    EXPECT_GT(rejected.load(), 0u) << "some reads overlapped a reuse or a retired index";
}

}  // namespace
}  // namespace tt::tt_metal
