// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <latch>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "tt_metal/common/broadcast_ring.hpp"

namespace tt::tt_metal {
namespace {

using tt::tt_metal::experimental::ProgramRealtimeRecord;

constexpr uint32_t kBatchSeed = 0x9E3779B9u;

ProgramRealtimeRecord make_record(uint32_t runtime_id, std::span<const std::string_view> sources = {}) {
    return ProgramRealtimeRecord{
        .runtime_id = runtime_id,
        .chip_id = 7,
        .start_timestamp = 100 + runtime_id,
        .end_timestamp = 200 + runtime_id,
        .frequency = 1.5,
        .kernel_sources = sources,
    };
}

TEST(BroadcastRing, CapacityRoundsUpToPowerOfTwo) {
    EXPECT_EQ(BroadcastRing<uint32_t>(0).capacity(), 1u);
    EXPECT_EQ(BroadcastRing<uint32_t>(1).capacity(), 1u);
    EXPECT_EQ(BroadcastRing<uint32_t>(3).capacity(), 4u);
    EXPECT_EQ(BroadcastRing<uint32_t>(4).capacity(), 4u);
    EXPECT_EQ(BroadcastRing<uint32_t>(1000).capacity(), 1024u);
}

TEST(BroadcastRing, ReaderStartsAtTailAndSeesSubsequentItems) {
    BroadcastRing<uint32_t> ring(4);
    ring.writer().publish(1);
    auto reader = ring.make_reader();

    uint32_t value = 0;
    EXPECT_FALSE(reader.read(value));

    ring.writer().publish(2);
    ASSERT_TRUE(reader.read(value));
    EXPECT_EQ(value, 2u);
    EXPECT_FALSE(reader.read(value));
}

TEST(BroadcastRing, ReadBatchIsOrderedAndBoundedByOutput) {
    BroadcastRing<uint32_t> ring(8);
    auto reader = ring.make_reader();

    uint32_t items[] = {1, 2, 3, 4, 5};
    ring.writer().publish_batch(items);

    uint32_t none[1] = {};
    EXPECT_TRUE(reader.read_batch(std::span<uint32_t>(none, 0)).empty());

    uint32_t out2[2] = {};
    auto first = reader.read_batch(out2);
    ASSERT_EQ(first.size(), 2u);
    EXPECT_EQ(first[0], 1u);
    EXPECT_EQ(first[1], 2u);

    uint32_t out8[8] = {};
    auto rest = reader.read_batch(out8);
    ASSERT_EQ(rest.size(), 3u);
    EXPECT_EQ(rest[0], 3u);
    EXPECT_EQ(rest[1], 4u);
    EXPECT_EQ(rest[2], 5u);

    EXPECT_EQ(reader.dropped(), 0u);
    EXPECT_TRUE(reader.read_batch(out8).empty());
}

TEST(BroadcastRing, RecordsBroadcastToAllReadersWithSpanFieldsIntact) {
    static_assert(std::is_trivially_copyable_v<ProgramRealtimeRecord>);
    EXPECT_TRUE(BroadcastRing<ProgramRealtimeRecord>::is_always_lock_free);

    std::string_view sources[] = {"kernel_a.cpp", "kernel_b.cpp"};
    BroadcastRing<ProgramRealtimeRecord> ring(8);
    auto reader_a = ring.make_reader();
    auto reader_b = ring.make_reader();

    ProgramRealtimeRecord records[] = {make_record(1, sources), make_record(2, sources), make_record(3, sources)};
    ring.writer().publish_batch(records);

    ProgramRealtimeRecord out_a[3];
    ProgramRealtimeRecord out_b[3];
    auto batch_a = reader_a.read_batch(out_a);
    auto batch_b = reader_b.read_batch(out_b);

    ASSERT_EQ(batch_a.size(), 3u);
    ASSERT_EQ(batch_b.size(), 3u);
    for (size_t i = 0; i < batch_a.size(); ++i) {
        for (const auto* record : {&batch_a[i], &batch_b[i]}) {
            EXPECT_EQ(record->runtime_id, i + 1);
            EXPECT_EQ(record->chip_id, 7u);
            EXPECT_EQ(record->start_timestamp, 100 + record->runtime_id);
            EXPECT_EQ(record->end_timestamp, 200 + record->runtime_id);
            EXPECT_DOUBLE_EQ(record->frequency, 1.5);
            ASSERT_EQ(record->kernel_sources.size(), 2u);
            EXPECT_EQ(record->kernel_sources[0], "kernel_a.cpp");
            EXPECT_EQ(record->kernel_sources[1], "kernel_b.cpp");
        }
    }
    EXPECT_EQ(reader_a.dropped(), 0u);
    EXPECT_EQ(reader_b.dropped(), 0u);
}

TEST(BroadcastRing, NonTrivialTypeUsesLockedSlotWithMoveAndDrop) {
    struct NonTrivialRecord {
        uint32_t runtime_id = 0;
        std::string name;
    };
    static_assert(!std::is_trivially_copyable_v<NonTrivialRecord>);
    static_assert(!BroadcastRing<NonTrivialRecord>::is_always_lock_free);

    BroadcastRing<NonTrivialRecord> ring(4);
    auto reader = ring.make_reader();

    std::vector<NonTrivialRecord> records;
    records.reserve(7);
    for (uint32_t i = 0; i < 7; ++i) {
        records.push_back({.runtime_id = i, .name = std::string(256, static_cast<char>('a' + i))});
    }
    ring.writer().publish_batch_move(std::span<NonTrivialRecord>(records));

    NonTrivialRecord out[4];
    auto batch = reader.read_batch(out);

    ASSERT_EQ(batch.size(), 4u);
    EXPECT_EQ(reader.dropped(), 3u);
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(batch[i].runtime_id, i + 3);
        EXPECT_EQ(batch[i].name, std::string(256, static_cast<char>('a' + i + 3)));
    }
}

TEST(BroadcastRing, OversizedSinglePublishRetainsLastCapacity) {
    BroadcastRing<uint32_t> ring(4);
    auto reader = ring.make_reader();

    uint32_t items[] = {0, 1, 2, 3, 4, 5};
    ring.writer().publish_batch(items);

    uint32_t out[4] = {};
    auto batch = reader.read_batch(out);

    ASSERT_EQ(batch.size(), 4u);
    EXPECT_EQ(reader.dropped(), 2u);
    EXPECT_EQ(batch[0], 2u);
    EXPECT_EQ(batch[1], 3u);
    EXPECT_EQ(batch[2], 4u);
    EXPECT_EQ(batch[3], 5u);
}

TEST(BroadcastRing, ReadersLagAndDropIndependently) {
    BroadcastRing<uint32_t> ring(4);
    auto fast_reader = ring.make_reader();
    auto slow_reader = ring.make_reader();

    uint32_t first_batch[] = {1, 2};
    ring.writer().publish_batch(first_batch);

    uint32_t out[4] = {};
    ASSERT_EQ(fast_reader.read_batch(out).size(), 2u);
    EXPECT_EQ(out[0], 1u);
    EXPECT_EQ(out[1], 2u);
    EXPECT_EQ(fast_reader.dropped(), 0u);

    uint32_t second_batch[] = {3, 4, 5, 6};
    ring.writer().publish_batch(second_batch);

    auto fast_batch = fast_reader.read_batch(out);
    ASSERT_EQ(fast_batch.size(), 4u);
    EXPECT_EQ(fast_reader.dropped(), 0u);
    EXPECT_EQ(fast_batch.front(), 3u);
    EXPECT_EQ(fast_batch.back(), 6u);

    auto slow_batch = slow_reader.read_batch(out);
    ASSERT_EQ(slow_batch.size(), 4u);
    EXPECT_EQ(slow_reader.dropped(), 2u);
    EXPECT_EQ(slow_batch.front(), 3u);
    EXPECT_EQ(slow_batch.back(), 6u);
}

TEST(BroadcastRing, DrainingBacklogDropsOnlyCapacityOverflow) {
    constexpr uint32_t kCapacity = 64;

    {
        BroadcastRing<uint32_t> ring(kCapacity);
        auto small_reader = ring.make_reader();
        auto full_reader = ring.make_reader();
        uint32_t items[kCapacity];
        for (uint32_t i = 0; i < kCapacity; ++i) {
            items[i] = i;
        }
        ring.writer().publish_batch(items);

        uint32_t small_out[8] = {};
        std::vector<uint32_t> got;
        while (true) {
            auto batch = small_reader.read_batch(small_out);
            if (batch.empty()) {
                break;
            }
            got.insert(got.end(), batch.begin(), batch.end());
        }
        EXPECT_EQ(small_reader.dropped(), 0u);
        ASSERT_EQ(got.size(), static_cast<size_t>(kCapacity));
        for (uint32_t i = 0; i < kCapacity; ++i) {
            EXPECT_EQ(got[i], i) << "small buffer, index " << i;
        }

        uint32_t full_out[kCapacity] = {};
        auto batch = full_reader.read_batch(full_out);
        EXPECT_EQ(full_reader.dropped(), 0u);
        ASSERT_EQ(batch.size(), static_cast<size_t>(kCapacity));
        for (uint32_t i = 0; i < kCapacity; ++i) {
            EXPECT_EQ(batch[i], i) << "full buffer, index " << i;
        }
    }

    {
        constexpr uint32_t kTotal = kCapacity + kCapacity / 2;
        constexpr uint32_t kBatch = kTotal / 2;
        BroadcastRing<uint32_t> ring(kCapacity);
        auto reader = ring.make_reader();
        for (uint32_t base : {0u, kBatch}) {
            uint32_t items[kBatch];
            for (uint32_t i = 0; i < kBatch; ++i) {
                items[i] = base + i;
            }
            ring.writer().publish_batch(items);
        }

        uint32_t out[8] = {};
        std::vector<uint32_t> got;
        while (true) {
            auto batch = reader.read_batch(out);
            if (batch.empty()) {
                break;
            }
            got.insert(got.end(), batch.begin(), batch.end());
        }
        EXPECT_EQ(reader.dropped(), kTotal - kCapacity);
        ASSERT_EQ(got.size(), static_cast<size_t>(kCapacity));
        for (uint32_t i = 0; i < kCapacity; ++i) {
            EXPECT_EQ(got[i], (kTotal - kCapacity) + i) << "at index " << i;
        }
    }
}

TEST(BroadcastRing, WaitWakesOnPublishThenWake) {
    BroadcastRing<uint32_t> ring(4);
    auto reader = ring.make_reader();
    std::latch waiting{1};
    std::atomic<bool> received{false};

    std::thread reader_thread([&]() {
        uint32_t out = 0;
        EXPECT_FALSE(reader.read(out));
        const auto token = reader.wait_token();
        waiting.count_down();
        reader.wait(token);
        received.store(reader.read(out) && out == 42, std::memory_order_release);
    });

    waiting.wait();
    ring.writer().publish(42);
    ring.writer().wake_readers();

    reader_thread.join();
    EXPECT_TRUE(received.load(std::memory_order_acquire));
}

TEST(BroadcastRing, WaitWakesOnWakeWithoutData) {
    BroadcastRing<uint32_t> ring(4);
    auto reader = ring.make_reader();
    std::latch waiting{1};
    std::atomic<bool> woke{false};

    std::thread reader_thread([&]() {
        uint32_t out = 0;
        EXPECT_FALSE(reader.read(out));
        const auto token = reader.wait_token();
        waiting.count_down();
        reader.wait(token);
        EXPECT_FALSE(reader.read(out));
        woke.store(true, std::memory_order_release);
    });

    waiting.wait();
    ring.writer().wake_readers();

    reader_thread.join();
    EXPECT_TRUE(woke.load(std::memory_order_acquire));
}

TEST(BroadcastRing, WakeBetweenTokenAndWaitIsNotLost) {
    BroadcastRing<uint32_t> ring(4);
    auto reader = ring.make_reader();

    const auto token = reader.wait_token();
    ring.writer().wake_readers();
    reader.wait(token);

    uint32_t out = 0;
    EXPECT_FALSE(reader.read(out));
}

TEST(BroadcastRing, ReaderIsMovable) {
    BroadcastRing<uint32_t> ring(8);
    auto reader = ring.make_reader();
    ring.writer().publish(10);
    ring.writer().publish(20);

    uint32_t value = 0;
    auto moved = std::move(reader);
    ASSERT_TRUE(moved.read(value));
    EXPECT_EQ(value, 10u);

    auto target = ring.make_reader();
    target = std::move(moved);
    ASSERT_TRUE(target.read(value));
    EXPECT_EQ(value, 20u);
    EXPECT_FALSE(target.read(value));
}

ProgramRealtimeRecord make_seq_record(uint32_t seq) {
    return ProgramRealtimeRecord{
        .runtime_id = seq + 1,
        .chip_id = (seq % 31) + 1,
        .start_timestamp = static_cast<uint64_t>(seq) * 7 + 3,
        .end_timestamp = static_cast<uint64_t>(seq) * 7 + 9,
        .frequency = 1.0 + static_cast<double>(seq) * 0.001,
        .kernel_sources = {},
    };
}

void verify_seq_records(const std::vector<ProgramRealtimeRecord>& received, uint64_t dropped, uint64_t total) {
    EXPECT_EQ(received.size() + dropped, total);
    uint32_t prev_seq = 0;
    bool has_prev = false;
    for (const auto& r : received) {
        const uint32_t seq = r.runtime_id - 1;
        ASSERT_EQ(r.chip_id, (seq % 31) + 1) << "torn/stale record: chip_id mismatch at seq " << seq;
        ASSERT_EQ(r.start_timestamp, static_cast<uint64_t>(seq) * 7 + 3)
            << "torn/stale record: start_timestamp mismatch at seq " << seq;
        ASSERT_EQ(r.end_timestamp, static_cast<uint64_t>(seq) * 7 + 9)
            << "torn/stale record: end_timestamp mismatch at seq " << seq;
        ASSERT_DOUBLE_EQ(r.frequency, 1.0 + static_cast<double>(seq) * 0.001)
            << "torn/stale record: frequency mismatch at seq " << seq;
        if (has_prev) {
            ASSERT_GT(seq, prev_seq) << "non-monotonic or duplicate delivery at seq " << seq;
        }
        prev_seq = seq;
        has_prev = true;
    }
}

TEST(BroadcastRing, ConcurrentIntegrityUnderDropPressure) {
    constexpr uint64_t kTotalRecords = 1u << 20;
    constexpr size_t kCapacity = 512;
    constexpr size_t kReadBatchSize = 64;
    constexpr size_t kMaxPublishBatchSize = 200;

    BroadcastRing<ProgramRealtimeRecord> ring(kCapacity);
    auto fast_reader = ring.make_reader();
    auto slow_reader = ring.make_reader();

    struct ReaderResult {
        std::vector<ProgramRealtimeRecord> received;
        uint64_t dropped = 0;
    };
    ReaderResult fast_result;
    ReaderResult slow_result;
    fast_result.received.reserve(kTotalRecords / 4);
    slow_result.received.reserve(kTotalRecords / 4);

    std::latch ready{3};
    std::latch start{1};
    std::atomic<bool> writer_done{false};

    auto run_reader = [&](BroadcastRing<ProgramRealtimeRecord>::Reader& reader, ReaderResult& result, bool throttle) {
        std::array<ProgramRealtimeRecord, kReadBatchSize> out;
        uint32_t reads = 0;
        ready.count_down();
        start.wait();
        while (true) {
            const bool done = writer_done.load(std::memory_order_acquire);
            const uint64_t dropped_before = reader.dropped();
            auto batch = reader.read_batch(std::span<ProgramRealtimeRecord>(out));
            if (batch.empty()) {
                if (reader.dropped() != dropped_before) {
                    continue;
                }
                if (done) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }
            result.received.insert(result.received.end(), batch.begin(), batch.end());
            if (throttle && (++reads % 4 == 0)) {
                std::this_thread::sleep_for(std::chrono::microseconds(40));
            }
        }
        result.dropped = reader.dropped();
    };

    std::thread fast_thread(run_reader, std::ref(fast_reader), std::ref(fast_result), false);
    std::thread slow_thread(run_reader, std::ref(slow_reader), std::ref(slow_result), true);
    std::thread writer_thread([&]() {
        ready.count_down();
        start.wait();
        std::array<ProgramRealtimeRecord, kMaxPublishBatchSize> records;
        std::mt19937 rng(kBatchSeed);
        std::uniform_int_distribution<size_t> pick_batch(1, kMaxPublishBatchSize);
        uint64_t next = 0;
        while (next < kTotalRecords) {
            const uint64_t base = next;
            const size_t batch_size = std::min<uint64_t>(pick_batch(rng), kTotalRecords - base);
            for (size_t k = 0; k < batch_size; ++k) {
                records[k] = make_seq_record(static_cast<uint32_t>(base + k));
            }
            ring.writer().publish_batch(std::span<const ProgramRealtimeRecord>(records.data(), batch_size));
            next += batch_size;
            if ((next & 0x3fffu) == 0) {
                std::this_thread::yield();
            }
        }
        writer_done.store(true, std::memory_order_release);
    });

    ready.wait();
    start.count_down();
    writer_thread.join();
    fast_thread.join();
    slow_thread.join();

    verify_seq_records(fast_result.received, fast_result.dropped, kTotalRecords);
    verify_seq_records(slow_result.received, slow_result.dropped, kTotalRecords);
    EXPECT_GT(fast_result.received.size(), 0u);
    EXPECT_GT(slow_result.received.size(), 0u);
}

bool seq_record_ok(const ProgramRealtimeRecord& r) {
    const uint32_t seq = r.runtime_id - 1;
    return r.chip_id == (seq % 31) + 1 && r.start_timestamp == static_cast<uint64_t>(seq) * 7 + 3 &&
           r.end_timestamp == static_cast<uint64_t>(seq) * 7 + 9 &&
           r.frequency == 1.0 + static_cast<double>(seq) * 0.001;
}

TEST(BroadcastRing, ConcurrentManyReadersMixedRates) {
    constexpr uint64_t kTotalRecords = 1u << 20;
    constexpr size_t kCapacity = 1024;
    constexpr size_t kReadBatchSize = 128;
    constexpr uint32_t kNumReaders = 8;
    constexpr size_t kMaxPublishBatchSize = 300;

    BroadcastRing<ProgramRealtimeRecord> ring(kCapacity);

    struct ReaderCheck {
        uint64_t received = 0;
        uint64_t dropped = 0;
        uint32_t last_seq = 0;
        bool has_last = false;
        bool fields_ok = true;
        bool ordered = true;
    };
    std::vector<BroadcastRing<ProgramRealtimeRecord>::Reader> readers;
    std::vector<ReaderCheck> checks(kNumReaders);
    readers.reserve(kNumReaders);
    for (uint32_t i = 0; i < kNumReaders; ++i) {
        readers.push_back(ring.make_reader());
    }

    std::latch ready{kNumReaders + 1};
    std::latch start{1};
    std::atomic<bool> writer_done{false};

    auto run_reader = [&](uint32_t idx) {
        std::array<ProgramRealtimeRecord, kReadBatchSize> out;
        ReaderCheck& c = checks[idx];
        ready.count_down();
        start.wait();
        while (true) {
            const bool done = writer_done.load(std::memory_order_acquire);
            const uint64_t dropped_before = readers[idx].dropped();
            auto batch = readers[idx].read_batch(std::span<ProgramRealtimeRecord>(out));
            if (batch.empty()) {
                if (readers[idx].dropped() != dropped_before) {
                    continue;
                }
                if (done) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }
            for (const auto& r : batch) {
                const uint32_t seq = r.runtime_id - 1;
                c.fields_ok &= seq_record_ok(r);
                c.ordered &= !c.has_last || seq > c.last_seq;
                c.last_seq = seq;
                c.has_last = true;
                ++c.received;
            }
            if (idx != 0 && (c.received % (64u * idx)) == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(idx * 10));
            }
        }
        c.dropped = readers[idx].dropped();
    };

    std::vector<std::thread> reader_threads;
    reader_threads.reserve(kNumReaders);
    for (uint32_t i = 0; i < kNumReaders; ++i) {
        reader_threads.emplace_back(run_reader, i);
    }
    std::thread writer_thread([&]() {
        ready.count_down();
        start.wait();
        std::array<ProgramRealtimeRecord, kMaxPublishBatchSize> records;
        std::mt19937 rng(kBatchSeed);
        std::uniform_int_distribution<size_t> pick_batch(1, kMaxPublishBatchSize);
        uint64_t next = 0;
        while (next < kTotalRecords) {
            const uint64_t base = next;
            const size_t batch_size = std::min<uint64_t>(pick_batch(rng), kTotalRecords - base);
            for (size_t k = 0; k < batch_size; ++k) {
                records[k] = make_seq_record(static_cast<uint32_t>(base + k));
            }
            ring.writer().publish_batch(std::span<const ProgramRealtimeRecord>(records.data(), batch_size));
            next += batch_size;
        }
        writer_done.store(true, std::memory_order_release);
    });

    ready.wait();
    start.count_down();
    writer_thread.join();
    for (auto& t : reader_threads) {
        t.join();
    }

    for (uint32_t i = 0; i < kNumReaders; ++i) {
        EXPECT_TRUE(checks[i].fields_ok) << "reader " << i << " saw a torn/stale record";
        EXPECT_TRUE(checks[i].ordered) << "reader " << i << " saw out-of-order / duplicate delivery";
        EXPECT_EQ(checks[i].received + checks[i].dropped, kTotalRecords) << "reader " << i << " lost records";
    }
}

TEST(BroadcastRing, ConcurrentBlockingWaitDeliversEveryItem) {
    constexpr uint64_t kTotalRecords = 1u << 18;
    constexpr size_t kCapacity = 4096;
    constexpr size_t kReadBatchSize = 256;
    constexpr uint32_t kNumReaders = 4;
    constexpr size_t kMaxPublishBatchSize = 64;

    BroadcastRing<ProgramRealtimeRecord> ring(kCapacity);

    struct ReaderResult {
        std::vector<ProgramRealtimeRecord> received;
        uint64_t dropped = 0;
    };
    std::vector<BroadcastRing<ProgramRealtimeRecord>::Reader> readers;
    std::vector<ReaderResult> results(kNumReaders);
    std::vector<std::atomic<uint64_t>> consumed(kNumReaders);
    readers.reserve(kNumReaders);
    for (uint32_t i = 0; i < kNumReaders; ++i) {
        readers.push_back(ring.make_reader());
        results[i].received.reserve(kTotalRecords);
    }

    std::latch ready{kNumReaders + 1};
    std::latch start{1};
    std::atomic<bool> stop{false};

    auto run_reader = [&](uint32_t idx) {
        std::array<ProgramRealtimeRecord, kReadBatchSize> out;
        ready.count_down();
        start.wait();
        uint64_t got = 0;
        while (true) {
            const auto token = readers[idx].wait_token();
            const bool stopping = stop.load(std::memory_order_acquire);
            auto batch = readers[idx].read_batch(std::span<ProgramRealtimeRecord>(out));
            if (!batch.empty()) {
                results[idx].received.insert(results[idx].received.end(), batch.begin(), batch.end());
                got += batch.size();
                consumed[idx].store(got, std::memory_order_release);
                continue;
            }
            if (stopping) {
                break;
            }
            readers[idx].wait(token);
        }
        results[idx].dropped = readers[idx].dropped();
    };

    std::vector<std::thread> reader_threads;
    reader_threads.reserve(kNumReaders);
    for (uint32_t i = 0; i < kNumReaders; ++i) {
        reader_threads.emplace_back(run_reader, i);
    }
    std::thread writer_thread([&]() {
        ready.count_down();
        start.wait();
        std::array<ProgramRealtimeRecord, kMaxPublishBatchSize> records;
        std::mt19937 rng(kBatchSeed);
        std::uniform_int_distribution<size_t> pick_batch(1, kMaxPublishBatchSize);
        uint64_t next = 0;
        while (next < kTotalRecords) {
            const uint64_t base = next;
            const size_t batch_size = std::min<uint64_t>(pick_batch(rng), kTotalRecords - base);
            while (true) {
                uint64_t slowest = next;
                for (auto& c : consumed) {
                    slowest = std::min(slowest, c.load(std::memory_order_acquire));
                }
                if (next + batch_size - slowest <= kCapacity / 2) {
                    break;
                }
                std::this_thread::yield();
            }
            for (size_t k = 0; k < batch_size; ++k) {
                records[k] = make_seq_record(static_cast<uint32_t>(base + k));
            }
            ring.writer().publish_batch(std::span<const ProgramRealtimeRecord>(records.data(), batch_size));
            ring.writer().wake_readers();
            next += batch_size;
        }
    });

    ready.wait();
    start.count_down();
    writer_thread.join();
    stop.store(true, std::memory_order_release);
    ring.writer().wake_readers();
    for (auto& t : reader_threads) {
        t.join();
    }

    for (uint32_t i = 0; i < kNumReaders; ++i) {
        EXPECT_EQ(results[i].dropped, 0u) << "reader " << i << " dropped under a bounded wait/wake load";
        verify_seq_records(results[i].received, results[i].dropped, kTotalRecords);
    }
}

TEST(BroadcastRing, ConcurrentReaderChurnUnderLiveWriter) {
    constexpr size_t kCapacity = 512;
    constexpr size_t kReadBatchSize = 64;
    constexpr size_t kMaxPublishBatchSize = 128;
    constexpr uint64_t kLongReaderTarget = 1u << 16;

    BroadcastRing<ProgramRealtimeRecord> ring(kCapacity);
    auto long_reader = ring.make_reader();

    std::latch ready{3};
    std::latch start{1};
    std::atomic<bool> long_done{false};
    std::atomic<uint64_t> long_consumed{0};

    auto long_result = std::vector<ProgramRealtimeRecord>{};
    long_result.reserve(kLongReaderTarget + kReadBatchSize);
    std::thread long_thread([&]() {
        std::array<ProgramRealtimeRecord, kReadBatchSize> out;
        ready.count_down();
        start.wait();
        while (long_result.size() < kLongReaderTarget) {
            auto batch = long_reader.read_batch(std::span<ProgramRealtimeRecord>(out));
            long_result.insert(long_result.end(), batch.begin(), batch.end());
            long_consumed.store(long_result.size(), std::memory_order_release);
        }
        long_done.store(true, std::memory_order_release);
    });

    std::atomic<bool> churn_fields_ok{true};
    std::atomic<bool> churn_ordered{true};
    std::thread churn_thread([&]() {
        std::array<ProgramRealtimeRecord, kReadBatchSize> out;
        ready.count_down();
        start.wait();
        while (!long_done.load(std::memory_order_acquire)) {
            auto reader = ring.make_reader();
            uint32_t last_seq = 0;
            bool has_last = false;
            for (int reads = 0; reads < 3; ++reads) {
                auto batch = reader.read_batch(std::span<ProgramRealtimeRecord>(out));
                for (const auto& r : batch) {
                    if (!seq_record_ok(r)) {
                        churn_fields_ok.store(false, std::memory_order_relaxed);
                    }
                    const uint32_t seq = r.runtime_id - 1;
                    if (has_last && seq <= last_seq) {
                        churn_ordered.store(false, std::memory_order_relaxed);
                    }
                    last_seq = seq;
                    has_last = true;
                }
            }
        }
    });

    std::thread writer_thread([&]() {
        ready.count_down();
        start.wait();
        std::array<ProgramRealtimeRecord, kMaxPublishBatchSize> records;
        uint64_t next = 0;
        while (!long_done.load(std::memory_order_acquire)) {
            while (next + kMaxPublishBatchSize - long_consumed.load(std::memory_order_acquire) > kCapacity / 2 &&
                   !long_done.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            const uint64_t base = next;
            for (size_t k = 0; k < kMaxPublishBatchSize; ++k) {
                records[k] = make_seq_record(static_cast<uint32_t>(base + k));
            }
            ring.writer().publish_batch(std::span<const ProgramRealtimeRecord>(records.data(), kMaxPublishBatchSize));
            next += kMaxPublishBatchSize;
        }
    });

    ready.wait();
    start.count_down();
    long_thread.join();
    churn_thread.join();
    writer_thread.join();

    uint32_t prev_seq = 0;
    bool has_prev = false;
    for (const auto& r : long_result) {
        const uint32_t seq = r.runtime_id - 1;
        ASSERT_TRUE(seq_record_ok(r)) << "long reader saw a torn/stale record at seq " << seq;
        if (has_prev) {
            ASSERT_GT(seq, prev_seq) << "long reader saw out-of-order / duplicate delivery at seq " << seq;
        }
        prev_seq = seq;
        has_prev = true;
    }
    EXPECT_EQ(long_reader.dropped(), 0u);
    EXPECT_GE(long_result.size(), kLongReaderTarget);
    EXPECT_TRUE(churn_fields_ok.load()) << "a churned reader saw a torn/stale record";
    EXPECT_TRUE(churn_ordered.load()) << "a churned reader saw out-of-order / duplicate delivery";
}

TEST(BroadcastRing, ConcurrentLockedSlotIntegrity) {
    struct NonTrivialRecord {
        uint32_t seq = 0;
        std::string payload;
    };
    static_assert(!BroadcastRing<NonTrivialRecord>::is_always_lock_free);

    constexpr uint64_t kTotalRecords = 1u << 15;
    constexpr size_t kCapacity = 256;
    constexpr size_t kReadBatchSize = 32;
    constexpr size_t kMaxPublishBatchSize = 48;

    // Heap-allocated and uniquely derived from seq so a corrupted or wrong-generation slot is detectable.
    auto make_payload = [](uint32_t seq) {
        return "rec-" + std::to_string(seq) + std::string(32, static_cast<char>('a' + seq % 26));
    };

    BroadcastRing<NonTrivialRecord> ring(kCapacity);
    auto reader = ring.make_reader();

    std::vector<NonTrivialRecord> received;
    received.reserve(kTotalRecords);
    uint64_t dropped = 0;

    std::latch ready{2};
    std::latch start{1};
    std::atomic<bool> writer_done{false};

    std::thread reader_thread([&]() {
        std::array<NonTrivialRecord, kReadBatchSize> out;
        ready.count_down();
        start.wait();
        while (true) {
            const bool done = writer_done.load(std::memory_order_acquire);
            const uint64_t dropped_before = reader.dropped();
            auto batch = reader.read_batch(std::span<NonTrivialRecord>(out));
            if (batch.empty()) {
                if (reader.dropped() != dropped_before) {
                    continue;
                }
                if (done) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }
            for (auto& r : batch) {
                received.push_back(std::move(r));
            }
        }
        dropped = reader.dropped();
    });

    std::thread writer_thread([&]() {
        ready.count_down();
        start.wait();
        std::mt19937 rng(kBatchSeed);
        std::uniform_int_distribution<size_t> pick_batch(1, kMaxPublishBatchSize);
        std::vector<NonTrivialRecord> records(kMaxPublishBatchSize);
        uint64_t next = 0;
        while (next < kTotalRecords) {
            const uint64_t base = next;
            const size_t batch_size = std::min<uint64_t>(pick_batch(rng), kTotalRecords - base);
            for (size_t k = 0; k < batch_size; ++k) {
                records[k].seq = static_cast<uint32_t>(base + k);
                records[k].payload = make_payload(static_cast<uint32_t>(base + k));
            }
            ring.writer().publish_batch_move(std::span<NonTrivialRecord>(records.data(), batch_size));
            next += batch_size;
        }
        writer_done.store(true, std::memory_order_release);
    });

    ready.wait();
    start.count_down();
    writer_thread.join();
    reader_thread.join();

    EXPECT_EQ(received.size() + dropped, kTotalRecords);
    uint32_t prev_seq = 0;
    bool has_prev = false;
    for (const auto& r : received) {
        ASSERT_EQ(r.payload, make_payload(r.seq)) << "corrupted LockedSlot record at seq " << r.seq;
        if (has_prev) {
            ASSERT_GT(r.seq, prev_seq) << "out-of-order / duplicate LockedSlot delivery at seq " << r.seq;
        }
        prev_seq = r.seq;
        has_prev = true;
    }
    EXPECT_GT(received.size(), 0u);
}

}  // namespace
}  // namespace tt::tt_metal
