// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tt_metal/impl/realtime_profiler/realtime_profiler_service.hpp"

namespace tt::tt_metal {
namespace {

using namespace std::chrono_literals;
using experimental::ProgramRealtimeRecord;
using experimental::ProgramRealtimeRecordBatch;

ProgramRealtimeRecord make_record(uint32_t runtime_id, uint32_t chip_id) {
    return ProgramRealtimeRecord{
        .runtime_id = runtime_id,
        .chip_id = chip_id,
        .start_timestamp = runtime_id * 10,
        .end_timestamp = runtime_id * 10 + 5,
        .frequency = 1.0,
        .clock_sync = {.device_cycle_offset = 0, .sync_error_ns = 0},
        .kernel_sources = {},
    };
}

template <typename Predicate>
bool wait_for(std::condition_variable& cv, std::unique_lock<std::mutex>& lock, Predicate&& predicate) {
    return cv.wait_for(lock, 5s, std::forward<Predicate>(predicate));
}

TEST(RealtimeProfilerService, OneConsumerThreadReadsMultipleRings) {
    RealtimeProfilerService service;
    RealtimeProfilerRecordRing ring_a(16);
    RealtimeProfilerRecordRing ring_b(16);

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<ProgramRealtimeRecord> received;
    std::set<std::thread::id> callback_threads;
    const auto handle = service.register_consumer(
        std::make_unique<RealtimeProfilerConsumer>([&](const ProgramRealtimeRecordBatch& batch) {
            {
                std::lock_guard lock(mutex);
                callback_threads.insert(std::this_thread::get_id());
                received.insert(received.end(), batch.records.begin(), batch.records.end());
            }
            cv.notify_all();
        }));

    service.attach_ring(ring_a, 8);
    service.attach_ring(ring_b, 8);
    const ProgramRealtimeRecord records_a[] = {make_record(1, 1), make_record(2, 1)};
    const ProgramRealtimeRecord records_b[] = {make_record(3, 2), make_record(4, 2)};
    ring_a.writer().publish_batch(records_a);
    ring_b.writer().publish_batch(records_b);
    service.wake_consumers();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(wait_for(cv, lock, [&] { return received.size() == 4; }));
        EXPECT_EQ(callback_threads.size(), 1u);
        std::set<uint32_t> runtime_ids;
        for (const auto& record : received) {
            runtime_ids.insert(record.runtime_id);
        }
        EXPECT_EQ(runtime_ids, (std::set<uint32_t>{1, 2, 3, 4}));
    }

    service.detach_ring(ring_a);
    service.detach_ring(ring_b);
    service.unregister_consumer(handle);
}

TEST(RealtimeProfilerService, LateConsumerReceivesOnlyFutureRecords) {
    RealtimeProfilerService service;
    RealtimeProfilerRecordRing ring(8);
    service.attach_ring(ring, 4);

    // Published before the consumer registers: a reader starts at the current head, so this must not be delivered.
    ring.writer().publish(make_record(1, 7));
    service.wake_consumers();

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<uint32_t> received;
    const auto handle = service.register_consumer(
        std::make_unique<RealtimeProfilerConsumer>([&](const ProgramRealtimeRecordBatch& batch) {
            {
                std::lock_guard lock(mutex);
                for (const auto& record : batch.records) {
                    received.push_back(record.runtime_id);
                }
            }
            cv.notify_all();
        }));

    ring.writer().publish(make_record(2, 7));
    service.wake_consumers();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(wait_for(cv, lock, [&] { return !received.empty(); }));
        EXPECT_EQ(received, (std::vector<uint32_t>{2}));
    }

    service.detach_ring(ring);
    service.unregister_consumer(handle);
}

TEST(RealtimeProfilerService, ThrowingRecordConsumerDoesNotAffectSibling) {
    RealtimeProfilerService service;
    RealtimeProfilerRecordRing ring(8);
    service.attach_ring(ring, 4);

    const auto throwing = service.register_consumer(std::make_unique<RealtimeProfilerConsumer>(
        [](const ProgramRealtimeRecordBatch&) { throw std::runtime_error("intentional record failure"); }));

    std::mutex mutex;
    std::condition_variable cv;
    size_t good_count = 0;
    const auto good = service.register_consumer(
        std::make_unique<RealtimeProfilerConsumer>([&](const ProgramRealtimeRecordBatch& batch) {
            {
                std::lock_guard lock(mutex);
                good_count += batch.records.size();
            }
            cv.notify_all();
        }));

    ring.writer().publish(make_record(1, 1));
    service.wake_consumers();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(wait_for(cv, lock, [&] { return good_count == 1; }));
    }

    service.detach_ring(ring);
    service.unregister_consumer(throwing);
    service.unregister_consumer(good);
}

TEST(RealtimeProfilerService, UnregisterWaitsForInFlightCallback) {
    RealtimeProfilerService service;
    RealtimeProfilerRecordRing ring(8);
    service.attach_ring(ring, 4);

    std::mutex mutex;
    std::condition_variable cv;
    bool callback_started = false;
    bool release_callback = false;
    const auto handle =
        service.register_consumer(std::make_unique<RealtimeProfilerConsumer>([&](const ProgramRealtimeRecordBatch&) {
            std::unique_lock lock(mutex);
            callback_started = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_callback; });
        }));

    ring.writer().publish(make_record(1, 1));
    service.wake_consumers();
    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(wait_for(cv, lock, [&] { return callback_started; }));
    }

    std::atomic<bool> unregister_returned = false;
    std::thread unregister_thread([&] {
        service.unregister_consumer(handle);
        unregister_returned.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(20ms);
    EXPECT_FALSE(unregister_returned.load(std::memory_order_acquire));

    {
        std::lock_guard lock(mutex);
        release_callback = true;
    }
    cv.notify_all();
    unregister_thread.join();
    EXPECT_TRUE(unregister_returned.load(std::memory_order_acquire));
    service.detach_ring(ring);
}

TEST(RealtimeProfilerService, DropsAreReportedSinceThePreviousCallback) {
    RealtimeProfilerService service;
    RealtimeProfilerRecordRing overflowing_ring(4);
    RealtimeProfilerRecordRing blocked_ring(4);
    service.attach_ring(overflowing_ring, 4);
    service.attach_ring(blocked_ring, 4);

    struct BatchResult {
        uint32_t chip_id;
        uint64_t dropped;
    };
    std::mutex mutex;
    std::condition_variable cv;
    bool blocked_callback_started = false;
    bool release_blocked_callback = false;
    std::vector<BatchResult> results;
    const auto handle = service.register_consumer(
        std::make_unique<RealtimeProfilerConsumer>([&](const ProgramRealtimeRecordBatch& batch) {
            std::unique_lock lock(mutex);
            const uint32_t chip_id = batch.records.front().chip_id;
            if (chip_id == 2 && !blocked_callback_started) {
                blocked_callback_started = true;
                cv.notify_all();
                cv.wait(lock, [&] { return release_blocked_callback; });
            }
            results.push_back({chip_id, batch.dropped});
            cv.notify_all();
        }));

    blocked_ring.writer().publish(make_record(1, 2));
    service.wake_consumers();
    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(wait_for(cv, lock, [&] { return blocked_callback_started; }));
    }

    for (uint32_t id = 10; id < 22; ++id) {
        overflowing_ring.writer().publish(make_record(id, 1));
    }
    service.wake_consumers();
    {
        std::lock_guard lock(mutex);
        release_blocked_callback = true;
    }
    cv.notify_all();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(wait_for(cv, lock, [&] {
            return std::any_of(results.begin(), results.end(), [](const auto& result) {
                return result.chip_id == 1 && result.dropped > 0;
            });
        }));
    }

    service.detach_ring(overflowing_ring);
    service.detach_ring(blocked_ring);
    service.unregister_consumer(handle);
}

TEST(RealtimeProfilerService, DetachDrainsPublishedRecordsBeforeReturning) {
    RealtimeProfilerService service;
    RealtimeProfilerRecordRing ring(16);
    service.attach_ring(ring, 8);

    std::atomic<size_t> delivered = 0;
    const auto handle = service.register_consumer(std::make_unique<RealtimeProfilerConsumer>(
        [&](const ProgramRealtimeRecordBatch& batch) { delivered.fetch_add(batch.records.size()); }));
    for (uint32_t id = 0; id < 8; ++id) {
        ring.writer().publish(make_record(id, 1));
    }
    service.wake_consumers();

    service.detach_ring(ring);
    EXPECT_EQ(delivered.load(), 8u);
    service.unregister_consumer(handle);
}

TEST(RealtimeProfilerService, DestructorWakesIdleConsumerThread) {
    RealtimeProfilerService service;
    service.register_consumer(std::make_unique<RealtimeProfilerConsumer>());
}

TEST(RealtimeProfilerService, ManyRingsAndConsumersDeliverWithoutLoss) {
    constexpr size_t kRingCount = 8;
    constexpr size_t kConsumerCount = 4;
    constexpr size_t kRecordsPerRing = 64;

    RealtimeProfilerService service;
    std::vector<std::unique_ptr<RealtimeProfilerRecordRing>> rings;
    rings.reserve(kRingCount);
    for (size_t i = 0; i < kRingCount; ++i) {
        rings.push_back(std::make_unique<RealtimeProfilerRecordRing>(128));
        service.attach_ring(*rings.back(), 32);
    }

    std::vector<std::atomic<size_t>> received(kConsumerCount);
    for (auto& count : received) {
        count.store(0, std::memory_order_relaxed);
    }
    std::vector<experimental::ProgramRealtimeProfilerCallbackHandle> handles;
    handles.reserve(kConsumerCount);
    for (size_t consumer = 0; consumer < kConsumerCount; ++consumer) {
        handles.push_back(service.register_consumer(std::make_unique<RealtimeProfilerConsumer>(
            [&count = received[consumer]](const ProgramRealtimeRecordBatch& batch) {
                count.fetch_add(batch.records.size(), std::memory_order_relaxed);
            })));
    }

    for (size_t ring = 0; ring < kRingCount; ++ring) {
        for (size_t record = 0; record < kRecordsPerRing; ++record) {
            rings[ring]->writer().publish(
                make_record(static_cast<uint32_t>(ring * kRecordsPerRing + record), static_cast<uint32_t>(ring)));
        }
    }
    service.wake_consumers();

    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
        const bool complete = std::all_of(received.begin(), received.end(), [](const auto& count) {
            return count.load(std::memory_order_relaxed) == kRingCount * kRecordsPerRing;
        });
        if (complete) {
            break;
        }
        std::this_thread::sleep_for(1ms);
    }
    for (const auto& count : received) {
        EXPECT_EQ(count.load(std::memory_order_relaxed), kRingCount * kRecordsPerRing);
    }

    for (auto& ring : rings) {
        service.detach_ring(*ring);
    }
    for (auto handle : handles) {
        service.unregister_consumer(handle);
    }
}

}  // namespace
}  // namespace tt::tt_metal
