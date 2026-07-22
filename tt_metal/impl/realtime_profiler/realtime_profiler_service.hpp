// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "tt_metal/common/broadcast_ring.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_consumer.hpp"

namespace tt::tt_metal {

using RealtimeProfilerRecordRing = BroadcastRing<tt::tt_metal::experimental::ProgramRealtimeRecord>;

// Context-wide owner of real-time profiler consumers. Managers attach independent record rings; every registered
// consumer gets one delivery thread which reads all attached rings.
class RealtimeProfilerService {
public:
    RealtimeProfilerService() = default;
    ~RealtimeProfilerService();

    RealtimeProfilerService(const RealtimeProfilerService&) = delete;
    RealtimeProfilerService& operator=(const RealtimeProfilerService&) = delete;
    RealtimeProfilerService(RealtimeProfilerService&&) = delete;
    RealtimeProfilerService& operator=(RealtimeProfilerService&&) = delete;

    tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle register_consumer(
        std::unique_ptr<RealtimeProfilerConsumer> consumer);
    void unregister_consumer(tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle handle);

    void attach_ring(RealtimeProfilerRecordRing& ring, size_t max_batch_records);
    // The ring's writer must be stopped before this call. Blocks until every consumer has drained and released it.
    void detach_ring(RealtimeProfilerRecordRing& ring);

    // Wakes delivery threads after a manager publishes records.
    void wake_consumers() noexcept;

    bool is_active() const;

private:
    struct RingReader {
        RingReader(RealtimeProfilerRecordRing::Reader reader, size_t max_batch_records) :
            reader(std::move(reader)), max_batch_records(max_batch_records) {}

        RealtimeProfilerRecordRing::Reader reader;
        size_t max_batch_records;
        uint64_t observed_dropped = 0;
        bool draining = false;
    };

    struct ConsumerRegistration {
        ConsumerRegistration(
            tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle handle,
            std::unique_ptr<RealtimeProfilerConsumer> consumer) :
            handle(handle), consumer(std::move(consumer)) {}

        ConsumerRegistration(const ConsumerRegistration&) = delete;
        ConsumerRegistration& operator=(const ConsumerRegistration&) = delete;
        ConsumerRegistration(ConsumerRegistration&&) = delete;
        ConsumerRegistration& operator=(ConsumerRegistration&&) = delete;

        tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle handle;
        std::unique_ptr<RealtimeProfilerConsumer> consumer;

        // Worker-owned hot state.
        std::unordered_map<RealtimeProfilerRecordRing*, RingReader> readers;

        // Cold control inbox. Record delivery only checks control_pending and otherwise does not take this mutex.
        std::mutex control_mutex;
        std::atomic<bool> control_pending{false};
        std::unordered_map<RealtimeProfilerRecordRing*, RingReader> readers_to_add;
        std::vector<RealtimeProfilerRecordRing*> rings_to_drain;

        // Total losses observed across all readers since this consumer's previous record callback.
        uint64_t pending_dropped = 0;
        std::jthread thread;
    };

    using ConsumerMap =
        std::unordered_map<tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle, ConsumerRegistration>;

    void run_consumer(std::stop_token stop_token, ConsumerRegistration& registration);
    void stop_registration(ConsumerMap::node_type registration);

    mutable std::mutex topology_mutex_;
    std::unordered_map<RealtimeProfilerRecordRing*, size_t> max_batch_records_by_ring_;
    ConsumerMap consumers_;
    tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle next_consumer_handle_ = 0;
    std::atomic<uint32_t> wake_generation_{0};
};

}  // namespace tt::tt_metal
