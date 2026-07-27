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

#include "context/context_types.hpp"
#include "tt_metal/common/broadcast_ring.hpp"

namespace tt::tt_metal {

using RealtimeProfilerRecordRing = BroadcastRing<tt::tt_metal::experimental::ProgramRealtimeRecord>;

// Owner of real-time profiler consumers. Receivers attach independent record rings; every registered
// consumer gets one thread of its own, which drains every attached ring.
class RealtimeProfilerService {
public:
    RealtimeProfilerService() = default;
    ~RealtimeProfilerService();

    RealtimeProfilerService(const RealtimeProfilerService&) = delete;
    RealtimeProfilerService& operator=(const RealtimeProfilerService&) = delete;
    RealtimeProfilerService(RealtimeProfilerService&&) = delete;
    RealtimeProfilerService& operator=(RealtimeProfilerService&&) = delete;

    tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle register_consumer(
        tt::tt_metal::experimental::ProgramRealtimeProfilerCallback callback);
    void unregister_consumer(tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle handle);

    void attach_ring(RealtimeProfilerRecordRing& ring, size_t max_batch_records);
    // The ring's writer must be stopped before this call. Blocks until every consumer has drained and released it.
    void detach_ring(RealtimeProfilerRecordRing& ring);

    // Wakes the consumer threads after a receiver publishes records.
    void wake_consumers() noexcept;

    bool is_active() const;

private:
    struct RingReader {
        RingReader(RealtimeProfilerRecordRing* ring, RealtimeProfilerRecordRing::Reader reader, size_t max_batch) :
            ring(ring), reader(std::move(reader)), max_batch_records(max_batch) {}

        RealtimeProfilerRecordRing* ring;
        RealtimeProfilerRecordRing::Reader reader;
        size_t max_batch_records;
        uint64_t observed_dropped = 0;
        bool draining = false;
    };

    struct ConsumerRegistration {
        ConsumerRegistration(
            tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle handle,
            tt::tt_metal::experimental::ProgramRealtimeProfilerCallback callback) :
            handle(handle), callback(std::move(callback)) {}

        ConsumerRegistration(const ConsumerRegistration&) = delete;
        ConsumerRegistration& operator=(const ConsumerRegistration&) = delete;
        ConsumerRegistration(ConsumerRegistration&&) = delete;
        ConsumerRegistration& operator=(ConsumerRegistration&&) = delete;

        tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle handle;
        tt::tt_metal::experimental::ProgramRealtimeProfilerCallback callback;

        std::vector<RingReader> readers;

        // Set by a callback retiring itself.
        std::atomic<bool> retired{false};

        // Cross-thread inbox. The hot loop reads only control_pending, which stays shared in cache; control_mutex
        // guards the rest.
        std::atomic<bool> control_pending{false};
        std::mutex control_mutex;
        std::vector<RingReader> readers_to_add;
        std::vector<RealtimeProfilerRecordRing*> rings_to_drain;

        std::jthread thread;
    };

    using ConsumerMap =
        std::unordered_map<tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle, ConsumerRegistration>;

    void run_consumer(std::stop_token stop_token, ConsumerRegistration& registration);
    void destroy_consumer(ConsumerRegistration& registration);
    // Joins and drops registrations that retired themselves. A callback cannot join its own thread, so it only marks
    // itself retired; the next caller that is not a consumer thread finishes the job.
    void reap_retired_consumers();

    // The registration this consumer thread is serving, so a callback can be recognized as unregistering itself.
    inline static thread_local ConsumerRegistration* current_registration_ = nullptr;

    mutable std::mutex topology_mutex_;
    // Attached rings and the batch size limit each was attached with, so a consumer registering after a ring
    // arrives can build a reader for it.
    std::unordered_map<RealtimeProfilerRecordRing*, size_t> attached_rings_;
    ConsumerMap consumers_;
    tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle next_consumer_handle_ = 0;

    std::atomic<uint32_t> wake_generation_{0};
};

// Process-wide: a registration is owned by whoever made it and ends only at Unregister, so it cannot be scoped to
// anything shorter-lived.
RealtimeProfilerService& realtime_profiler_service();

void register_builtin_realtime_profiler_consumers();

}  // namespace tt::tt_metal
