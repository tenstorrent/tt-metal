// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "tt_metal/common/broadcast_ring.hpp"

namespace tt::tt_metal {

using RealtimeProfilerRecordRing = BroadcastRing<experimental::ProgramRealtimeRecord>;

// Owner of real-time profiler consumers.
class RealtimeProfilerService {
public:
    RealtimeProfilerService() = default;
    ~RealtimeProfilerService();

    RealtimeProfilerService(const RealtimeProfilerService&) = delete;
    RealtimeProfilerService& operator=(const RealtimeProfilerService&) = delete;
    RealtimeProfilerService(RealtimeProfilerService&&) = delete;
    RealtimeProfilerService& operator=(RealtimeProfilerService&&) = delete;

    experimental::ProgramRealtimeProfilerCallbackHandle register_consumer(
        experimental::ProgramRealtimeProfilerCallback callback);
    void unregister_consumer(experimental::ProgramRealtimeProfilerCallbackHandle handle);

    // `max_batch_records` is the most records a callback may be handed at a time from this ring.
    void attach_producer(RealtimeProfilerRecordRing& ring, size_t max_batch_records) noexcept;
    // The producer must have stopped publishing before this call. Blocks until every consumer has drained and released
    // the ring. Idempotent.
    void detach_producer(RealtimeProfilerRecordRing& ring);

    // Wakes the consumer threads after a receiver publishes records.
    void wake_consumers() noexcept;

    bool is_active() const;

    // True while at least one consumer is registered.
    bool has_consumers() const { return num_live_consumers_.load(std::memory_order_relaxed) != 0; }

private:
    struct ProducerReader {
        ProducerReader(RealtimeProfilerRecordRing* ring, RealtimeProfilerRecordRing::Reader reader, size_t max_batch) :
            ring(ring), reader(std::move(reader)), max_batch_records(max_batch) {}

        RealtimeProfilerRecordRing* ring;
        RealtimeProfilerRecordRing::Reader reader;
        size_t max_batch_records;
        uint64_t observed_dropped = 0;
        bool draining = false;
    };

    struct ConsumerRegistration {
        ConsumerRegistration(
            experimental::ProgramRealtimeProfilerCallbackHandle handle,
            experimental::ProgramRealtimeProfilerCallback callback) :
            handle(handle), callback(std::move(callback)) {}

        ConsumerRegistration(const ConsumerRegistration&) = delete;
        ConsumerRegistration& operator=(const ConsumerRegistration&) = delete;
        ConsumerRegistration(ConsumerRegistration&&) = delete;
        ConsumerRegistration& operator=(ConsumerRegistration&&) = delete;

        experimental::ProgramRealtimeProfilerCallbackHandle handle;
        experimental::ProgramRealtimeProfilerCallback callback;

        std::vector<ProducerReader> readers;

        // Set by a callback unregistering itself.
        std::atomic<bool> retired{false};

        // Cross-thread inbox. The hot loop only checks control_pending; control_mutex guards the actual resources.
        std::atomic<bool> control_pending{false};
        std::mutex control_mutex;
        std::vector<ProducerReader> readers_to_add;
        std::vector<RealtimeProfilerRecordRing*> rings_to_drain;

        std::jthread thread;
    };

    void run_consumer(std::stop_token stop_token, ConsumerRegistration& registration);
    void destroy_consumer(ConsumerRegistration& registration);
    // A callback cannot join its own thread, so a self-unregistering consumer only marks itself retired;
    // this joins and drops it from the service on behalf of the next non-consumer-thread caller.
    void reap_retired_consumers();

    // Identifies which registration the running consumer thread belongs to.
    inline static thread_local ConsumerRegistration* current_registration_ = nullptr;

    mutable std::mutex topology_mutex_;
    struct AttachedProducer {
        RealtimeProfilerRecordRing* ring;
        size_t max_batch_records;
    };
    std::vector<AttachedProducer> attached_producers_;
    using ConsumerMap = std::unordered_map<experimental::ProgramRealtimeProfilerCallbackHandle, ConsumerRegistration>;
    ConsumerMap consumers_;
    experimental::ProgramRealtimeProfilerCallbackHandle next_consumer_handle_ = 0;

    std::atomic<uint32_t> wake_generation_{0};
    std::atomic<size_t> num_live_consumers_{0};
};

// Process-wide singleton; callback registrations live until an explicit UnregisterProgramRealtimeProfilerCallback call.
RealtimeProfilerService& realtime_profiler_service();

}  // namespace tt::tt_metal
