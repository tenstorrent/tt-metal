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

using RealtimeProfilerRecordRing = BroadcastRing<experimental::ProgramRealtimeRecord>;

// Owner of real-time profiler consumers. Receivers attach independent record rings; each registered
// consumer gets its own thread draining every attached ring.
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

    void attach_ring(RealtimeProfilerRecordRing& ring, size_t max_batch_records);
    // The ring's writer must be stopped before this call. Blocks until every consumer has drained and released it.
    void detach_ring(RealtimeProfilerRecordRing& ring);

    // Wakes the consumer threads after a receiver publishes records.
    void wake_consumers() noexcept;

    bool is_active() const;

    // Atomic (not a locked consumers_ lookup) because the drain loop reads this on every iteration.
    bool has_consumers() const { return num_consumers_.load(std::memory_order_relaxed) != 0; }

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
            experimental::ProgramRealtimeProfilerCallbackHandle handle,
            experimental::ProgramRealtimeProfilerCallback callback) :
            handle(handle), callback(std::move(callback)) {}

        ConsumerRegistration(const ConsumerRegistration&) = delete;
        ConsumerRegistration& operator=(const ConsumerRegistration&) = delete;
        ConsumerRegistration(ConsumerRegistration&&) = delete;
        ConsumerRegistration& operator=(ConsumerRegistration&&) = delete;

        experimental::ProgramRealtimeProfilerCallbackHandle handle;
        experimental::ProgramRealtimeProfilerCallback callback;

        std::vector<RingReader> readers;

        // Set by a callback retiring itself.
        std::atomic<bool> retired{false};

        // Cross-thread inbox: the hot loop only checks control_pending; control_mutex guards the rest.
        std::atomic<bool> control_pending{false};
        std::mutex control_mutex;
        std::vector<RingReader> readers_to_add;
        std::vector<RealtimeProfilerRecordRing*> rings_to_drain;

        std::jthread thread;
    };

    using ConsumerMap = std::unordered_map<experimental::ProgramRealtimeProfilerCallbackHandle, ConsumerRegistration>;

    void run_consumer(std::stop_token stop_token, ConsumerRegistration& registration);
    void destroy_consumer(ConsumerRegistration& registration);
    // A callback cannot join its own thread, so a self-unregistering consumer only marks itself retired here;
    // this joins and drops it on behalf of the next non-consumer-thread caller.
    void reap_retired_consumers();

    // Identifies which registration the running consumer thread belongs to.
    inline static thread_local ConsumerRegistration* current_registration_ = nullptr;

    mutable std::mutex topology_mutex_;
    // Batch-size limit per attached ring, so a consumer registering later can build readers for existing rings.
    std::unordered_map<RealtimeProfilerRecordRing*, size_t> attached_rings_;
    ConsumerMap consumers_;
    experimental::ProgramRealtimeProfilerCallbackHandle next_consumer_handle_ = 0;

    std::atomic<uint32_t> wake_generation_{0};
    std::atomic<size_t> num_consumers_{0};
};

// Process-wide singleton: a registration lives until an explicit Unregister call.
RealtimeProfilerService& realtime_profiler_service();

void register_builtin_realtime_profiler_consumers();

}  // namespace tt::tt_metal
