// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <memory>
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

class RealtimeProfilerTracyConsumer;

using RealtimeProfilerRecordRing = BroadcastRing<experimental::ProgramRealtimeRecord>;

// Owner of real-time profiler consumers. Receivers attach their record rings; each registered consumer gets its own
// thread, which drains every attached ring into the consumer's callback.
class RealtimeProfilerService {
public:
    RealtimeProfilerService();

    RealtimeProfilerService(const RealtimeProfilerService&) = delete;
    RealtimeProfilerService& operator=(const RealtimeProfilerService&) = delete;

    uint64_t register_consumer(const experimental::ProgramRealtimeProfilerCallback& callback);
    void unregister_consumer(uint64_t handle);

    // `max_batch_records` is the most records a callback may be handed at a time from this ring.
    void attach_producer(RealtimeProfilerRecordRing& ring, size_t max_batch_records) noexcept;
    // The producer must have stopped publishing before this call. Blocks until every consumer has drained and released
    // the ring.
    void detach_producer(RealtimeProfilerRecordRing& ring);

    // Wakes the consumer threads after a receiver publishes records.
    void wake_consumers() noexcept;

    // True while a receiver is attached.
    bool is_active() const { return num_attached_producers_.load(std::memory_order_relaxed) != 0; }

    RealtimeProfilerTracyConsumer& tracy() { return *tracy_; }

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

    // Signalled once a consumer's thread has been joined and the consumer removed. Shared, so a thread waiting on it
    // keeps it alive after the registration is gone.
    struct Completion {
        std::mutex mutex;
        std::condition_variable condition;
        bool done = false;
    };

    struct ConsumerRegistration {
        ConsumerRegistration(uint64_t handle, experimental::ProgramRealtimeProfilerCallback callback) :
            handle(handle), callback(std::move(callback)) {}

        uint64_t handle;
        experimental::ProgramRealtimeProfilerCallback callback;

        // Touched only by the consumer thread: its readers, the buffer batches are read into, and drops noticed on
        // a reader with nothing to deliver, carried until a batch exists to report them on.
        std::vector<ProducerReader> readers;
        std::vector<experimental::ProgramRealtimeRecord> records;
        uint64_t pending_dropped = 0;

        // Set by a callback unregistering itself.
        std::atomic<bool> retired{false};

        // Set by the one thread that joins this consumer's thread and removes it. Until that is done the registration
        // stays in the service, so a second unregister of the same handle finds it and waits on `finished`, and does
        // not return while the callback is still running.
        std::atomic<bool> destroying{false};
        std::shared_ptr<Completion> finished = std::make_shared<Completion>();

        // Cross-thread inbox. The hot loop only checks control_pending; control_mutex guards the actual resources.
        std::atomic<bool> control_pending{false};
        std::mutex control_mutex;
        std::vector<ProducerReader> readers_to_add;
        std::vector<RealtimeProfilerRecordRing*> rings_to_drain;

        std::jthread thread;
    };

    void run_consumer(const std::stop_token& stop_token, ConsumerRegistration& registration);
    // Takes the readers to add and the rings to drain from the inbox. False once the consumer has retired.
    bool apply_control(ConsumerRegistration& registration);
    // Hands the callback one batch from each reader that has records, and releases drained readers. True if it did
    // either.
    bool deliver(ConsumerRegistration& registration);
    void invoke_callback(
        ConsumerRegistration& registration,
        std::span<const experimental::ProgramRealtimeRecord> batch,
        uint64_t dropped);
    // Joins the consumer's thread, removes it from the service and signals anyone waiting on it. Only the thread that
    // won `destroying` calls this.
    void destroy_consumer(ConsumerRegistration& registration);
    // Unregisters the consumer on a thread that is not its own. Returns once its callback can no longer run: if
    // another thread is already destroying it, waits for that.
    void unregister_from_other_thread(uint64_t handle);
    // A callback cannot join its own thread, so a self-unregistering consumer only marks itself retired;
    // this joins and drops it from the service on behalf of the next non-consumer-thread caller.
    void reap_retired_consumers();

    // Identifies which registration the running consumer thread belongs to.
    inline static thread_local ConsumerRegistration* current_registration_ = nullptr;

    std::mutex topology_mutex_;
    struct AttachedProducer {
        RealtimeProfilerRecordRing* ring;
        size_t max_batch_records;
    };
    std::vector<AttachedProducer> attached_producers_;
    std::atomic<size_t> num_attached_producers_{0};
    using ConsumerMap = std::unordered_map<uint64_t, ConsumerRegistration>;
    ConsumerMap consumers_;
    uint64_t next_consumer_handle_ = 0;
    // The Tracy consumer's handle. Its thread also gets idle time, to place zones that new anchors have unblocked.
    uint64_t tracy_handle_;

    std::atomic<uint32_t> wake_generation_{0};
    std::atomic<size_t> num_live_consumers_{0};

    std::unique_ptr<RealtimeProfilerTracyConsumer> tracy_;
};

// The process-wide service. It is never destroyed: a callback stays registered until it is unregistered or the process
// exits.
RealtimeProfilerService& realtime_profiler_service();

}  // namespace tt::tt_metal
