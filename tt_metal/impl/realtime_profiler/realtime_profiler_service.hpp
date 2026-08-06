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
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "context/context_types.hpp"
#include "tt_metal/common/broadcast_ring.hpp"

namespace tt::tt_metal {

using RealtimeProfilerRecordRing = BroadcastRing<experimental::ProgramRealtimeRecord>;

/**
 * @brief A source of real-time profiler records that consumers can be attached to.
 *
 * Consumers attach to a producer rather than to whatever transport the producer publishes through, so the service that
 * owns them never has to know where records come from. One producer exists per MeshDevice in a normal run; tests
 * implement this over a bare ring.
 *
 * Threading: every method may be called concurrently with the producer publishing, and make_reader() may be called
 * concurrently from consumer registration. Implementations are responsible for that; the ring-backed ones get it from
 * the ring.
 */
class ProgramRecordProducer {
public:
    ProgramRecordProducer() = default;
    virtual ~ProgramRecordProducer() = default;

    ProgramRecordProducer(const ProgramRecordProducer&) = delete;
    ProgramRecordProducer& operator=(const ProgramRecordProducer&) = delete;
    ProgramRecordProducer(ProgramRecordProducer&&) = delete;
    ProgramRecordProducer& operator=(ProgramRecordProducer&&) = delete;

    /** @brief Most records one callback may be handed at a time from this producer. */
    [[nodiscard]] virtual size_t max_batch_records() const = 0;

    /** @brief A reader positioned so that it sees exactly the records published after this call. */
    [[nodiscard]] virtual RealtimeProfilerRecordRing::Reader make_reader() = 0;

    /**
     * @brief Blocks until every reader made from this producer has been released.
     *
     * The producer must have stopped publishing before this is called.
     */
    virtual void wait_until_no_readers() = 0;

    /**
     * @brief Records published so far.
     *
     * A reader sees exactly the records published after it was made, so a consumer's accounting is
     * `received + dropped == num_published_records() - (its value when that consumer's reader was made)`. Comparing one
     * consumer's total against another's is only valid when both readers were made at the same point in the stream.
     */
    [[nodiscard]] virtual uint64_t num_published_records() const = 0;
};

// Owner of real-time profiler consumers. Producers attach themselves; each registered consumer gets its own thread
// draining every attached producer.
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

    void attach_producer(ProgramRecordProducer& producer);
    // The producer must have stopped publishing before this call. Blocks until every consumer has drained and released
    // it. Idempotent.
    void detach_producer(ProgramRecordProducer& producer);

    // Wakes the consumer threads after a receiver publishes records.
    void wake_consumers() noexcept;

    bool is_active() const;

private:
    struct ProducerReader {
        ProducerReader(ProgramRecordProducer* producer, RealtimeProfilerRecordRing::Reader reader, size_t max_batch) :
            producer(producer), reader(std::move(reader)), max_batch_records(max_batch) {}

        ProgramRecordProducer* producer;
        RealtimeProfilerRecordRing::Reader reader;
        // Cached rather than asked of the producer per batch, since the drain loop reads it on every pass.
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

        // Set by a callback retiring itself.
        std::atomic<bool> retired{false};

        // Cross-thread inbox: the hot loop only checks control_pending; control_mutex guards the rest.
        std::atomic<bool> control_pending{false};
        std::mutex control_mutex;
        std::vector<ProducerReader> readers_to_add;
        std::vector<ProgramRecordProducer*> producers_to_drain;

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
    // So a consumer registering later can build readers for producers that are already attached.
    std::unordered_set<ProgramRecordProducer*> attached_producers_;
    ConsumerMap consumers_;
    experimental::ProgramRealtimeProfilerCallbackHandle next_consumer_handle_ = 0;

    std::atomic<uint32_t> wake_generation_{0};
};

// Process-wide singleton: a registration lives until an explicit Unregister call.
RealtimeProfilerService& realtime_profiler_service();

void register_builtin_realtime_profiler_consumers();

}  // namespace tt::tt_metal
