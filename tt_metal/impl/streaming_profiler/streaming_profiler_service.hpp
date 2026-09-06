// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The streaming profiler's consumer side, one per process. Each capture's receiver attaches as a producer;
// each subscriber is a consumer on its own thread, with a reader per producer ring, decoding frames into records
// for its callback. A consumer that falls behind drops its own oldest lines and nothing else backs up. Producers
// come and go with MeshDevices; consumers persist until removed.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"

namespace tt::llrt {
class RunTimeOptions;
}

namespace tt::tt_metal {
template <typename T>
class BroadcastRing;
namespace streaming_profiler {

struct RingLine;
class TracySink;

struct ProducerStream {
    BroadcastRing<RingLine>* ring = nullptr;
    uint32_t dev = 0;  // index into the producer's CaptureContext::devices
};

// One capture's frame rings and what a consumer needs to decode them. Valid from attach_producer() until
// detach_producer() returns.
class Producer {
public:
    virtual ~Producer() = default;
    virtual std::span<const ProducerStream> streams() const = 0;
    virtual const CaptureContext& capture_context() const = 0;
    virtual std::vector<experimental::streaming_profiler::Clock> clocks() const = 0;
};

// Wraps a public-batch callback as an internal record consumer: decodes the records of the subscribed channels,
// resolves names and cores, assembles payloads. Defined with the public API.
RecordCallback make_public_adapter(
    experimental::streaming_profiler::Channel channels,
    std::function<void(const experimental::streaming_profiler::Batch<experimental::streaming_profiler::Channel::All>&)>
        callback);

class Service {
public:
    Service();
    ~Service();
    Service(const Service&) = delete;
    Service& operator=(const Service&) = delete;

    // The callback runs on the consumer's own thread, one call at a time, for every attached producer. Not from
    // inside a consumer callback.
    ConsumerHandle add_consumer(std::string name, RecordCallback cb);
    // Returns once the callback can no longer run.
    void remove_consumer(ConsumerHandle handle);

    // Returns once every consumer reads the producer's rings, so nothing published afterwards is missed.
    void attach_producer(Producer& producer);
    // Returns once every consumer has drained the producer's rings and released its readers. Detaching the last
    // producer writes the file sinks.
    void detach_producer(Producer& producer);
    bool is_active() const;

    // The Tracy sink and the CSV writers rtoptions select; subsequent calls do nothing.
    void register_builtin_consumers(const tt::llrt::RunTimeOptions& rtoptions);

    // A producer calls this once after a pass that published. A reader takes wake_token() before checking the rings
    // and, finding nothing, wait_wake()s on it, so a bump between the two returns at once.
    void wake_consumers() {
        wake_gen_.fetch_add(1, std::memory_order_release);
        wake_gen_.notify_all();
    }
    uint32_t wake_token() const { return wake_gen_.load(std::memory_order_acquire); }
    void wait_wake(uint32_t seen) const { wake_gen_.wait(seen, std::memory_order_acquire); }

private:
    struct Consumer;
    struct FileSink {
        std::string name;
        std::string path;
        std::shared_ptr<void> owner;
        std::function<void(const std::string&)> write;
    };
    void consumer_thread(Consumer& c);
    void post_control(Consumer& c, Producer* attach, Producer* detach);
    void wait_acks(std::unique_lock<std::mutex>& lk);
    void log_consumer_drops() const;

    // Serializes add/remove/attach/detach against each other; never taken by a consumer thread.
    std::mutex topology_mu_;
    // Guards the lists below and the ack handshake; taken briefly by consumer threads.
    mutable std::mutex mu_;
    std::condition_variable ack_cv_;
    uint64_t pending_acks_ = 0;
    alignas(64) std::atomic<uint32_t> wake_gen_{0};
    std::vector<std::unique_ptr<Consumer>> consumers_;
    std::vector<Producer*> producers_;
    std::vector<FileSink> sinks_;
    std::unique_ptr<TracySink> tracy_;
    ConsumerHandle next_handle_ = 1;
    bool builtins_registered_ = false;
};

// The process's Service. Subscriptions outlive every device and context, so it is never destroyed.
Service& service();

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
