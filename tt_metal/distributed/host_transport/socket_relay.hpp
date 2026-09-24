// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "tt_metal/distributed/host_transport/host_transport.hpp"

namespace tt::tt_metal::distributed {
class D2HSocket;
class H2DSocket;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::distributed::host_transport {

// One relayed direction. poll() never blocks.
class RelayEndpoint {
public:
    virtual ~RelayEndpoint() = default;
    virtual bool poll() = 0;

    // Watermark, not quiesce. Waiting for full idle never returns on a socket
    // the peer keeps feeding.
    virtual uint64_t watermark() const = 0;
    virtual bool drained(uint64_t target) const = 0;

    virtual std::string describe() const = 0;

    // Bumped after every completed poll. All the counters below start at zero, so
    // "nothing outstanding" and "never looked" are indistinguishable without it:
    // a barrier entered before the relay's first poll would otherwise conclude
    // the stream was already drained.
    uint64_t polls() const { return polls_.load(std::memory_order_acquire); }

    // poll() runs on the relay thread, where an escaping exception would kill the
    // process instead of failing the caller. The flag is atomic so the hot barrier
    // loop can test it without locking; the message is copied under a lock, since
    // the relay thread writes it.
    bool failed() const { return failed_.load(std::memory_order_acquire); }
    std::string error() const {
        std::lock_guard<std::mutex> lock(error_mutex_);
        return error_;
    }

protected:
    void set_error(std::string message) {
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            error_ = std::move(message);
        }
        failed_.store(true, std::memory_order_release);
    }

    // Bumps polls_ on every exit path from poll().
    struct PollMark {
        std::atomic<uint64_t>& counter;
        ~PollMark() { counter.fetch_add(1, std::memory_order_release); }
    };

    mutable std::mutex error_mutex_;
    std::string error_;
    std::atomic<bool> failed_{false};
    std::atomic<uint64_t> polls_{0};
    friend class RelayLoop;
};

// Device -> local host -> peer host. Retiring a page is what frees ring space
// for the device kernel.
class RelaySender final : public RelayEndpoint {
public:
    RelaySender(
        HostTransport& transport,
        tt::tt_metal::distributed::D2HSocket& socket,
        RingGeometry geometry,
        uint32_t max_batch_pages = 8);

    bool poll() override;
    uint64_t watermark() const override { return forwarded_.load(std::memory_order_acquire); }
    // Converges because the local device stops producing before a barrier; no
    // peer races ahead on this side.
    bool drained(uint64_t target) const override {
        return retired_.load(std::memory_order_acquire) >= target &&
               forwarded_.load(std::memory_order_acquire) == retired_.load(std::memory_order_acquire) &&
               unforwarded_.load(std::memory_order_acquire) == 0;
    }
    std::string describe() const override;

    uint64_t pages_forwarded() const { return forwarded_.load(std::memory_order_acquire); }
    uint64_t pages_retired() const { return retired_.load(std::memory_order_acquire); }

    // Batch-handoff to peer-credit round trip. Off by default: one timestamp per
    // batch.
    void set_credit_latency_sampling(bool enabled);
    std::vector<uint64_t> take_credit_latencies_ns();

private:
    HostTransport& transport_;
    tt::tt_metal::distributed::D2HSocket& socket_;
    RingGeometry geom_;
    uint32_t max_batch_pages_;
    // Written by the polling thread, read by a barrier on another.
    std::atomic<uint64_t> forwarded_{0};
    std::atomic<uint64_t> retired_{0};
    std::atomic<uint32_t> unforwarded_{0};  // pages seen available at the last poll

    struct Sample {
        uint64_t total;  // forwarded total this batch reached
        uint64_t sent_ns;
    };
    std::atomic<bool> sampling_{false};
    std::deque<Sample> pending_samples_;
    std::vector<uint64_t> latencies_ns_;
    mutable std::mutex latency_mutex_;
};

// Peer host -> local host -> device. The peer's NIC already placed the payload
// in the ring; this only publishes it and returns credit.
class RelayReceiver final : public RelayEndpoint {
public:
    RelayReceiver(HostTransport& transport, tt::tt_metal::distributed::H2DSocket& socket, RingGeometry geometry);

    bool poll() override;
    uint64_t watermark() const override { return arrived_.load(std::memory_order_acquire); }
    // Nothing host-side to drain. Waiting for device consumption deadlocks: the
    // peer pipelines into the next round, so the barrier would block on pages
    // only the caller's *next* kernel launch reads.
    bool drained(uint64_t) const override { return true; }
    std::string describe() const override;

    uint64_t pages_arrived() const { return arrived_.load(std::memory_order_acquire); }

private:
    HostTransport& transport_;
    tt::tt_metal::distributed::H2DSocket& socket_;
    RingGeometry geom_;
    // Written by the polling thread, read by a barrier on another.
    std::atomic<uint64_t> arrived_{0};
    std::atomic<uint64_t> credited_pages_{0};
    // bytes_acked wraps at 2^32 but page_size need not divide it, so the division
    // to pages has to happen on an unwrapped total: accumulate 32-bit deltas.
    uint64_t consumed_bytes_ = 0;
    uint32_t last_acked_wire_ = 0;
};

// Process-wide non-blocking event loop; one thread services every endpoint.
class RelayLoop {
public:
    static RelayLoop& instance();

    void add(const std::shared_ptr<RelayEndpoint>& endpoint);
    void remove(const std::shared_ptr<RelayEndpoint>& endpoint);

    // Serialized against add()/remove(): a removed endpoint is never polled once
    // remove() returns.
    bool poll_once();

    std::string first_error() const;

    // Serialized against poll_once(), which holds the same lock for a whole sweep.
    // describe() reads counters the relay thread mutates without synchronization
    // of their own, so formatting them from another thread needs this. Only the
    // timeout path uses it, so the contention does not matter.
    std::string describe(const RelayEndpoint& endpoint) const;

    // Wrong NUMA node roughly halves the ring's dependent-load rate.
    void set_numa_node(int node);

    ~RelayLoop();

private:
    RelayLoop() = default;
    void ensure_thread();
    void run();

    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<RelayEndpoint>> endpoints_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<int> numa_node_{-1};
};

}  // namespace tt::tt_metal::distributed::host_transport
