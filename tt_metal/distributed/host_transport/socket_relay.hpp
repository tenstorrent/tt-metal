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

#include "tt_metal/distributed/host_transport/rdma_link.hpp"

namespace tt::tt_metal::distributed {
class D2HSocket;
class H2DSocket;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::distributed::host_transport {

// Ring geometry shared by both ends of a relayed connection. Both FIFOs hold the
// same whole number of pages, which makes a page's source and destination index
// the same value: the wrap is one modulo and a batch is at most two contiguous
// runs. Requiring fifo_size % page_size == 0 also removes the FIFO tail gap that
// the socket protocol would otherwise have to charge to its counters.
struct RingGeometry {
    uint32_t page_size = 0;
    uint32_t num_pages = 0;

    uint64_t fifo_bytes() const { return static_cast<uint64_t>(page_size) * num_pages; }
};

// A 32-bit counter the peer RMAs in, read as a 64-bit monotonic value. The peer
// is never a full wrap ahead of what it has been credited, so the low 32 bits are
// enough to reconstruct the delta.
inline uint64_t widen(uint64_t observed, uint32_t wire_value) {
    return observed + static_cast<uint32_t>(wire_value - static_cast<uint32_t>(observed));
}

// One relayed direction. poll() never blocks and returns whether it progressed.
class RelayEndpoint {
public:
    virtual ~RelayEndpoint() = default;
    virtual bool poll() = 0;

    // A barrier is a watermark, not a quiesce: it snapshots what is outstanding
    // when it starts and waits for exactly that to drain. Waiting for the
    // endpoint to fall completely idle would never finish on a socket the peer
    // keeps feeding -- the far side pipelines ahead, which is the point.
    virtual uint64_t watermark() const = 0;
    virtual bool drained(uint64_t target) const = 0;

    // Counter state, for diagnosing a stall.
    virtual std::string describe() const = 0;

    // Set when poll() threw. The relay runs on its own thread, where an escaping
    // exception would terminate the process instead of failing the caller, so the
    // loop records it here and stops servicing this endpoint.
    const std::string& error() const { return error_; }
    bool failed() const { return !error_.empty(); }

protected:
    std::string error_;
    friend class RelayLoop;
};

// Device -> local host -> peer host. Drains the D2H socket's pinned ring straight
// onto the wire and retires pages once the NIC has finished reading them; that
// retirement is what frees ring space for the device kernel.
class RelaySender final : public RelayEndpoint {
public:
    RelaySender(
        RdmaChannel& channel,
        tt::tt_metal::distributed::D2HSocket& socket,
        RdmaRegion& fifo_region,
        RdmaRegion& credit_region,
        RingGeometry geometry,
        uint32_t max_batch_pages = 8);

    bool poll() override;
    uint64_t watermark() const override { return forwarded_.load(std::memory_order_acquire); }
    // Drained means: nothing of ours is still on the wire, and the relay's most
    // recent look at the local ring found nothing left to forward. The local
    // device stops producing before a barrier is called, so this converges;
    // unlike the receiver there is no peer racing ahead here.
    bool drained(uint64_t target) const override {
        return retired_.load(std::memory_order_acquire) >= target &&
               forwarded_.load(std::memory_order_acquire) == retired_.load(std::memory_order_acquire) &&
               unforwarded_.load(std::memory_order_acquire) == 0;
    }
    std::string describe() const override;

    uint64_t pages_forwarded() const { return forwarded_.load(std::memory_order_acquire); }
    uint64_t pages_retired() const { return retired_.load(std::memory_order_acquire); }

    // Samples the time from handing a batch to the NIC until the peer credits it,
    // i.e. the round trip through the far host and the far device. Combined with
    // an idle round-trip measurement this separates transit from queueing delay.
    // Off by default; it adds a timestamp per batch.
    void set_credit_latency_sampling(bool enabled);
    std::vector<uint64_t> take_credit_latencies_ns();

private:
    struct Batch {
        uint64_t wr_id;
        uint32_t pages;
    };

    uint64_t peer_credit();

    RdmaChannel& channel_;
    tt::tt_metal::distributed::D2HSocket& socket_;
    RdmaRegion& fifo_region_;
    RdmaRegion& credit_region_;
    RingGeometry geom_;
    uint32_t max_batch_pages_;
    std::deque<Batch> batches_;
    // Written only by the polling thread, read by a barrier on another thread.
    std::atomic<uint64_t> forwarded_{0};
    std::atomic<uint64_t> retired_{0};
    std::atomic<uint32_t> unforwarded_{0};  // pages seen available at the last poll
    uint64_t credited_ = 0;

    struct Sample {
        uint64_t total;  // forwarded total this batch reached
        uint64_t sent_ns;
    };
    bool sampling_ = false;
    std::deque<Sample> pending_samples_;
    std::vector<uint64_t> latencies_ns_;
    mutable std::mutex latency_mutex_;
};

// Peer host -> local host -> device. The payload is already in the H2D socket's
// pinned ring, placed there by the peer's NIC, so this only publishes it and
// returns credit.
class RelayReceiver final : public RelayEndpoint {
public:
    RelayReceiver(
        RdmaChannel& channel,
        tt::tt_metal::distributed::H2DSocket& socket,
        RdmaRegion& doorbell_region,
        RingGeometry geometry);

    bool poll() override;
    uint64_t watermark() const override { return arrived_.load(std::memory_order_acquire); }
    // A receiver has nothing host-side to drain. Publication into its ring is
    // driven by the peer, and the only thing left after publishing is the device
    // consuming -- which the owner controls by running its own program, not by
    // waiting here. Waiting for consumption deadlocks: the peer pipelines into
    // the next round, so a barrier would block on pages that only the caller's
    // *next* kernel launch will read.
    bool drained(uint64_t) const override { return true; }
    std::string describe() const override;

    uint64_t pages_arrived() const { return arrived_.load(std::memory_order_acquire); }

private:
    uint64_t peer_forwarded();

    RdmaChannel& channel_;
    tt::tt_metal::distributed::H2DSocket& socket_;
    RdmaRegion& doorbell_region_;
    RingGeometry geom_;
    // Written only by the polling thread, read by a barrier on another thread.
    std::atomic<uint64_t> arrived_{0};
    std::atomic<uint64_t> credited_pages_{0};
    uint64_t consumed_bytes_ = 0;
};

// Process-wide non-blocking event loop. One thread services every registered
// endpoint, so a socket imposes no polling duty on its owner. Tests that want
// determinism can skip the thread and call poll_once() themselves.
class RelayLoop {
public:
    static RelayLoop& instance();

    void add(const std::shared_ptr<RelayEndpoint>& endpoint);
    void remove(const std::shared_ptr<RelayEndpoint>& endpoint);

    // Drive one sweep on the caller's thread. Returns whether anything progressed.
    // Serialized against add()/remove(), so a removed endpoint is never polled
    // again once remove() has returned.
    bool poll_once();

    // First error recorded by any endpoint, empty if none have failed.
    std::string first_error() const;

    // Pin the loop thread to the NUMA node the device's FIFOs are bound to.
    // Walked from the wrong node, the ring's dependent loads run at roughly half
    // the rate and the FIFOs back up.
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
