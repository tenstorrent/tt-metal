// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_transport/socket_relay.hpp"

#include <algorithm>
#include <chrono>

#include <sys/prctl.h>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::distributed::host_transport {

namespace {

// Bounds the bookkeeping deque; peer ring credit is the real limit.
constexpr size_t kMaxOutstandingBatches = 64;

// Two payload runs plus the doorbell.
constexpr uint32_t kSlotsPerBatch = 3;

constexpr uint32_t kSpinsBeforeSleep = 1000;
constexpr uint32_t kSleepCapUs = 50;

struct Backoff {
    uint32_t empty = 0;
    uint32_t sleep_us = 1;

    void idle() {
        if (++empty < kSpinsBeforeSleep) {
            __builtin_ia32_pause();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            sleep_us = std::min(sleep_us + sleep_us / 4 + 1, kSleepCapUs);
        }
    }
    void reset() {
        empty = 0;
        sleep_us = 1;
    }
};

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

constexpr size_t kMaxPendingSamples = 4096;

}  // namespace

void RelaySender::set_credit_latency_sampling(bool enabled) {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    sampling_ = enabled;
    pending_samples_.clear();
}

std::vector<uint64_t> RelaySender::take_credit_latencies_ns() {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    return std::move(latencies_ns_);
}

RelaySender::RelaySender(
    RdmaChannel& channel,
    tt::tt_metal::distributed::D2HSocket& socket,
    RdmaRegion& fifo_region,
    RdmaRegion& credit_region,
    RingGeometry geometry,
    uint32_t max_batch_pages) :
    channel_(channel),
    socket_(socket),
    fifo_region_(fifo_region),
    credit_region_(credit_region),
    geom_(geometry),
    max_batch_pages_(std::max(1u, max_batch_pages)) {
    TT_FATAL(geom_.page_size != 0 && geom_.num_pages != 0, "RelaySender needs a non-empty ring geometry");
    TT_FATAL(
        credit_region_.len() >= sizeof(uint32_t), "credit region is {} bytes, need at least 4", credit_region_.len());
    *static_cast<volatile uint32_t*>(credit_region_.addr()) = 0;
}

uint32_t RelaySender::peer_credit_wire() const {
    return *static_cast<volatile uint32_t*>(credit_region_.addr());
}

bool RelaySender::poll() {
    bool progress = false;

    // pop() acks the device, which is what lets the kernel push further.
    channel_.poll_send();
    while (!batches_.empty() && channel_.reaped() > batches_.front().wr_id) {
        socket_.pop(batches_.front().pages);
        retired_.store(retired_.load(std::memory_order_relaxed) + batches_.front().pages, std::memory_order_release);
        batches_.pop_front();
        progress = true;
    }

    if (batches_.size() >= kMaxOutstandingBatches || channel_.send_slots_available() < kSlotsPerBatch) {
        return progress;
    }

    const uint64_t forwarded = forwarded_.load(std::memory_order_relaxed);
    // Both counts are page totals and differ by at most a ring, so the subtraction
    // is exact in 32-bit modular arithmetic even once the peer's counter wraps.
    const uint32_t in_peer_ring = static_cast<uint32_t>(forwarded) - peer_credit_wire();
    if (sampling_) {
        std::lock_guard<std::mutex> lock(latency_mutex_);
        const uint64_t now = now_ns();
        const uint64_t credited = forwarded - in_peer_ring;
        while (!pending_samples_.empty() && pending_samples_.front().total <= credited) {
            latencies_ns_.push_back(now - pending_samples_.front().sent_ns);
            pending_samples_.pop_front();
        }
    }
    if (in_peer_ring >= geom_.num_pages) {
        return progress;
    }
    uint32_t room = geom_.num_pages - in_peer_ring;

    // pages_available() still counts pages already handed to the NIC, since the
    // ack only happens in pop() above. Subtract them or they go out twice.
    const uint32_t available = socket_.pages_available();
    const uint64_t awaiting_pop = forwarded - retired_.load(std::memory_order_relaxed);
    // For barrier: is anything still waiting to go out.
    unforwarded_.store(
        available > awaiting_pop ? static_cast<uint32_t>(available - awaiting_pop) : 0u, std::memory_order_release);
    if (available <= awaiting_pop) {
        return progress;
    }
    uint32_t n = std::min({static_cast<uint32_t>(available - awaiting_pop), room, max_batch_pages_});
    if (n == 0) {
        return progress;
    }

    // Source and destination page indices are equal, so one modulo serves both.
    uint32_t index = static_cast<uint32_t>(forwarded % geom_.num_pages);
    uint32_t head_pages = std::min(n, geom_.num_pages - index);
    uint64_t head_offset = static_cast<uint64_t>(index) * geom_.page_size;

    TT_FATAL(
        channel_.post_write(fifo_region_, head_offset, head_offset, head_pages * geom_.page_size),
        "send queue full after the slot check");
    if (n > head_pages) {
        TT_FATAL(
            channel_.post_write(fifo_region_, 0, 0, (n - head_pages) * geom_.page_size), "send queue full mid-batch");
    }
    // RC puts the doorbell behind the payload above, so the peer cannot see it
    // before those bytes land.
    const uint64_t total = forwarded + n;
    forwarded_.store(total, std::memory_order_release);
    unforwarded_.store(static_cast<uint32_t>(available - awaiting_pop - n), std::memory_order_release);
    TT_FATAL(channel_.post_doorbell(static_cast<uint32_t>(total)), "send queue full before the doorbell");

    batches_.push_back(Batch{.wr_id = channel_.last_posted_id(), .pages = n});
    if (sampling_) {
        std::lock_guard<std::mutex> lock(latency_mutex_);
        if (pending_samples_.size() < kMaxPendingSamples) {
            pending_samples_.push_back(Sample{.total = total, .sent_ns = now_ns()});
        }
    }
    return true;
}

std::string RelaySender::describe() const {
    return fmt::format(
        "sender forwarded={} retired={} credit_wire={} batches={} ring_pages={} unforwarded={} send_slots={}",
        forwarded_.load(std::memory_order_acquire),
        retired_.load(std::memory_order_acquire),
        peer_credit_wire(),
        batches_.size(),
        geom_.num_pages,
        unforwarded_.load(std::memory_order_acquire),
        channel_.send_slots_available());
}

RelayReceiver::RelayReceiver(
    RdmaChannel& channel,
    tt::tt_metal::distributed::H2DSocket& socket,
    RdmaRegion& doorbell_region,
    RingGeometry geometry) :
    channel_(channel), socket_(socket), doorbell_region_(doorbell_region), geom_(geometry) {
    TT_FATAL(geom_.page_size != 0 && geom_.num_pages != 0, "RelayReceiver needs a non-empty ring geometry");
    TT_FATAL(
        doorbell_region_.len() >= sizeof(uint32_t),
        "doorbell region is {} bytes, need at least 4",
        doorbell_region_.len());
    *static_cast<volatile uint32_t*>(doorbell_region_.addr()) = 0;
}

bool RelayReceiver::poll() {
    bool progress = false;

    // Refresh first: commit_pages() bounds against the cached bytes_acked, and a
    // stale value rejects a legal commit.
    const uint32_t acked_wire = socket_.bytes_acked_snapshot();
    consumed_bytes_ += acked_wire - last_acked_wire_;
    last_acked_wire_ = acked_wire;

    const uint64_t arrived = arrived_.load(std::memory_order_relaxed);
    // Exact in 32-bit modular arithmetic: the peer is at most a ring ahead. More
    // than that means the slot was misread, not real progress.
    const uint32_t pages =
        *static_cast<volatile uint32_t*>(doorbell_region_.addr()) - static_cast<uint32_t>(arrived);
    TT_FATAL(
        pages <= geom_.num_pages,
        "implausible doorbell: {} pages against {} published, ring holds {}",
        pages,
        arrived,
        geom_.num_pages);
    if (pages != 0) {
        try {
            socket_.commit_pages(pages);
        } catch (const std::exception& e) {
            TT_THROW(
                "relay receiver failed to publish {} page(s): arrived={} credited={} consumed_bytes={} "
                "ring_pages={} page_size={}\n  {}",
                pages,
                arrived,
                credited_pages_.load(std::memory_order_relaxed),
                consumed_bytes_,
                geom_.num_pages,
                geom_.page_size,
                e.what());
        }
        arrived_.store(arrived + pages, std::memory_order_release);
        progress = true;
    }

    const uint64_t consumed_pages = consumed_bytes_ / geom_.page_size;
    if (consumed_pages != credited_pages_.load(std::memory_order_relaxed) &&
        channel_.post_credit(static_cast<uint32_t>(consumed_pages))) {
        credited_pages_.store(consumed_pages, std::memory_order_release);
        progress = true;
    }

    channel_.poll_send();
    return progress;
}

std::string RelayReceiver::describe() const {
    return fmt::format(
        "receiver arrived={} credited={} consumed_bytes={} doorbell_slot={} ring_pages={}",
        arrived_.load(std::memory_order_acquire),
        credited_pages_.load(std::memory_order_acquire),
        consumed_bytes_,
        *static_cast<volatile uint32_t*>(doorbell_region_.addr()),
        geom_.num_pages);
}

RelayLoop& RelayLoop::instance() {
    static RelayLoop loop;
    return loop;
}

RelayLoop::~RelayLoop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void RelayLoop::add(const std::shared_ptr<RelayEndpoint>& endpoint) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        endpoints_.push_back(endpoint);
    }
    ensure_thread();
}

void RelayLoop::remove(const std::shared_ptr<RelayEndpoint>& endpoint) {
    std::lock_guard<std::mutex> lock(mutex_);
    endpoints_.erase(std::remove(endpoints_.begin(), endpoints_.end(), endpoint), endpoints_.end());
}

void RelayLoop::set_numa_node(int node) { numa_node_.store(node, std::memory_order_release); }

bool RelayLoop::poll_once() {
    // Held for the whole sweep: an endpoint only borrows its socket, so once
    // remove() returns the owner may destroy it.
    std::lock_guard<std::mutex> lock(mutex_);
    bool progress = false;
    for (auto& endpoint : endpoints_) {
        if (endpoint->failed()) {
            continue;
        }
        try {
            progress |= endpoint->poll();
        } catch (const std::exception& e) {
            endpoint->error_ = e.what();
            log_error(tt::LogDistributed, "host_transport relay endpoint stopped: {}", e.what());
        }
    }
    return progress;
}

std::string RelayLoop::first_error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& endpoint : endpoints_) {
        if (endpoint->failed()) {
            return endpoint->error();
        }
    }
    return {};
}

void RelayLoop::ensure_thread() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
    }
    thread_ = std::thread([this] { run(); });
}

void RelayLoop::run() {
    prctl(PR_SET_NAME, "tt-host-relay", 0, 0, 0);
    // Default 50us slack would round every probe sleep up to the cap.
    prctl(PR_SET_TIMERSLACK, 1000, 0, 0, 0);

    Backoff backoff;
    while (running_.load(std::memory_order_acquire)) {
        // Drain fully before parking: one sweep can unblock the next.
        bool any = false;
        for (bool progress = true; progress;) {
            progress = poll_once();
            any |= progress;
        }
        if (any) {
            backoff.reset();
        } else {
            backoff.idle();
        }
    }
}

}  // namespace tt::tt_metal::distributed::host_transport
