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

// Batches outstanding on the wire before the sender stops issuing. Bounds the
// bookkeeping deque; the peer ring credit is the real limit.
constexpr size_t kMaxOutstandingBatches = 64;

// Send-queue slots one batch can consume: two payload runs plus the doorbell.
constexpr uint32_t kSlotsPerBatch = 3;

// Empty sweeps to spin before parking, then a sleep growing to this cap.
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

// Bounds the sampler's bookkeeping; credits normally land within a few batches.
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

uint64_t RelaySender::peer_credit() {
    uint32_t wire = *static_cast<volatile uint32_t*>(credit_region_.addr());
    credited_ = widen(credited_, wire);
    return credited_;
}

bool RelaySender::poll() {
    bool progress = false;

    // Retire batches the NIC has finished reading. pop() advances the D2H ring
    // and acks the device, which is what lets the kernel push further.
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

    // Peer ring space, in pages. The credit is an absolute count, so a stale or
    // duplicated update only costs a lap of latency.
    const uint64_t forwarded = forwarded_.load(std::memory_order_relaxed);
    const uint64_t credited = peer_credit();
    if (sampling_) {
        std::lock_guard<std::mutex> lock(latency_mutex_);
        const uint64_t now = now_ns();
        while (!pending_samples_.empty() && pending_samples_.front().total <= credited) {
            latencies_ns_.push_back(now - pending_samples_.front().sent_ns);
            pending_samples_.pop_front();
        }
    }
    uint64_t in_peer_ring = forwarded - credited;
    if (in_peer_ring >= geom_.num_pages) {
        return progress;
    }
    uint32_t room = geom_.num_pages - static_cast<uint32_t>(in_peer_ring);

    // pages_available() counts everything the device has produced and the host
    // has not yet acked, and the ack only happens in pop() above. Pages already
    // handed to the NIC are therefore still counted, so subtract them or they get
    // sent twice.
    const uint32_t available = socket_.pages_available();
    const uint64_t awaiting_pop = forwarded - retired_.load(std::memory_order_relaxed);
    // What a barrier needs to know: whether anything is still waiting to go out.
    unforwarded_.store(
        available > awaiting_pop ? static_cast<uint32_t>(available - awaiting_pop) : 0u, std::memory_order_release);
    if (available <= awaiting_pop) {
        return progress;
    }
    uint32_t n = std::min({static_cast<uint32_t>(available - awaiting_pop), room, max_batch_pages_});
    if (n == 0) {
        return progress;
    }

    // Source and destination page indices are equal by construction, so one
    // modulo serves both and the batch is at most two contiguous runs.
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
    // Ordered doorbell: RC delivery puts it behind the payload above, so the peer
    // cannot observe it before those bytes have landed. It carries the absolute
    // running total, so the peer derives the delta itself and a duplicate is a
    // no-op.
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
        "sender forwarded={} retired={} credited={} batches={} ring_pages={} unforwarded={} send_slots={}",
        forwarded_.load(std::memory_order_acquire),
        retired_.load(std::memory_order_acquire),
        credited_,
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

uint64_t RelayReceiver::peer_forwarded() {
    const uint32_t wire = *static_cast<volatile uint32_t*>(doorbell_region_.addr());
    return widen(arrived_.load(std::memory_order_relaxed), wire);
}

bool RelayReceiver::poll() {
    bool progress = false;

    // Refresh the socket's cached bytes_acked first: commit_pages() bounds the
    // publish against it, and a stale value would reject a legal commit.
    consumed_bytes_ = widen(consumed_bytes_, socket_.bytes_acked_snapshot());

    // The doorbell slot holds the peer's absolute forwarded-page total, written by
    // its NIC behind the payload it describes. Reading it is a local load, and
    // because it is absolute a repeated or coalesced update cannot inflate the
    // count. The peer cannot have more than a ring's worth outstanding, so
    // anything larger means the slot was misread rather than that the peer really
    // got that far ahead.
    const uint64_t arrived = arrived_.load(std::memory_order_relaxed);
    const uint64_t forwarded_total = peer_forwarded();
    TT_FATAL(
        forwarded_total >= arrived && forwarded_total - arrived <= geom_.num_pages,
        "implausible doorbell: peer total {} against {} published, ring holds {} pages",
        forwarded_total,
        arrived,
        geom_.num_pages);
    const uint32_t pages = static_cast<uint32_t>(forwarded_total - arrived);
    if (pages != 0) {
        // The payload is already resident in the pinned ring; publishing it is a
        // counter advance plus one 4-byte write into device L1.
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

    // Credit is an absolute page count. bytes_acked is a wrapping 32-bit byte
    // counter and the page size need not divide 2^32, so widen before dividing.
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
    // Held for the whole sweep, not just to copy the list: an endpoint only
    // borrows its socket, so once remove() returns the owner is free to destroy
    // that socket. Sweeping a snapshot outside the lock would let this thread
    // poll it afterwards.
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
    // The default 50us timer slack would round every probe sleep up to the cap.
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
