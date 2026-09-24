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
#include <tt_stl/tt_pause.hpp>

namespace tt::tt_metal::distributed::host_transport {

namespace {

constexpr uint32_t kSpinsBeforeSleep = 1000;
constexpr uint32_t kSleepCapUs = 50;

struct Backoff {
    uint32_t empty = 0;
    uint32_t sleep_us = 1;

    void idle() {
        if (++empty < kSpinsBeforeSleep) {
            ttsl::pause();
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

RelaySender::RelaySender(
    HostTransport& transport,
    tt::tt_metal::distributed::D2HSocket& socket,
    RingGeometry geometry,
    uint32_t max_batch_pages) :
    transport_(transport), socket_(socket), geom_(geometry), max_batch_pages_(std::max(1u, max_batch_pages)) {
    TT_FATAL(geom_.page_size != 0 && geom_.num_pages != 0, "RelaySender needs a non-empty ring geometry");
}

void RelaySender::set_credit_latency_sampling(bool enabled) {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    sampling_ = enabled;
    pending_samples_.clear();
}

std::vector<uint64_t> RelaySender::take_credit_latencies_ns() {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    return std::move(latencies_ns_);
}

bool RelaySender::poll() {
    PollMark mark{polls_};
    bool progress = false;
    transport_.poll();

    // pop() acks the device, which is what lets the kernel push further.
    const uint64_t released = transport_.pages_released();
    uint64_t retired = retired_.load(std::memory_order_relaxed);
    if (released > retired) {
        socket_.pop(static_cast<uint32_t>(released - retired));
        retired_.store(released, std::memory_order_release);
        retired = released;
        progress = true;
    }

    const uint64_t forwarded = forwarded_.load(std::memory_order_relaxed);
    const uint64_t credited = transport_.peer_consumed_pages();
    if (sampling_) {
        std::lock_guard<std::mutex> lock(latency_mutex_);
        const uint64_t now = now_ns();
        while (!pending_samples_.empty() && pending_samples_.front().total <= credited) {
            latencies_ns_.push_back(now - pending_samples_.front().sent_ns);
            pending_samples_.pop_front();
        }
    }

    const uint64_t in_peer_ring = forwarded - credited;
    if (in_peer_ring >= geom_.num_pages) {
        return progress;
    }
    const uint32_t room = geom_.num_pages - static_cast<uint32_t>(in_peer_ring);

    // pages_available() still counts pages already handed to the transport, since
    // the ack only happens in pop() above. Subtract them or they go out twice.
    const uint32_t available = socket_.pages_available();
    const uint64_t awaiting_pop = forwarded - retired;
    // For barrier: is anything still waiting to go out.
    unforwarded_.store(
        available > awaiting_pop ? static_cast<uint32_t>(available - awaiting_pop) : 0u, std::memory_order_release);
    if (available <= awaiting_pop) {
        return progress;
    }

    const uint32_t n = std::min({static_cast<uint32_t>(available - awaiting_pop), room, max_batch_pages_});
    if (n == 0 || !transport_.can_send(n)) {
        return progress;
    }
    if (!transport_.send(forwarded, n)) {
        return progress;
    }

    const uint64_t total = forwarded + n;
    forwarded_.store(total, std::memory_order_release);
    unforwarded_.store(static_cast<uint32_t>(available - awaiting_pop - n), std::memory_order_release);
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
        "sender forwarded={} retired={} unforwarded={} ring_pages={} | {}",
        forwarded_.load(std::memory_order_acquire),
        retired_.load(std::memory_order_acquire),
        unforwarded_.load(std::memory_order_acquire),
        geom_.num_pages,
        transport_.describe());
}

RelayReceiver::RelayReceiver(
    HostTransport& transport, tt::tt_metal::distributed::H2DSocket& socket, RingGeometry geometry) :
    transport_(transport), socket_(socket), geom_(geometry) {
    TT_FATAL(geom_.page_size != 0 && geom_.num_pages != 0, "RelayReceiver needs a non-empty ring geometry");
}

bool RelayReceiver::poll() {
    PollMark mark{polls_};
    bool progress = false;

    // Refresh first: commit_pages() bounds against the cached bytes_acked, and a
    // stale value rejects a legal commit.
    const uint32_t acked_wire = socket_.bytes_acked_snapshot();
    consumed_bytes_ += acked_wire - last_acked_wire_;
    last_acked_wire_ = acked_wire;
    const uint64_t consumed_pages = consumed_bytes_ / geom_.page_size;
    // A two-sided transport needs this to know which ring pages it may receive
    // into again; a one-sided one only uses it to bound a sanity check.
    transport_.set_consumed(consumed_pages);

    transport_.poll();

    const uint64_t arrived = arrived_.load(std::memory_order_relaxed);
    const uint64_t delivered = transport_.pages_delivered();
    if (delivered > arrived) {
        const uint32_t pages = static_cast<uint32_t>(delivered - arrived);
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
        arrived_.store(delivered, std::memory_order_release);
        progress = true;
    }

    if (consumed_pages != credited_pages_.load(std::memory_order_relaxed) && transport_.post_credit(consumed_pages)) {
        credited_pages_.store(consumed_pages, std::memory_order_release);
        progress = true;
    }
    return progress;
}

std::string RelayReceiver::describe() const {
    return fmt::format(
        "receiver arrived={} credited={} consumed_bytes={} ring_pages={} | {}",
        arrived_.load(std::memory_order_acquire),
        credited_pages_.load(std::memory_order_acquire),
        consumed_bytes_,
        geom_.num_pages,
        transport_.describe());
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
            endpoint->set_error(e.what());
            log_error(tt::LogDistributed, "host_transport relay endpoint stopped: {}", e.what());
        }
    }
    return progress;
}

std::string RelayLoop::describe(const RelayEndpoint& endpoint) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return endpoint.describe();
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
