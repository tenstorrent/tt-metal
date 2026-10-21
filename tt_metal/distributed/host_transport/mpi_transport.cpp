// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_transport/mpi_transport.hpp"

#include <tt-metalium/distributed_context.hpp>
#include <tt_stl/assert.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <deque>

namespace tt::tt_metal::distributed::host_transport {

namespace {

// One message per contiguous run of pages, never across the ring wrap, so no
// framing is needed: point-to-point delivery between a rank pair with one tag is
// ordered, so the receiver probes each arrived message's size, posts a receive of
// exactly that length at the next page, and the message lands where the sender
// took it from. Both sides agree on page index without putting it on the wire,
// and any number of messages can be outstanding on both sides.
//
// The batching is whatever the relay found ready at this poll, up to the end of
// the ring: one page goes out on its own, and when the link falls behind the
// pages that pile up meanwhile go out together. Nothing is held back waiting for
// company.
constexpr int kPayloadTagOffset = 0;
constexpr int kCreditTagOffset = 1;

}  // namespace

MpiTransport::MpiTransport(const TransportParams& params) :
    geom_(params.geometry),
    is_sender_(params.is_sender),
    ring_(static_cast<std::byte*>(params.ring)),
    context_(params.context),
    peer_(multihost::Rank{params.peer_rank}),
    payload_tag_(multihost::Tag{params.tag_base + kPayloadTagOffset}),
    credit_tag_(multihost::Tag{params.tag_base + kCreditTagOffset}) {
    TT_FATAL(context_ != nullptr, "MpiTransport needs a DistributedContext");
    TT_FATAL(ring_ != nullptr, "MpiTransport needs a ring");
    TT_FATAL(geom_.page_size != 0 && geom_.num_pages != 0, "MpiTransport needs a ring geometry");

    warm_ring();
    if (is_sender_) {
        // One standing receive for the peer's credit; re-posted on completion.
        arm_credit_recv();
    } else {
        post_receives();
    }
}

void MpiTransport::warm_ring() {
    // One message the size of the whole ring, sender to receiver, before any page: the data
    // is meaningless and no counter moves, but it leaves each side's ring registered with
    // the NIC as a single region. Registered piecemeal instead, as the first lap of real
    // pages would do, UCX's registration cache merges each new page into the growing
    // region by re-registering the whole thing, and with a ring of megabytes that is
    // ~150 us on every message for as long as the socket lives. Measured here: 176 us per
    // 14 KiB round trip without this, 26 us with the cache bypassed.
    auto whole = std::span<std::byte>(ring_, geom_.fifo_bytes());
    multihost::RequestPtr r =
        is_sender_ ? context_->isend(whole, peer_, payload_tag_) : context_->irecv(whole, peer_, payload_tag_);
    (void)r->wait();
}

std::span<std::byte> MpiTransport::run_span(uint64_t first, uint32_t pages) const {
    const uint64_t slot = first % geom_.num_pages;
    TT_ASSERT(slot + pages <= geom_.num_pages, "run of {} pages from slot {} crosses the ring wrap", pages, slot);
    return std::span<std::byte>(ring_ + slot * geom_.page_size, static_cast<size_t>(pages) * geom_.page_size);
}

void MpiTransport::arm_credit_recv() {
    credit_recv_ = context_->irecv(
        std::span<std::byte>(reinterpret_cast<std::byte*>(&credit_inbox_), sizeof(credit_inbox_)), peer_, credit_tag_);
}

void MpiTransport::post_receives() {
    // A receive per message whose envelope has arrived, each exactly the message's
    // length at the next ring page. An eager message is already buffered and
    // copies on match; a rendezvous one starts its transfer on match. Either way
    // nothing waits for an earlier message's data, so the link stays full.
    while (pending_recv_.size() < kMaxOutstanding) {
        const auto bytes = context_->try_snoop_incoming_msg_size(peer_, payload_tag_);
        if (!bytes.has_value()) {
            return;
        }
        TT_FATAL(
            *bytes != 0 && *bytes % geom_.page_size == 0,
            "mpi receiver sees a {} B message on a {} B page ring",
            *bytes,
            geom_.page_size);
        const auto pages = static_cast<uint32_t>(*bytes / geom_.page_size);
        const uint32_t slot = static_cast<uint32_t>(posted_ % geom_.num_pages);
        TT_FATAL(
            slot + pages <= geom_.num_pages && posted_ + pages <= consumed_ + geom_.num_pages,
            "mpi receiver: {}-page message at slot {} of {} (posted={} consumed={}) breaks the ring",
            pages,
            slot,
            geom_.num_pages,
            posted_,
            consumed_);
        pending_recv_.emplace_back(context_->irecv(run_span(posted_, pages), peer_, payload_tag_), pages);
        posted_ += pages;
    }
}

bool MpiTransport::can_send(uint32_t /*pages*/) const {
    // A batch that straddles the wrap is two messages; let it through whole.
    return pending_send_.size() + 2 <= kMaxOutstanding;
}

bool MpiTransport::send(uint64_t first_page, uint32_t pages) {
    TT_ASSERT(is_sender_, "send() on a receiving transport");
    // Both sides derive the destination page from their own counter, so a caller
    // that skips or repeats a page would silently misplace every later one.
    TT_ASSERT(first_page == sent_, "send() expects the absolute page index: got {}, sent {}", first_page, sent_);
    if (pages == 0 || !can_send(pages)) {
        return false;
    }
    uint64_t next = first_page;
    uint32_t left = pages;
    while (left > 0) {
        const uint32_t slot = static_cast<uint32_t>(next % geom_.num_pages);
        const uint32_t run = std::min(left, geom_.num_pages - slot);
        pending_send_.emplace_back(context_->isend(run_span(next, run), peer_, payload_tag_), run);
        next += run;
        left -= run;
    }
    sent_ += pages;
    return true;
}

// A request only becomes inactive once it has been tested, so this is what
// actually retires the credit send rather than just observing it.
bool MpiTransport::retire_credit_send() {
    if (credit_send_ && credit_send_->test().has_value()) {
        credit_send_.reset();
    }
    return credit_send_ == nullptr;
}

bool MpiTransport::post_credit(uint64_t consumed_pages) {
    TT_ASSERT(!is_sender_, "post_credit() on a sending transport");
    // Absolute, so dropping one only costs latency: keep a single send in flight
    // and skip while it is outstanding.
    if (!retire_credit_send()) {
        return false;
    }
    credit_outbox_ = consumed_pages;
    credit_send_ = context_->isend(
        std::span<std::byte>(reinterpret_cast<std::byte*>(&credit_outbox_), sizeof(credit_outbox_)),
        peer_,
        credit_tag_);
    return true;
}

void MpiTransport::set_consumed(uint64_t consumed_pages) { consumed_ = consumed_pages; }

void MpiTransport::poll() {
    if (is_sender_) {
        // Completion order is not guaranteed, so retire from the front only: the
        // count stays monotonic and never claims a page is free early.
        while (!pending_send_.empty() && pending_send_.front().first->test().has_value()) {
            released_ += pending_send_.front().second;
            pending_send_.pop_front();
        }
        if (credit_recv_ && credit_recv_->test().has_value()) {
            peer_consumed_ = std::max(peer_consumed_, credit_inbox_);
            arm_credit_recv();
        }
    } else {
        post_receives();
        // Front-retired only: delivery order is the page order.
        while (!pending_recv_.empty() && pending_recv_.front().first->test().has_value()) {
            delivered_ += pending_recv_.front().second;
            pending_recv_.pop_front();
        }
        retire_credit_send();
    }
}

std::string MpiTransport::describe() const {
    return is_sender_ ? fmt::format(
                            "mpi sender sent={} released={} peer_consumed={} in_flight={}",
                            sent_,
                            released_,
                            peer_consumed_,
                            pending_send_.size())
                      : fmt::format(
                            "mpi receiver delivered={} posted={} consumed={} recv_outstanding={}",
                            delivered_,
                            posted_,
                            consumed_,
                            pending_recv_.size());
}

MpiTransport::~MpiTransport() {
    // Point-to-point requests can be cancelled, unlike MPI RMA ones, so a
    // half-finished stream tears down instead of leaking.
    //
    // Known gap: DistributedContext::cancel() is MPI_Cancel followed by
    // MPI_Request_free, which does not establish completion -- MPI may still
    // touch a buffer after this returns. The ring outlives us (the socket that
    // owns it is destroyed after the transport), but credit_inbox_/outbox_ are
    // members and do not. Closing this needs cancel-then-wait in the shared
    // request abstraction, which is outside this change; see the PR thread.
    for (auto& [r, pages] : pending_send_) {
        r->cancel();
    }
    for (auto& [r, pages] : pending_recv_) {
        r->cancel();
    }
    if (credit_recv_) {
        credit_recv_->cancel();
    }
    if (credit_send_) {
        credit_send_->cancel();
    }
}

}  // namespace tt::tt_metal::distributed::host_transport
