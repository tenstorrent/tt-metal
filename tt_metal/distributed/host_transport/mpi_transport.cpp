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

// One message per page, so no framing is needed and a message never spans the
// ring wrap. Point-to-point delivery between a rank pair with one tag is ordered,
// so the receiver's queued receives fill in send order and each lands on the page
// the sender took it from -- which is what lets both sides agree on page index
// without putting it on the wire.
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

    if (is_sender_) {
        // One standing receive for the peer's credit; re-posted on completion.
        arm_credit_recv();
    } else {
        // Queue a receive for every page the ring can hold. The sender is credit
        // gated to at most a ring ahead, so a receive is always already waiting.
        top_up_receives();
    }
}

std::span<std::byte> MpiTransport::page_span(uint64_t page_index) const {
    const uint64_t slot = page_index % geom_.num_pages;
    return {ring_ + slot * geom_.page_size, geom_.page_size};
}

void MpiTransport::arm_credit_recv() {
    credit_recv_ = context_->irecv(
        std::span<std::byte>(reinterpret_cast<std::byte*>(&credit_inbox_), sizeof(credit_inbox_)), peer_, credit_tag_);
}

void MpiTransport::top_up_receives() {
    // Only up to the consumed watermark plus a ring: receiving into a page the
    // device has not read yet would overwrite live data.
    const uint64_t limit = consumed_ + geom_.num_pages;
    while (posted_ < limit) {
        pending_recv_.push_back(context_->irecv(page_span(posted_), peer_, payload_tag_));
        posted_++;
    }
}

bool MpiTransport::can_send(uint32_t pages) const { return pending_send_.size() + pages <= kMaxOutstanding; }

bool MpiTransport::send(uint64_t first_page, uint32_t pages) {
    TT_ASSERT(is_sender_, "send() on a receiving transport");
    // Both sides derive the destination page from their own counter, so a caller
    // that skips or repeats a page would silently misplace every later one.
    TT_ASSERT(first_page == sent_, "send() expects the absolute page index: got {}, sent {}", first_page, sent_);
    if (!can_send(pages)) {
        return false;
    }
    for (uint32_t i = 0; i < pages; i++) {
        pending_send_.push_back(context_->isend(page_span(first_page + i), peer_, payload_tag_));
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

void MpiTransport::set_consumed(uint64_t consumed_pages) {
    consumed_ = consumed_pages;
    if (!is_sender_) {
        top_up_receives();
    }
}

void MpiTransport::poll() {
    if (is_sender_) {
        // Completion order is not guaranteed, so retire from the front only: the
        // count stays monotonic and never claims a page is free early.
        while (!pending_send_.empty() && pending_send_.front()->test().has_value()) {
            pending_send_.pop_front();
            released_++;
        }
        if (credit_recv_ && credit_recv_->test().has_value()) {
            peer_consumed_ = std::max(peer_consumed_, credit_inbox_);
            arm_credit_recv();
        }
    } else {
        while (!pending_recv_.empty() && pending_recv_.front()->test().has_value()) {
            pending_recv_.pop_front();
            delivered_++;
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
                            "mpi receiver delivered={} posted={} consumed={} queued_recv={}",
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
    for (auto& r : pending_send_) {
        r->cancel();
    }
    for (auto& r : pending_recv_) {
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
