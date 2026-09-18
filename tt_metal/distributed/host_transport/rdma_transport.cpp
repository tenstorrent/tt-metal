// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_transport/rdma_transport.hpp"

#include <tt-metalium/distributed_context.hpp>
#include <tt_stl/assert.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <mutex>

namespace tt::tt_metal::distributed::host_transport {

namespace {

// One context per process; device and protection domain are shared by every
// channel, completion queues stay per-channel.
RdmaContext& shared_context(const std::string& device, int gid_index) {
    static std::unique_ptr<RdmaContext> ctx;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (ctx == nullptr) {
        ctx = std::make_unique<RdmaContext>(RdmaContext::Config{.device_name = device, .gid_index = gid_index});
    }
    return *ctx;
}

std::span<std::byte> as_bytes(RdmaEndpoint& e) { return {reinterpret_cast<std::byte*>(&e), sizeof(RdmaEndpoint)}; }

// Extend a wrapping 32-bit peer counter. The peer is never a full wrap ahead.
uint64_t advance(uint64_t observed, uint32_t wire) {
    return observed + static_cast<uint32_t>(wire - static_cast<uint32_t>(observed));
}

}  // namespace

RdmaTransport::RdmaTransport(const TransportParams& params) :
    geom_(params.geometry),
    is_sender_(params.is_sender),
    ctx_(shared_context(params.rdma_device, params.gid_index)),
    max_batch_pages_(std::max(1u, params.max_batch_pages)) {
    TT_FATAL(params.context != nullptr, "RdmaTransport needs a DistributedContext for the handshake");
    TT_FATAL(params.ring != nullptr, "RdmaTransport needs a ring");

    channel_ = std::make_unique<RdmaChannel>(ctx_);
    ring_region_ = std::make_unique<RdmaRegion>(ctx_, params.ring, geom_.fifo_bytes());
    slot_ = std::make_unique<uint32_t>(0);
    slot_region_ = std::make_unique<RdmaRegion>(ctx_, slot_.get(), sizeof(uint32_t));

    RdmaEndpoint local = channel_->local_endpoint();
    if (is_sender_) {
        local.credit = slot_region_->descriptor();  // peer writes what its device consumed
    } else {
        local.fifo = ring_region_->descriptor();
        local.doorbell = slot_region_->descriptor();
    }

    // Opposite order on the two sides so neither blocks on an unread send.
    const multihost::Rank peer{params.peer_rank};
    const multihost::Tag tag{params.tag_base};
    RdmaEndpoint remote{};
    if (is_sender_) {
        params.context->send(as_bytes(local), peer, tag);
        params.context->recv(as_bytes(remote), peer, tag);
        TT_FATAL(
            remote.fifo.len == geom_.fifo_bytes(),
            "peer ring is {} bytes, local is {}: both must hold the same page count",
            remote.fifo.len,
            geom_.fifo_bytes());
        TT_FATAL(remote.doorbell.len >= sizeof(uint32_t), "peer advertised no doorbell slot");
    } else {
        params.context->recv(as_bytes(remote), peer, tag);
        params.context->send(as_bytes(local), peer, tag);
    }
    channel_->connect(remote);
}

RdmaTransport::~RdmaTransport() = default;

bool RdmaTransport::can_send(uint32_t pages) const {
    return pages <= max_batch_pages_ && batches_.size() < kMaxOutstandingBatches &&
           channel_->send_slots_available() >= kSlotsPerBatch;
}

bool RdmaTransport::send(uint32_t first_page, uint32_t pages) {
    TT_ASSERT(is_sender_, "send() on a receiving transport");
    if (!can_send(pages)) {
        return false;
    }
    // Source and destination page indices are equal, so one modulo serves both.
    const uint32_t index = first_page % geom_.num_pages;
    const uint32_t head = std::min(pages, geom_.num_pages - index);
    const uint64_t offset = static_cast<uint64_t>(index) * geom_.page_size;

    TT_FATAL(
        channel_->post_write(*ring_region_, offset, offset, head * geom_.page_size),
        "send queue full after the slot check");
    if (pages > head) {
        TT_FATAL(
            channel_->post_write(*ring_region_, 0, 0, (pages - head) * geom_.page_size), "send queue full mid-batch");
    }
    sent_ += pages;
    // RC puts the doorbell behind the payload above, so the peer cannot see it
    // before those bytes land.
    TT_FATAL(channel_->post_doorbell(static_cast<uint32_t>(sent_)), "send queue full before the doorbell");
    batches_.push_back(Batch{.wr_id = channel_->last_posted_id(), .pages = pages});
    return true;
}

bool RdmaTransport::post_credit(uint64_t consumed_pages) {
    TT_ASSERT(!is_sender_, "post_credit() on a sending transport");
    return channel_->post_credit(static_cast<uint32_t>(consumed_pages));
}

void RdmaTransport::poll() {
    channel_->poll_send();
    if (is_sender_) {
        while (!batches_.empty() && channel_->reaped() > batches_.front().wr_id) {
            released_ += batches_.front().pages;
            batches_.pop_front();
        }
        peer_consumed_ = advance(peer_consumed_, *static_cast<volatile uint32_t*>(slot_region_->addr()));
    } else {
        delivered_ = advance(delivered_, *static_cast<volatile uint32_t*>(slot_region_->addr()));
        // More than a ring outstanding means the slot was misread, not that the
        // peer really got that far ahead.
        TT_FATAL(
            delivered_ - consumed_ <= geom_.num_pages,
            "implausible doorbell: {} delivered, {} consumed, ring holds {}",
            delivered_,
            consumed_,
            geom_.num_pages);
    }
}

std::string RdmaTransport::describe() const {
    return is_sender_
               ? fmt::format(
                     "rdma sender sent={} released={} peer_consumed={} batches={} slots={}",
                     sent_,
                     released_,
                     peer_consumed_,
                     batches_.size(),
                     channel_->send_slots_available())
               : fmt::format(
                     "rdma receiver delivered={} consumed={} ring_pages={}", delivered_, consumed_, geom_.num_pages);
}

}  // namespace tt::tt_metal::distributed::host_transport
