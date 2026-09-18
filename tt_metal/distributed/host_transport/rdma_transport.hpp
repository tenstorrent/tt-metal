// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/distributed/host_transport/host_transport.hpp"
#include "tt_metal/distributed/host_transport/rdma_link.hpp"

#include <deque>
#include <memory>

namespace tt::tt_metal::distributed::host_transport {

// One-sided verbs transport: the peer's NIC lands the bytes, so the receiving CPU
// only reads a counter. Saturates the link but needs a RoCE device on both hosts.
class RdmaTransport final : public HostTransport {
public:
    explicit RdmaTransport(const TransportParams& params);
    ~RdmaTransport() override;

    bool can_send(uint32_t pages) const override;
    bool send(uint32_t first_page, uint32_t pages) override;
    uint64_t pages_released() const override { return released_; }
    uint64_t peer_consumed_pages() const override { return peer_consumed_; }

    uint64_t pages_delivered() const override { return delivered_; }
    bool post_credit(uint64_t consumed_pages) override;
    // One-sided: nothing to re-arm, but the watermark bounds the sanity check.
    void set_consumed(uint64_t consumed_pages) override { consumed_ = consumed_pages; }

    void poll() override;
    std::string describe() const override;

private:
    // Send-queue slots one batch can need: two payload runs plus the doorbell.
    static constexpr uint32_t kSlotsPerBatch = 3;
    static constexpr size_t kMaxOutstandingBatches = 64;

    struct Batch {
        uint64_t wr_id;
        uint32_t pages;
    };

    RingGeometry geom_;
    bool is_sender_;
    RdmaContext& ctx_;
    std::unique_ptr<RdmaChannel> channel_;
    std::unique_ptr<RdmaRegion> ring_region_;
    std::unique_ptr<RdmaRegion> slot_region_;  // credit on a sender, doorbell on a receiver
    std::unique_ptr<uint32_t> slot_;
    uint32_t max_batch_pages_;

    std::deque<Batch> batches_;
    uint64_t sent_ = 0;
    uint64_t released_ = 0;
    uint64_t delivered_ = 0;
    uint64_t consumed_ = 0;
    uint64_t peer_consumed_ = 0;
};

}  // namespace tt::tt_metal::distributed::host_transport
