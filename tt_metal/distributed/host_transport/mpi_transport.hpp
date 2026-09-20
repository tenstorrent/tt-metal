// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/distributed/host_transport/host_transport.hpp"

#include <tt-metalium/distributed_context.hpp>

#include <cstddef>
#include <deque>
#include <span>

namespace tt::tt_metal::distributed::host_transport {

// Two-sided point-to-point transport. Goes through DistributedContext, so it
// needs no MPI headers and runs over whatever the MPI build has underneath --
// UCX or libfabric over RDMA where available, plain TCP where not. Bandwidth is
// therefore the MPI build's, not this code's: the in-tree ULFM OpenMPI carries
// only the self/sm/tcp BTLs and plateaus near 2.2 GB/s, while an MPI with
// pml_ucx reaches the link.
//
// Deliberately point-to-point rather than MPI one-sided RMA: Rput completes on
// origin-buffer reuse rather than remote visibility, separate Rputs are
// unordered, and MPI_Cancel is illegal on an RMA request, so a timeout leaks the
// slot. Two-sided has none of those problems.
class MpiTransport final : public HostTransport {
public:
    explicit MpiTransport(const TransportParams& params);
    ~MpiTransport() override;

    bool can_send(uint32_t pages) const override;
    bool send(uint64_t first_page, uint32_t pages) override;
    uint64_t pages_released() const override { return released_; }
    uint64_t peer_consumed_pages() const override { return peer_consumed_; }

    uint64_t pages_delivered() const override { return delivered_; }
    bool post_credit(uint64_t consumed_pages) override;
    void set_consumed(uint64_t consumed_pages) override;

    void poll() override;
    std::string describe() const override;

private:
    // Caps the request bookkeeping; the relay's credit gate is the real limit.
    static constexpr size_t kMaxOutstanding = 512;

    std::span<std::byte> page_span(uint64_t page_index) const;
    void arm_credit_recv();
    void top_up_receives();
    bool retire_credit_send();

    RingGeometry geom_;
    bool is_sender_;
    std::byte* ring_;
    std::shared_ptr<multihost::DistributedContext> context_;
    multihost::Rank peer_;
    multihost::Tag payload_tag_;
    multihost::Tag credit_tag_;

    // Front-retired only, so the counts stay monotonic; the page a request
    // belongs to is its position, which is why none is stored.
    std::deque<multihost::RequestPtr> pending_send_;
    std::deque<multihost::RequestPtr> pending_recv_;
    uint64_t sent_ = 0;
    uint64_t released_ = 0;
    uint64_t delivered_ = 0;
    uint64_t posted_ = 0;
    uint64_t consumed_ = 0;

    multihost::RequestPtr credit_recv_;
    multihost::RequestPtr credit_send_;
    uint64_t credit_inbox_ = 0;
    uint64_t credit_outbox_ = 0;
    uint64_t peer_consumed_ = 0;
};

}  // namespace tt::tt_metal::distributed::host_transport
