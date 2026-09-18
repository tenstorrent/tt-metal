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

// Two-sided point-to-point transport, for deployments without RDMA. Goes through
// DistributedContext, so it needs no MPI headers and works over whatever MPI is
// configured -- including the in-tree ULFM build, which cannot do one-sided RDMA
// at all.
//
// Much slower than the RDMA backend, by however much the MPI build's own
// transport is slower: the in-tree ULFM OpenMPI carries only the self/sm/tcp
// BTLs, so cross-host pages go through the kernel TCP stack and plateau near
// 2.2 GB/s against RDMA's 11.8. That is the MPI build's ceiling, not this
// code's -- adding sender cores does not move it.
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
    bool send(uint32_t first_page, uint32_t pages) override;
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
