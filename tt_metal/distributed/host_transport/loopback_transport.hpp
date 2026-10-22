// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/distributed/host_transport/host_transport.hpp"

#include <atomic>
#include <cstddef>

namespace tt::tt_metal::distributed::host_transport {

// Peer in the same process: the "network" hop is a memcpy from the sender's ring
// into the receiver's ring at the same page slot. The two ends find each other
// by tag_base in a process-wide registry, the way an MPI pair would by tag, so a
// link whose both ends are on this host (a host-bridged cable inside one mesh)
// runs through the same relay code as a cross-host one.
class LoopbackTransport final : public HostTransport {
public:
    explicit LoopbackTransport(const TransportParams& params);
    ~LoopbackTransport() override;

    bool can_send(uint32_t pages) const override;
    bool send(uint64_t first_page, uint32_t pages) override;
    uint64_t pages_released() const override { return released_.load(std::memory_order_acquire); }
    uint64_t peer_consumed_pages() const override { return peer_consumed_.load(std::memory_order_acquire); }

    uint64_t pages_delivered() const override { return delivered_.load(std::memory_order_acquire); }
    bool post_credit(uint64_t consumed_pages) override;
    void set_consumed(uint64_t consumed_pages) override { consumed_ = consumed_pages; }

    void poll() override {}
    std::string describe() const override;

private:
    RingGeometry geom_;
    bool is_sender_;
    std::byte* ring_;
    int tag_base_;
    std::atomic<LoopbackTransport*> peer_{nullptr};

    uint64_t sent_ = 0;
    uint64_t consumed_ = 0;
    std::atomic<uint64_t> released_{0};
    std::atomic<uint64_t> delivered_{0};
    std::atomic<uint64_t> peer_consumed_{0};
};

}  // namespace tt::tt_metal::distributed::host_transport
