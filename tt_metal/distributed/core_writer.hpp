// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include <tt-metalium/core_coord.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>  // ChipId

namespace tt {
class Cluster;
}
namespace tt::umd {
class IoWindow;
}

namespace tt::tt_metal::distributed {

// Host writes into one core's L1, for host-to-device rings and queues (H2D sockets, the host_rdma request service).
//
// Writes go through a TLB window of the writer's own where one reaches the range written: posted, write-only, on a
// static write VC, so writes to the core arrive in issue order and each is a plain store (no driver call, no wait for
// an ack). Otherwise (not Blackhole, a mock or emulated device, no TLB free, or the range is past the window) they go
// through Cluster::write_core. Callers that order host-memory stores against these writes (a count published after
// the data it covers) fence between them (tt_driver_atomics::sfence). Nothing is read through the window.
class CoreWriter {
public:
    // A writer for `virtual_core` of `chip`, for local L1 addresses below `end`. `l1_offset` is where the core's L1
    // starts in its NoC address space (0 for a Tensix or Ethernet core, the DRAM-L1 offset for a DRAM core); write()
    // takes local addresses and adds it. `owner` names the user in a warning when no window is free.
    CoreWriter(
        const tt::Cluster& cluster,
        ChipId chip,
        const tt_metal::CoreCoord& virtual_core,
        uint64_t end,
        uint64_t l1_offset = 0,
        std::string_view owner = "CoreWriter");
    ~CoreWriter();

    CoreWriter(const CoreWriter&) = delete;
    CoreWriter& operator=(const CoreWriter&) = delete;

    // Writes `num_bytes` at local address `addr`.
    void write(const void* data, uint32_t num_bytes, uint64_t addr) const;

    // Whether writes go through a window of the writer's own (rather than Cluster::write_core).
    bool windowed() const { return window_ != nullptr; }

private:
    const tt::Cluster& cluster_;
    ChipId chip_;
    tt_metal::CoreCoord virtual_core_;
    uint64_t l1_offset_;
    std::unique_ptr<tt::umd::IoWindow> window_;
};

}  // namespace tt::tt_metal::distributed
